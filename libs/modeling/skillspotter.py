"""
SkillSpotter: pose-aware, multi-view skilled action detection and grading.

A single ConvTransformer (ActionFormer) backbone with three toggleable
SkillSpotter modules:
  1. Adaptive temporal suppression (learnable, scenario-aware NMS)
  2. Gated 3D body-pose fusion
  3. Bidirectional cross-view (ego <-> exo) attention

Shared building blocks live in blocks.py / necks.py / backbones.py /
loc_generators.py / crossview_modules.py.

Registered as meta-arch "skillspotter".
Used by: configs/egoexo4d_skillspotter_omnivore.yaml
"""
import os
import math
import json
import logging

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .models import register_meta_arch, make_backbone, make_neck, make_generator
from .blocks import MaskedConv1D, LayerNorm
from .losses import sigmoid_focal_loss
from ..utils import batched_nms_timestamps
from .crossview_modules import (
    build_crossview_modules,
    build_pose_stream,
    fuse_pose_gated,
    run_crossview_fpn,
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Submodules
# -----------------------------------------------------------------------------
# -----------------------------
# Learnable temporal suppression
# -----------------------------
class LearnableTemporalSuppression(nn.Module):
    """
    Differentiable soft suppression module.

    Intended use:
      - TRAIN: auxiliary loss on a small set of top-M candidates (per video)
      - TEST: score reweighting before (hard/soft) NMS

    It predicts a per-candidate suppression radius from candidate features and applies
    soft down-weighting based on pairwise temporal distance and relative score.

    Notes:
      - Uses pairwise O(M^2) on M candidates; keep M small (e.g., 256-1024).
      - Radii are in the same units as the timestamps you pass (grid units or seconds).
    """
    def __init__(self, feat_dim: int, num_scenarios: int = 1, hidden_dim: int = 128):
        super().__init__()

        self.scenario_radius = nn.Embedding(num_scenarios, 1)
        nn.init.zeros_(self.scenario_radius.weight)

        self.radius_predictor = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),  # positive radius
        )
        self.suppression_temp = nn.Parameter(torch.tensor(1.0))

    @torch.no_grad()
    def predict_radii(
            self,
            features: torch.Tensor,  # (M, D)
            scenario_id: torch.Tensor,  # scalar or shape (1,)
    ) -> torch.Tensor:
        """
        Returns per-candidate radius (M,). Same computation as used in forward.
        """
        if scenario_id.ndim == 0:
            scenario_id = scenario_id.view(1)
        scenario_id = scenario_id.long().to(features.device)

        radii = self.radius_predictor(features).squeeze(-1)  # (M,)
        radii = radii + self.scenario_radius(scenario_id).view(1)
        radii = torch.clamp(radii, min=1e-6)
        return radii

    def forward(
        self,
        timestamps: torch.Tensor,   # (M,)
        scores: torch.Tensor,       # (M,)
        features: torch.Tensor,     # (M, D)
        scenario_id: torch.Tensor,  # scalar LongTensor or shape (1,)
        class_ids: torch.Tensor = None,  # (M,) optional
        class_aware: bool = False,
    ) -> torch.Tensor:
        """
        Returns adjusted scores of shape (M,).
        """
        if timestamps.numel() == 0:
            return scores

        if scenario_id.ndim == 0:
            scenario_id = scenario_id.view(1)
        scenario_id = scenario_id.long().to(features.device)

        # # --- ORIGINAL: learned per-detection radii ---
        radii = self.radius_predictor(features).squeeze(-1)  # (M,)
        radii = radii + self.scenario_radius(scenario_id).view(1)  # (M,)
        radii = torch.clamp(radii, min=1e-6)

        # Pairwise temporal distances (M,M)
        t = timestamps.view(-1, 1)
        t_diff = torch.abs(t - t.transpose(0, 1))  # (M,M)

        # Only higher-score suppresses lower-score (mask)
        s = scores.view(-1, 1)
        higher = (s > s.transpose(0, 1)).float()  # (M,M)

        # Optional class-aware suppression: only suppress within same class/group
        if class_aware and (class_ids is not None):
            class_ids = class_ids.view(-1, 1)
            same_class = (class_ids == class_ids.transpose(0, 1)).float()
            higher = higher * same_class

        # i suppresses j based on i's radius
        r_i = radii.view(-1, 1)  # (M,1)
        sup_ij = torch.exp(-t_diff / r_i) * higher  # (M,M)
        sup_ij.fill_diagonal_(0.0)

        total_sup = sup_ij.sum(dim=0)  # (M,)
        temp = torch.clamp(self.suppression_temp, min=1e-3)
        adjusted = scores * torch.exp(-total_sup / temp)
        return adjusted


# -----------------------------
# Classification head (returns features too)
# -----------------------------
class RegTransformerClsHead(nn.Module):
    """
    1D Conv heads for classification

    Modification:
      - returns both logits and the last feature map (for learnable suppression)
    """
    def __init__(
        self,
        input_dim,
        feat_dim,
        num_classes,
        prior_prob=0.01,
        num_layers=3,
        kernel_size=3,
        act_layer=nn.ReLU,
        with_ln=False,
        empty_cls=None,
    ):
        super().__init__()
        empty_cls = empty_cls or []
        self.act = act_layer()

        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()
        for idx in range(num_layers - 1):
            in_dim = input_dim if idx == 0 else feat_dim
            out_dim = feat_dim
            self.head.append(
                MaskedConv1D(
                    in_dim, out_dim, kernel_size,
                    stride=1,
                    padding=kernel_size // 2,
                    bias=(not with_ln)
                )
            )
            self.norm.append(LayerNorm(out_dim) if with_ln else nn.Identity())

        self.cls_head = MaskedConv1D(
            feat_dim, num_classes, kernel_size,
            stride=1, padding=kernel_size // 2
        )

        if prior_prob > 0:
            bias_value = -(math.log((1 - prior_prob) / prior_prob))
            torch.nn.init.constant_(self.cls_head.conv.bias, bias_value)

        if len(empty_cls) > 0:
            bias_value = -(math.log((1 - 1e-6) / 1e-6))
            for idx in empty_cls:
                torch.nn.init.constant_(self.cls_head.conv.bias[idx], bias_value)

    def forward(self, fpn_feats, fpn_masks):
        assert len(fpn_feats) == len(fpn_masks)

        out_logits = tuple()
        out_feats = tuple()
        for _, (cur_feat, cur_mask) in enumerate(zip(fpn_feats, fpn_masks)):
            cur_out = cur_feat
            for idx in range(len(self.head)):
                cur_out, _ = self.head[idx](cur_out, cur_mask)
                cur_out = self.act(self.norm[idx](cur_out))

            # cur_out is the last feature map before classifier conv
            cur_logits, _ = self.cls_head(cur_out, cur_mask)
            out_logits += (cur_logits,)
            out_feats += (cur_out,)

        return out_logits, out_feats


# -----------------------------------------------------------------------------
# Helper: separate ConvTransformer exo backbone + FPN neck (non-shared case)
# -----------------------------------------------------------------------------
def _build_convtransformer_exo_backbone_neck(
    input_dim, embd_dim, n_head, embd_kernel_size, max_seq_len,
    backbone_arch, mha_win_size, scale_factor, embd_with_ln,
    train_dropout, train_droppath, use_abs_pe, use_rel_pe,
    fpn_type, fpn_dim, fpn_start_level, fpn_with_ln,
):
    """Build a separate ConvTransformer exo backbone + FPN neck (for non-shared case)."""
    exo_backbone = make_backbone(
        'convTransformer',
        **{
            'n_in': input_dim,
            'n_embd': embd_dim,
            'n_head': n_head,
            'n_embd_ks': embd_kernel_size,
            'max_len': max_seq_len,
            'arch': backbone_arch,
            'mha_win_size': mha_win_size,
            'scale_factor': scale_factor,
            'with_ln': embd_with_ln,
            'attn_pdrop': 0.0,
            'proj_pdrop': train_dropout,
            'path_pdrop': train_droppath,
            'use_abs_pe': use_abs_pe,
            'use_rel_pe': use_rel_pe,
        }
    )

    if isinstance(embd_dim, (list, tuple)):
        embd_dim_scalar = sum(embd_dim)
    else:
        embd_dim_scalar = embd_dim

    exo_neck = make_neck(
        fpn_type,
        **{
            'in_channels': [embd_dim_scalar] * (backbone_arch[-1] + 1),
            'out_channel': fpn_dim,
            'scale_factor': scale_factor,
            'start_level': fpn_start_level,
            'with_ln': fpn_with_ln,
        }
    )
    return exo_backbone, exo_neck



@register_meta_arch("skillspotter")
class SkillSpotter(nn.Module):
    """
    SkillSpotter: pose-aware, multi-view skilled action detection and grading.

    A ConvTransformer (ActionFormer) backbone with three toggleable modules:
      * adaptive temporal suppression (learnable, scenario-aware NMS)
      * gated 3D body-pose fusion
      * bidirectional cross-view (ego <-> exo) attention

    For single-view settings (Ego or Exos) the cross-view branch is bypassed
    and the model behaves as a standard adaptive detector.
    """

    def __init__(
        self,
        # Standard backbone/head params
        fpn_type,
        backbone_arch,
        scale_factor,
        input_dim,
        max_seq_len,
        max_buffer_len_factor,
        n_head,
        n_mha_win_size,
        embd_kernel_size,
        embd_dim,
        embd_with_ln,
        fpn_dim,
        fpn_with_ln,
        fpn_start_level,
        head_dim,
        regression_range,
        head_num_layers,
        head_kernel_size,
        head_with_ln,
        use_abs_pe,
        use_rel_pe,
        num_classes,
        train_cfg,
        test_cfg,
        # Cross-view parameters
        cross_view_n_head=4,
        cross_view_attn_pdrop=0.0,
        cross_view_proj_pdrop=0.1,
        cross_view_gate_init=-2.0,
        cross_view_fusion_dropout=0.1,
        cross_view_fusion_gate_init=-1.0,
        cross_view_shared_backbone=True,
        # SkillSpotter module toggles
        learnable_suppression=False,
        build_crossview=False,
        use_pose=False,
        pose_input_dim=None,
        pose_backbone_type="conv",
        pose_embd_dim=256,
        pose_embd_with_ln=True,
        pose_neck_type="fpn",
        pose_gate_init=-2.0,
        pose_fusion_dropout=0.1,
        # Accepted for config compatibility
        backbone_type='convTransformer',
    ):
        super().__init__()

        # ── Core attributes ──
        self.fpn_strides = [
            scale_factor ** i
            for i in range(fpn_start_level, backbone_arch[-1] + 1)
        ]
        self.reg_range = regression_range
        assert len(self.fpn_strides) == len(self.reg_range)
        self.scale_factor = scale_factor
        self.num_classes = num_classes

        # MHA window size and max_div_factor (ConvTransformer uses attention windows)
        self.max_seq_len = max_seq_len
        if isinstance(n_mha_win_size, int):
            self.mha_win_size = [n_mha_win_size] * (1 + backbone_arch[-1])
        else:
            assert len(n_mha_win_size) == (1 + backbone_arch[-1])
            self.mha_win_size = n_mha_win_size
        max_div_factor = 1
        for s, w in zip(self.fpn_strides, self.mha_win_size):
            stride = s * (w // 2) * 2 if w > 1 else s
            assert max_seq_len % stride == 0
            if max_div_factor < stride:
                max_div_factor = stride
        self.max_div_factor = max_div_factor

        # Train config
        self.train_center_sample = train_cfg['center_sample']
        assert self.train_center_sample in ['radius', 'none']
        self.train_center_sample_radius = train_cfg['center_sample_radius']
        self.train_loss_weight = train_cfg['loss_weight']
        self.train_cls_prior_prob = train_cfg['cls_prior_prob']
        self.train_dropout = train_cfg['dropout']
        self.train_droppath = train_cfg['droppath']
        self.train_label_smoothing = train_cfg['label_smoothing']

        # Test config
        self.test_pre_nms_thresh = test_cfg['pre_nms_thresh']
        self.test_pre_nms_topk = test_cfg['pre_nms_topk']
        self.test_min_score = test_cfg['min_score']
        self.test_max_t_num = test_cfg['max_t_num']
        self.test_nms_method = test_cfg['nms_method']
        self.test_nms_sigma = test_cfg['nms_sigma']
        assert self.test_nms_method in ['soft', 'hard', 'none']
        self.test_multiclass_nms = test_cfg['multiclass_nms']
        self.test_l1_radius = test_cfg['l1_nms_radius']

        # ── ConvTransformer Backbone ──
        assert backbone_type in ['convTransformer', 'conv']
        if backbone_type == 'convTransformer':
            self.backbone = make_backbone(
                'convTransformer',
                **{
                    'n_in': input_dim,
                    'n_embd': embd_dim,
                    'n_head': n_head,
                    'n_embd_ks': embd_kernel_size,
                    'max_len': max_seq_len,
                    'arch': backbone_arch,
                    'mha_win_size': self.mha_win_size,
                    'scale_factor': scale_factor,
                    'with_ln': embd_with_ln,
                    'attn_pdrop': 0.0,
                    'proj_pdrop': self.train_dropout,
                    'path_pdrop': self.train_droppath,
                    'use_abs_pe': use_abs_pe,
                    'use_rel_pe': use_rel_pe,
                }
            )
        else:
            self.backbone = make_backbone(
                'conv',
                **{
                    'n_in': input_dim,
                    'n_embd': embd_dim,
                    'n_embd_ks': embd_kernel_size,
                    'arch': backbone_arch,
                    'scale_factor': scale_factor,
                    'with_ln': embd_with_ln,
                }
            )

        if isinstance(embd_dim, (list, tuple)):
            embd_dim_scalar = sum(embd_dim)
        else:
            embd_dim_scalar = embd_dim

        # ── FPN neck ──
        assert fpn_type in ['fpn', 'identity']
        self.neck = make_neck(
            fpn_type,
            **{
                'in_channels': [embd_dim_scalar] * (backbone_arch[-1] + 1),
                'out_channel': fpn_dim,
                'scale_factor': scale_factor,
                'start_level': fpn_start_level,
                'with_ln': fpn_with_ln,
            }
        )

        # ── Point generator ──
        self.point_generator = make_generator(
            'point',
            **{
                'max_seq_len': max_seq_len * max_buffer_len_factor,
                'fpn_strides': self.fpn_strides,
                'regression_range': self.reg_range,
            }
        )

        # ── Classification head (returns logits + features for suppression) ──
        self.cls_head = RegTransformerClsHead(
            fpn_dim, head_dim, self.num_classes,
            kernel_size=head_kernel_size,
            prior_prob=self.train_cls_prior_prob,
            with_ln=head_with_ln,
            num_layers=head_num_layers,
            empty_cls=train_cfg['head_empty_cls'],
        )

        # Loss normalizer
        self.loss_normalizer = train_cfg['init_loss_norm']
        self.loss_normalizer_momentum = 0.9

        # ── Adaptive Suppression (toggleable via model.learnable_suppression) ──
        self.use_learnable_suppression = bool(learnable_suppression)
        self.suppress_topk_train = int(train_cfg.get('suppress_topk_train', 512))
        self.suppress_topk_test = int(test_cfg.get('suppress_topk_test', 2048))
        self.suppress_loss_weight = float(train_cfg.get('suppress_loss_weight', 1.0))
        self.suppress_target_sigma = float(train_cfg.get('suppress_target_sigma', 1.0))
        self.num_scenarios = int(train_cfg.get('num_scenarios', 1))
        self.suppress_class_aware = bool(train_cfg.get('suppress_class_aware', True))
        self.suppress_class_aware_test = bool(
            test_cfg.get('suppress_class_aware', self.suppress_class_aware)
        )

        self.suppressor = None
        if self.use_learnable_suppression:
            self.suppressor = LearnableTemporalSuppression(
                feat_dim=head_dim,
                num_scenarios=self.num_scenarios,
                hidden_dim=int(train_cfg.get('suppress_hidden_dim', 128)),
            )

        # Debug suppression dump
        self.debug_suppress_dump = bool(test_cfg.get('debug_suppress_dump', False))
        self.debug_suppress_dump_topk = int(
            test_cfg.get('debug_suppress_dump_topk', 5000)
        )
        self.debug_suppress_dump_file = None

        if self.debug_suppress_dump:
            base_path = test_cfg.get('debug_suppress_dump_file', None)
            if base_path is None:
                raise ValueError(
                    "test_cfg.debug_suppress_dump_file must be set "
                    "when debug_suppress_dump=True"
                )
            rank = int(os.environ.get("RANK", "0"))
            root, ext = os.path.splitext(base_path)
            if ext == "":
                ext = ".jsonl"
            self.debug_suppress_dump_file = f"{root}.rank{rank}{ext}"
            if os.path.exists(self.debug_suppress_dump_file):
                os.remove(self.debug_suppress_dump_file)

        # ── Cross-View Attention (built only when dataset.cross_view=True) ──
        self.build_crossview = bool(build_crossview)
        self.cross_view_shared_backbone = bool(cross_view_shared_backbone)
        self.num_fpn_levels = backbone_arch[-1] + 1 - fpn_start_level

        # Ablation flags
        self.ablate_crossview_attn = bool(
            test_cfg.get('ablate_crossview_attn', False)
        )
        self.ablate_crossview_fusion = bool(
            test_cfg.get('ablate_crossview_fusion', False)
        )
        self.ablate_pose_gates = bool(
            test_cfg.get('ablate_pose_gates', False)
        )

        if self.ablate_crossview_attn or self.ablate_crossview_fusion or self.ablate_pose_gates:
            logger.warning(
                f"Gate ablation active: attn={self.ablate_crossview_attn}, "
                f"fusion={self.ablate_crossview_fusion}, "
                f"pose={self.ablate_pose_gates}"
            )

        if self.build_crossview:
            # Exo backbone (ConvTransformer, separate weights if not shared)
            if not self.cross_view_shared_backbone:
                self.exo_backbone, self.exo_neck = _build_convtransformer_exo_backbone_neck(
                    input_dim=input_dim,
                    embd_dim=embd_dim,
                    n_head=n_head,
                    embd_kernel_size=embd_kernel_size,
                    max_seq_len=max_seq_len,
                    backbone_arch=backbone_arch,
                    mha_win_size=self.mha_win_size,
                    scale_factor=scale_factor,
                    embd_with_ln=embd_with_ln,
                    train_dropout=self.train_dropout,
                    train_droppath=self.train_droppath,
                    use_abs_pe=use_abs_pe,
                    use_rel_pe=use_rel_pe,
                    fpn_type=fpn_type,
                    fpn_dim=fpn_dim,
                    fpn_start_level=fpn_start_level,
                    fpn_with_ln=fpn_with_ln,
                )

            # Cross-view attention + fusion modules (per FPN level)
            self.cross_view_attn, self.cross_view_fusion = build_crossview_modules(
                num_fpn_levels=self.num_fpn_levels,
                fpn_dim=fpn_dim,
                cross_view_n_head=cross_view_n_head,
                cross_view_attn_pdrop=cross_view_attn_pdrop,
                cross_view_proj_pdrop=cross_view_proj_pdrop,
                cross_view_gate_init=cross_view_gate_init,
                cross_view_fusion_dropout=cross_view_fusion_dropout,
                cross_view_fusion_gate_init=cross_view_fusion_gate_init,
            )
        else:
            self.cross_view_attn = None
            self.cross_view_fusion = None

        # ── Pose Fusion (toggleable via model.use_pose) ──
        self.use_pose = bool(use_pose) and \
            pose_input_dim is not None and pose_input_dim > 0
        if self.use_pose:
            self.pose_input_dim = int(pose_input_dim)
            self.pose_fusion_dropout = float(pose_fusion_dropout)

            (
                self.pose_backbone,
                self.pose_neck,
                self.pose_fusion_proj,
                self.pose_gates,
                self.pose_fusion_drop,
            ) = build_pose_stream(
                pose_input_dim=self.pose_input_dim,
                pose_backbone_type=pose_backbone_type,
                pose_embd_dim=pose_embd_dim,
                pose_embd_with_ln=pose_embd_with_ln,
                pose_neck_type=pose_neck_type,
                n_head=n_head,
                embd_kernel_size=embd_kernel_size,
                max_seq_len=max_seq_len,
                backbone_arch=backbone_arch,
                n_mha_win_size=n_mha_win_size,
                scale_factor=scale_factor,
                train_dropout=self.train_dropout,
                train_droppath=self.train_droppath,
                use_abs_pe=use_abs_pe,
                use_rel_pe=use_rel_pe,
                fpn_dim=fpn_dim,
                fpn_start_level=fpn_start_level,
                fpn_with_ln=fpn_with_ln,
                pose_gate_init=pose_gate_init,
                pose_fusion_dropout=pose_fusion_dropout,
                num_fpn_levels=self.num_fpn_levels,
            )

    @property
    def device(self):
        return list(set(p.device for p in self.parameters()))[0]

    def _append_suppress_dump_rows(self, rows):
        if (not self.debug_suppress_dump) or (self.debug_suppress_dump_file is None) or (len(rows) == 0):
            return
        with open(self.debug_suppress_dump_file, "a") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    # -----------------------------
    # Forward

    def _has_cross_view_data(self, video_list):
        """Check if this batch has cross-view data (feats_ego and feats_exo)."""
        return "feats_ego" in video_list[0] and "feats_exo" in video_list[0]

    @torch.no_grad()
    def preprocessing(self, video_list, padding_val=0.0):
        """
        Preprocess video features, and optionally ego/exo/pose features.
        """
        feats = [x["feats"] for x in video_list]
        feats_lens = torch.as_tensor([feat.shape[-1] for feat in feats])
        max_len = feats_lens.max(0).values.item()

        has_cross_view = self._has_cross_view_data(video_list)
        has_pose = self.use_pose and "pose_feats" in video_list[0]

        if self.training:
            assert max_len <= self.max_seq_len
            max_len = self.max_seq_len
        else:
            assert len(video_list) == 1
            if max_len <= self.max_seq_len:
                max_len = self.max_seq_len
            else:
                stride = self.max_div_factor
                max_len = (max_len + (stride - 1)) // stride * stride

        B = len(video_list)
        C = feats[0].shape[0]

        # Main features (ego for cross-view, or standard single-view)
        batched_inputs = feats[0].new_full([B, C, max_len], padding_val)
        for i, feat in enumerate(feats):
            cur_len = min(int(feat.shape[-1]), max_len)
            batched_inputs[i, :, :cur_len].copy_(feat[:, :cur_len])

        batched_masks = (torch.arange(max_len)[None, :] < feats_lens[:, None]).unsqueeze(1)

        batched_inputs = batched_inputs.to(self.device)
        batched_masks = batched_masks.to(self.device)

        result = {
            "inputs": batched_inputs,
            "masks": batched_masks,
        }

        # Cross-view: exo features
        if has_cross_view:
            exo_feats = [x["feats_exo"] for x in video_list]
            C_exo = exo_feats[0].shape[0]
            batched_exo = feats[0].new_full([B, C_exo, max_len], padding_val)
            for i, exo_f in enumerate(exo_feats):
                cur_len = min(int(exo_f.shape[-1]), max_len)
                batched_exo[i, :, :cur_len].copy_(exo_f[:, :cur_len])
            result["exo_inputs"] = batched_exo.to(self.device)

        # Pose features
        if has_pose:
            batched_pose = feats[0].new_zeros([B, self.pose_input_dim, max_len])
            batched_pose_valid = torch.zeros([B, 1, max_len], dtype=torch.bool)

            for i, item in enumerate(video_list):
                pose_feat = item["pose_feats"]
                pose_valid = item["pose_valid"]

                if pose_feat.ndim == 1:
                    pose_feat = pose_feat.unsqueeze(0)

                pose_ch = min(int(pose_feat.shape[0]), self.pose_input_dim)
                cur_len = min(int(feats[i].shape[-1]), max_len)
                pose_len = min(int(pose_feat.shape[-1]), cur_len)
                batched_pose[i, :pose_ch, :pose_len].copy_(pose_feat[:pose_ch, :pose_len])

                pose_valid = pose_valid.bool()
                valid_len = min(int(pose_valid.shape[0]), pose_len)
                batched_pose_valid[i, 0, :valid_len] = pose_valid[:valid_len]

            result["pose_inputs"] = batched_pose.to(self.device)
            result["pose_valid"] = batched_pose_valid.to(self.device)

        return result

    def forward(self, video_list, return_loss=None):
        is_training = return_loss if return_loss is not None else self.training

        prep = self.preprocessing(video_list)
        batched_inputs = prep["inputs"]
        batched_masks = prep["masks"]

        has_cross_view = "exo_inputs" in prep
        has_pose = "pose_inputs" in prep

        # ── Ego stream (or standard single-view stream) ──
        ego_feats, ego_masks = self.backbone(batched_inputs, batched_masks)
        ego_fpn_feats, ego_fpn_masks = self.neck(ego_feats, ego_masks)

        if has_cross_view:
            # ── Exo stream ──
            if self.cross_view_shared_backbone:
                exo_feats, exo_masks = self.backbone(prep["exo_inputs"], batched_masks)
                exo_fpn_feats, exo_fpn_masks = self.neck(exo_feats, exo_masks)
            else:
                exo_feats, exo_masks = self.exo_backbone(prep["exo_inputs"], batched_masks)
                exo_fpn_feats, exo_fpn_masks = self.exo_neck(exo_feats, exo_masks)

            # ── Cross-view attention + fusion at each FPN level ──
            fpn_feats, fpn_masks = run_crossview_fpn(
                ego_fpn_feats, ego_fpn_masks,
                exo_fpn_feats, exo_fpn_masks,
                self.cross_view_attn, self.cross_view_fusion,
                self.num_fpn_levels,
                ablate_attn=self.ablate_crossview_attn,
                ablate_fusion=self.ablate_crossview_fusion,
            )
        else:
            # Single-view: no cross-view attention, pass through
            fpn_feats = ego_fpn_feats
            fpn_masks = ego_fpn_masks

        # ── Optional pose fusion ──
        if has_pose:
            pose_input_mask = batched_masks & prep["pose_valid"]
            pose_feats, pose_masks = self.pose_backbone(prep["pose_inputs"], pose_input_mask)
            pose_fpn_feats, pose_fpn_masks = self.pose_neck(pose_feats, pose_masks)

            fpn_feats, fpn_masks = fuse_pose_gated(
                fpn_feats, fpn_masks, pose_fpn_feats, pose_fpn_masks,
                self.pose_fusion_proj, self.pose_gates, self.pose_fusion_drop,
                ablate=self.ablate_pose_gates,
            )

        # ── Detection heads (identical to adaptive) ──
        points = self.point_generator(fpn_feats)
        out_cls_logits, out_cls_feats = self.cls_head(fpn_feats, fpn_masks)

        out_cls_logits = [x.permute(0, 2, 1) for x in out_cls_logits]
        out_cls_feats = [x.permute(0, 2, 1) for x in out_cls_feats]
        fpn_masks_squeezed = [x.squeeze(1) for x in fpn_masks]

        if is_training:
            assert video_list[0]["reg_points"] is not None
            assert video_list[0]["labels"] is not None
            gt_reg_points = [x["reg_points"].to(self.device) for x in video_list]
            gt_labels = [x["labels"].to(self.device) for x in video_list]

            gt_cls_labels = self.label_points(points, gt_reg_points, gt_labels)

            losses = self.losses(
                fpn_masks=fpn_masks_squeezed,
                out_cls_logits=out_cls_logits,
                gt_cls_labels=gt_cls_labels,
                points=points,
                gt_reg_points=gt_reg_points,
                out_cls_feats=out_cls_feats,
                video_list=video_list,
            )
            return losses

        results = self.inference(
            video_list, points, fpn_masks_squeezed, out_cls_logits, out_cls_feats
        )
        return results


    @torch.no_grad()
    def label_points(self, points, gt_reg_points, gt_labels):
        concat_points = torch.cat(points, dim=0)
        gt_cls = []
        for gt_reg_point, gt_label in zip(gt_reg_points, gt_labels):
            cls_targets = self.label_points_single_video(concat_points, gt_reg_point, gt_label)
            gt_cls.append(cls_targets)
        return gt_cls

    @torch.no_grad()
    def label_points_single_video(self, concat_points, gt_reg_point, gt_label):
        num_pts = concat_points.shape[0]
        num_gts = gt_reg_point.shape[0]
        if num_gts == 0:
            return gt_reg_point.new_full((num_pts, self.num_classes), 0)

        gt_t = gt_reg_point.view(-1)
        t_points = concat_points[:, 0]
        dists = torch.abs(t_points[:, None] - gt_t[None, :])

        if self.train_center_sample == "radius":
            radius = self.train_center_sample_radius
            inside_gt_seg_mask = dists <= (concat_points[:, 3, None] * radius)
        else:
            inside_gt_seg_mask = torch.ones_like(dists, dtype=torch.bool)

        reg_min = concat_points[:, 1, None]
        reg_max = concat_points[:, 2, None]
        inside_regress_range = (dists >= reg_min) & (dists <= reg_max)

        score = dists.clone()
        score.masked_fill_(inside_gt_seg_mask == 0, float("inf"))
        score.masked_fill_(inside_regress_range == 0, float("inf"))
        min_dist, _ = score.min(dim=1)

        min_len_mask = ((dists <= (min_dist[:, None] + 1e-6)) & (score < float("inf"))).to(dists.dtype)

        gt_label_one_hot = F.one_hot(gt_label.view(-1), self.num_classes).to(dists.dtype)
        cls_targets = min_len_mask @ gt_label_one_hot
        cls_targets.clamp_(min=0.0, max=1.0)
        return cls_targets

    # -----------------------------
    # Losses (+ optional suppression loss)
    # -----------------------------
    def losses(
        self,
        fpn_masks,
        out_cls_logits,
        gt_cls_labels,
        points,
        gt_reg_points,
        out_cls_feats,
        video_list,
    ):
        valid_mask = torch.cat(fpn_masks, dim=1)  # (B,FT)
        pred_logits_flat = torch.cat(out_cls_logits, dim=1)  # (B,FT,C)
        gt_cls = torch.stack(gt_cls_labels)  # (B,FT,C)

        pos_mask = torch.logical_and((gt_cls.sum(-1) > 0), valid_mask)
        num_pos = pos_mask.sum().item()
        self.loss_normalizer = (
            self.loss_normalizer_momentum * self.loss_normalizer
            + (1 - self.loss_normalizer_momentum) * max(num_pos, 1)
        )

        # ---- focal cls loss
        if self.train_label_smoothing > 0:
            gt_target = gt_cls * (1 - self.train_label_smoothing) + self.train_label_smoothing / self.num_classes
        else:
            gt_target = gt_cls

        cls_loss = sigmoid_focal_loss(
            pred_logits_flat[valid_mask],
            gt_target[valid_mask],
            reduction="sum",
        )
        cls_loss /= self.loss_normalizer

        # ---- regression loss (CDF/EMD-style) exactly as your current code
        yhat = pred_logits_flat.sigmoid() * valid_mask.unsqueeze(-1)  # (B,FT,C)

        sigma = getattr(self, "kwak_gauss_sigma", 2.0)
        win = int(getattr(self, "kwak_gauss_window", 19))
        if win % 2 == 0:
            win += 1
        t = torch.arange(win, device=gt_cls.device, dtype=gt_cls.dtype) - (win // 2)
        gk = torch.exp(-(t**2) / (2 * (sigma**2)))
        gk = (gk / gk.sum()).view(1, 1, win)

        B, FT, C = gt_cls.shape
        y_gt_bc = (gt_cls * valid_mask.unsqueeze(-1)).permute(0, 2, 1).contiguous().view(B * C, 1, FT)
        pad = win // 2
        y_gt_blur = F.conv1d(F.pad(y_gt_bc, (pad, pad), mode="replicate"), gk)
        y_gt_blur = y_gt_blur.view(B, C, FT).permute(0, 2, 1).contiguous()

        eps = 1e-8
        y_gt_sum = (y_gt_blur * valid_mask.unsqueeze(-1)).sum(dim=1, keepdim=True) + eps
        yhat_sum = (yhat * valid_mask.unsqueeze(-1)).sum(dim=1, keepdim=True) + eps
        y_gt_norm = y_gt_blur / y_gt_sum
        yhat_norm = yhat / yhat_sum

        t_idx_concat = torch.cat([p[:, 0] for p in points], dim=0)  # (FT,)
        sort_idx = torch.argsort(t_idx_concat)

        gather_idx_bc = sort_idx.view(1, -1, 1).expand(B, -1, C)
        y_gt_norm = torch.gather(y_gt_norm, 1, gather_idx_bc)
        yhat_norm = torch.gather(yhat_norm, 1, gather_idx_bc)
        valid_mask_sorted = torch.gather(valid_mask, 1, sort_idx.view(1, -1).expand(B, -1))

        y_gt_cum = torch.cumsum(y_gt_norm, dim=1)
        yhat_cum = torch.cumsum(yhat_norm, dim=1)
        diff2 = (y_gt_cum - yhat_cum).pow(2) * valid_mask_sorted.unsqueeze(-1)

        reg_loss = diff2.sum() / self.loss_normalizer

        loss_weight = self.train_loss_weight if self.train_loss_weight > 0 else cls_loss.detach() / max(reg_loss.item(), 0.01)
        final_loss = cls_loss + reg_loss * loss_weight

        # ---- optional learnable suppression aux loss
        suppress_loss = pred_logits_flat.new_tensor(0.0)
        if self.use_learnable_suppression and (self.suppressor is not None) and (self.suppress_loss_weight > 0):
            suppress_loss = self._suppression_aux_loss(
                pred_logits_flat=pred_logits_flat,        # (B,FT,C)
                pred_feats_flat=torch.cat(out_cls_feats, dim=1),  # (B,FT,D)
                valid_mask=valid_mask,                    # (B,FT)
                t_idx_concat=t_idx_concat,                # (FT,) grid indices
                gt_reg_points=gt_reg_points,              # list[B] of (N,)
                video_list=video_list,
            )
            final_loss = final_loss + self.suppress_loss_weight * suppress_loss

        return {
            "cls_loss": cls_loss,
            "reg_loss": reg_loss,
            "suppress_loss": suppress_loss,
            "final_loss": final_loss,
        }

    def _suppression_aux_loss(
        self,
        pred_logits_flat: torch.Tensor,   # (B,FT,C)
        pred_feats_flat: torch.Tensor,    # (B,FT,D)
        valid_mask: torch.Tensor,         # (B,FT)
        t_idx_concat: torch.Tensor,       # (FT,)
        gt_reg_points,                    # list of tensors (N,)
        video_list,
    ) -> torch.Tensor:
        """
        Train suppression on top-M candidates:
          - score per time = max over classes of sigmoid(logits)
          - select top-M within valid_mask
          - apply suppressor -> adjusted scores
          - match to gaussian target around GT timestamps (in grid units)
        """
        B, FT, C = pred_logits_flat.shape
        device = pred_logits_flat.device
        total = pred_logits_flat.new_tensor(0.0)

        # time indices in grid units (FT,)
        t_grid = t_idx_concat.to(device)

        for b in range(B):
            vm = valid_mask[b]  # (FT,)
            if vm.sum() == 0:
                continue

            # class-agnostic per-time score
            prob = pred_logits_flat[b].sigmoid()  # (FT,C)
            s_t, _ = prob.max(dim=-1)             # (FT,)
            s_t = s_t * vm.float()

            # predicted class id per time (for class-aware suppression)
            cls_t = torch.argmax(prob, dim=-1)    # (FT,)

            # top-M candidate selection (indices are non-diff; gradients still flow through gathered scores/features)
            M = min(self.suppress_topk_train, int(vm.sum().item()))
            if M <= 0:
                continue
            vals, idx = torch.topk(s_t, k=M, largest=True, sorted=False)  # (M,)

            cand_t = t_grid[idx]                # (M,)
            cand_s = vals                       # (M,)
            cand_f = pred_feats_flat[b, idx, :] # (M,D)
            cand_c = cls_t[idx]                 # (M,)

            # scenario id (optional) from video_list[b].get("scenario_id", 0)
            scen = int(video_list[b]["scenario_id"])
            scen = torch.tensor(scen, device=device, dtype=torch.long)

            # suppress scores
            cand_s_adj = self.suppressor(
                timestamps=cand_t,
                scores=cand_s,
                features=cand_f,
                scenario_id=scen,
                class_ids=cand_c,
                class_aware=self.suppress_class_aware,
            )  # (M,)

            # gaussian target around GT in grid units
            gt = gt_reg_points[b].view(-1).to(device)  # (N,)
            if gt.numel() == 0:
                y = torch.zeros_like(cand_s_adj)
            else:
                # y_i = max_j exp(-(t_i-gt_j)^2/(2*sigma^2))
                sigma = float(self.suppress_target_sigma)
                d2 = (cand_t.view(-1, 1) - gt.view(1, -1)).pow(2)  # (M,N)
                y = torch.exp(-d2 / (2.0 * (sigma ** 2))).max(dim=1).values  # (M,)

            # loss on adjusted scores (BCE)
            cand_s_adj = torch.clamp(cand_s_adj, 1e-6, 1 - 1e-6)
            cand_logits = torch.log(cand_s_adj) - torch.log1p(-cand_s_adj)  # logit(p)
            total = total + F.binary_cross_entropy_with_logits(cand_logits, y, reduction="mean")

        return total / max(B, 1)

    # -----------------------------
    # Inference
    # -----------------------------
    @torch.no_grad()
    def inference(self, video_list, points, fpn_masks, out_cls_logits, out_cls_feats):
        results = []

        vid_idxs = [x["video_id"] for x in video_list]
        vid_orig_files = [x["orig_video_file"] for x in video_list]
        vid_fps = [x["fps"] for x in video_list]
        vid_lens = [x["duration"] for x in video_list]
        vid_ft_stride = [x["feat_stride"] for x in video_list]
        vid_ft_nframes = [x["feat_num_frames"] for x in video_list]

        for idx, (vidx, vid_orig_file, fps, vlen, stride, nframes) in enumerate(
            zip(vid_idxs, vid_orig_files, vid_fps, vid_lens, vid_ft_stride, vid_ft_nframes)
        ):
            cls_logits_per_vid = [x[idx] for x in out_cls_logits]  # F [T,C]
            cls_feats_per_vid = [x[idx] for x in out_cls_feats]    # F [T,D]
            fpn_masks_per_vid = [x[idx] for x in fpn_masks]        # F [T]

            results_per_vid = self.inference_single_video(points, fpn_masks_per_vid, cls_logits_per_vid, cls_feats_per_vid)

            results_per_vid["video_id"] = vidx
            results_per_vid["orig_video_file"] = vid_orig_file
            results_per_vid["fps"] = fps
            results_per_vid["duration"] = vlen
            results_per_vid["feat_stride"] = stride
            results_per_vid["feat_num_frames"] = nframes

            # optional scenario id pass-through
            if "scenario_id" in video_list[idx]:
                results_per_vid["scenario_id"] = int(video_list[idx]["scenario_id"])

            results.append(results_per_vid)

        results = self.postprocessing(results)
        return results

    @torch.no_grad()
    def inference_single_video(self, points, fpn_masks, out_cls_logits, out_cls_feats):
        times_all = []
        scores_all = []
        cls_idxs_all = []
        feats_all = []

        for cls_i, feat_i, pts_i, mask_i in zip(out_cls_logits, out_cls_feats, points, fpn_masks):
            # cls_i: (T,C), feat_i: (T,D), mask_i: (T,)
            prob = cls_i.sigmoid() * mask_i.unsqueeze(-1)  # (T,C)

            flat = prob.flatten()  # (T*C,)
            keep = flat > self.test_pre_nms_thresh
            flat = flat[keep]
            topk_idxs = keep.nonzero(as_tuple=True)[0]

            num_topk = min(self.test_pre_nms_topk, topk_idxs.size(0))
            flat, order = flat.sort(descending=True)
            flat = flat[:num_topk].clone()
            topk_idxs = topk_idxs[order[:num_topk]].clone()

            pt_idxs = torch.div(topk_idxs, self.num_classes, rounding_mode="floor")
            cls_idxs = torch.fmod(topk_idxs, self.num_classes)

            pts = pts_i[pt_idxs]
            t_grid = pts[:, 0]  # (N,)

            # gather feature embeddings per candidate time (T,D) -> (N,D)
            f_sel = feat_i[pt_idxs]

            times_all.append(t_grid)
            scores_all.append(flat)
            cls_idxs_all.append(cls_idxs)
            feats_all.append(f_sel)

        times_all = torch.cat(times_all) if len(times_all) else torch.empty(0, device=self.device)
        scores_all = torch.cat(scores_all) if len(scores_all) else torch.empty(0, device=self.device)
        cls_idxs_all = torch.cat(cls_idxs_all) if len(cls_idxs_all) else torch.empty(0, device=self.device, dtype=torch.long)
        feats_all = torch.cat(feats_all) if len(feats_all) else torch.empty(0, device=self.device)

        return {"timestamps": times_all, "scores": scores_all, "labels": cls_idxs_all, "features": feats_all}

    @torch.no_grad()
    def postprocessing(self, results):
        """
        Postprocess per-video raw model outputs.

        Expected keys in each results_per_vid:
          - video_id, orig_video_file, fps, duration, feat_stride, feat_num_frames
          - timestamps: (N,) on feature grid (float, grid index)
          - scores:     (N,) confidence
          - labels:     (N,) class ids
          - features:   (N, D) per-candidate feature vectors (OPTIONAL, required for learnable suppression)
          - scenario_id: int (OPTIONAL, used if suppressor uses scenario embedding)

        Learnable suppression is applied BEFORE NMS on a TOP-M subset (O(M^2)).
        Suppression runs in GRID UNITS (same as training) to avoid scale mismatch.
        Everything is run on the suppressor's device (GPU) to avoid CPU/CUDA mismatch.
        Final outputs are moved to CPU for the evaluator.
        """
        processed_results = []

        # pick a single device for suppression (if enabled)
        sup_dev = None
        if getattr(self, "use_learnable_suppression", False) and (getattr(self, "suppressor", None) is not None):
            sup_dev = next(self.suppressor.parameters()).device

        for results_per_vid in results:
            vidx = results_per_vid["video_id"]
            orig_video_file = results_per_vid["orig_video_file"]
            fps = results_per_vid["fps"]
            vlen = results_per_vid["duration"]
            stride = results_per_vid["feat_stride"]
            nframes = results_per_vid["feat_num_frames"]

            # Keep tensors on their current device; do NOT .cpu() yet.
            # NOTE: "times" are in GRID UNITS here (feature indices).
            times = results_per_vid["timestamps"].detach()
            scores = results_per_vid["scores"].detach()
            labels = results_per_vid["labels"].detach()

            feats = results_per_vid.get("features", None)
            if feats is not None:
                feats = feats.detach()

            score_before = scores.detach().clone()
            score_after = scores.detach().clone()
            selected_topm_mask = torch.zeros_like(scores, dtype=torch.bool)
            pred_radii = torch.full_like(scores, float("nan"), dtype=torch.float32)

            # Learnable suppression BEFORE NMS (optional) in GRID UNITS
            if (
                    sup_dev is not None
                    and times.numel() > 0
                    and feats is not None
                    and scores.numel() == times.numel()
                    and feats.shape[0] == times.numel()
            ):
                # Move candidate tensors to suppressor device
                t_grid_d = times.to(sup_dev)
                scores_d = scores.to(sup_dev)
                labels_d = labels.to(sup_dev)
                feats_d = feats.to(sup_dev)

                # Reduce candidates to top-M (for O(M^2) suppression)
                M = int(getattr(self, "suppress_topk_test", 512))
                M = min(M, int(scores_d.numel()))
                if M < int(scores_d.numel()):
                    svals, sidx = torch.topk(scores_d, k=M, largest=True, sorted=False)
                    t_sel = t_grid_d[sidx]
                    l_sel = labels_d[sidx]
                    f_sel = feats_d[sidx]
                else:
                    sidx = None
                    t_sel, svals, l_sel, f_sel = t_grid_d, scores_d, labels_d, feats_d

                scen = int(results_per_vid.get("scenario_id", 0))
                scen = torch.tensor(scen, dtype=torch.long, device=sup_dev)

                score_before = scores_d.clone()
                selected_topm_mask = torch.zeros_like(scores_d, dtype=torch.bool)
                pred_radii = torch.full_like(scores_d, float("nan"), dtype=torch.float32)

                # Predict radii for selected candidates (for mechanism analysis)
                r_sel = self.suppressor.predict_radii(
                    features=f_sel,
                    scenario_id=scen,
                )

                # suppressor returns adjusted scores for the selected candidates
                s_adj = self.suppressor(
                    timestamps=t_sel,
                    scores=svals,
                    features=f_sel,
                    scenario_id=scen,
                    class_ids=l_sel,
                    class_aware=self.suppress_class_aware_test,
                )

                # Write back adjusted scores into full score vector
                if sidx is not None:
                    selected_topm_mask[sidx] = True
                    pred_radii[sidx] = r_sel
                    scores_d = scores_d.clone()
                    scores_d[sidx] = s_adj
                else:
                    selected_topm_mask = torch.ones_like(scores_d, dtype=torch.bool)
                    pred_radii = r_sel
                    scores_d = s_adj

                score_after = scores_d.clone()

                # Continue postproc on suppressor device
                times = t_grid_d
                scores = scores_d
                labels = labels_d

            if score_before.device != times.device:
                score_before = score_before.to(times.device)
            if score_after.device != times.device:
                score_after = score_after.to(times.device)
            if selected_topm_mask.device != times.device:
                selected_topm_mask = selected_topm_mask.to(times.device)
            if pred_radii.device != times.device:
                pred_radii = pred_radii.to(times.device)

            # Convert grid -> seconds on the same device as times
            if times.numel() > 0:
                t_secs = (times * stride + 0.5 * nframes) / fps
                t_secs = torch.clamp(t_secs, min=0.0, max=vlen)
            else:
                t_secs = times  # empty

            if self.debug_suppress_dump and times.numel() > 0:
                k_dump = min(self.debug_suppress_dump_topk, int(times.numel()))
                if k_dump > 0:
                    # Select top-k by score_before for stable comparisons
                    _, dump_idx = torch.topk(score_before, k=k_dump, largest=True, sorted=False)

                    t_secs_pre_nms = (times * stride + 0.5 * nframes) / fps
                    t_secs_pre_nms = torch.clamp(t_secs_pre_nms, min=0.0, max=vlen)

                    scen_id = int(results_per_vid.get("scenario_id", -1))
                    rows = []
                    for j in dump_idx.tolist():
                        pr = pred_radii[j]
                        if torch.isfinite(pr):
                            pr_val = float(pr.item())
                        else:
                            pr_val = None

                        rows.append({
                            "video_id": vidx,
                            "scenario_id": scen_id,
                            "label": int(labels[j].item()),
                            "time_grid": float(times[j].item()),
                            "time_sec": float(t_secs_pre_nms[j].item()),
                            "score_before": float(score_before[j].item()),
                            "score_after": float(score_after[j].item()),
                            "selected_topM": bool(selected_topm_mask[j].item()),
                            "pred_radius_grid": pr_val,
                            "suppression_applied": bool(self.use_learnable_suppression and (sup_dev is not None)),
                        })

                    self._append_suppress_dump_rows(rows)

            # NMS (runs on whatever device t_secs/scores/labels are on)
            if self.test_nms_method != "none" and t_secs.numel() > 0:
                t_secs, scores, labels = batched_nms_timestamps(
                    t_secs,
                    scores,
                    labels,
                    min_score=self.test_min_score,
                    l1_radius=self.test_l1_radius,
                    max_t_num=self.test_max_t_num,
                    multiclass=self.test_multiclass_nms,
                    nms_method=self.test_nms_method,
                    nms_sigma=self.test_nms_sigma,
                )

            # Final outputs to CPU for evaluator/serialization
            processed_results.append(
                {
                    "video_id": vidx,
                    "orig_video_file": orig_video_file,
                    "timestamps": t_secs.detach().cpu(),
                    "scores": scores.detach().cpu(),
                    "labels": labels.detach().cpu(),
                }
            )

        return processed_results

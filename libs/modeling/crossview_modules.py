"""
Shared cross-view attention and fusion modules.

Used by:
- actionformer_crossview_timestamps.py           (Standard + CrossView)
- actionformer_adaptive_crossview_timestamps.py   (Adaptive + CrossView)
- actionformer_adaptive_pose_crossview_timestamps.py (Adaptive + CrossView + Pose)
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

from .blocks import MaskedConv1D
from .models import make_backbone, make_neck


class CrossViewAttention(nn.Module):
    """
    Bidirectional cross-attention between ego and exo feature streams.

    ego_enhanced = ego + gate_ego * CrossAttn(Q=ego, KV=exo)
    exo_enhanced = exo + gate_exo * CrossAttn(Q=exo, KV=ego)

    Both streams must have the same channel dimension (fpn_dim).
    Operates on (B, C, T) tensors with (B, 1, T) boolean masks.
    """

    def __init__(self, n_embd, n_head=4, attn_pdrop=0.0, proj_pdrop=0.0, gate_init=-2.0):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_channels = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.n_channels)

        # Ego -> Exo cross-attention (Q from ego, KV from exo)
        self.q_ego = nn.Conv1d(n_embd, n_embd, 1)
        self.k_exo = nn.Conv1d(n_embd, n_embd, 1)
        self.v_exo = nn.Conv1d(n_embd, n_embd, 1)
        self.proj_ego = nn.Conv1d(n_embd, n_embd, 1)

        # Exo -> Ego cross-attention (Q from exo, KV from ego)
        self.q_exo = nn.Conv1d(n_embd, n_embd, 1)
        self.k_ego = nn.Conv1d(n_embd, n_embd, 1)
        self.v_ego = nn.Conv1d(n_embd, n_embd, 1)
        self.proj_exo = nn.Conv1d(n_embd, n_embd, 1)

        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)

        # Learnable gates (initialized near zero -> starts as identity)
        self.gate_ego = nn.Parameter(torch.tensor(float(gate_init)))
        self.gate_exo = nn.Parameter(torch.tensor(float(gate_init)))

    def _cross_attn(self, q_proj, k_proj, v_proj, out_proj, x_q, x_kv, mask_q, mask_kv):
        """
        Cross-attention: Q from x_q, K/V from x_kv.
        x_q: (B, C, T_q), x_kv: (B, C, T_kv)
        mask_q: (B, 1, T_q), mask_kv: (B, 1, T_kv)
        Returns: (B, C, T_q)
        """
        B, C, T_q = x_q.size()
        T_kv = x_kv.size(2)

        q = q_proj(x_q)   # (B, C, T_q)
        k = k_proj(x_kv)  # (B, C, T_kv)
        v = v_proj(x_kv)  # (B, C, T_kv)

        # Reshape for multi-head: (B, nh, T, hs)
        q = q.view(B, self.n_head, self.n_channels, T_q).transpose(2, 3)
        k = k.view(B, self.n_head, self.n_channels, T_kv).transpose(2, 3)
        v = v.view(B, self.n_head, self.n_channels, T_kv).transpose(2, 3)

        # Attention: (B, nh, T_q, T_kv)
        att = (q * self.scale) @ k.transpose(-2, -1)

        # Mask out invalid KV positions
        att = att.masked_fill(
            torch.logical_not(mask_kv[:, :, None, :]), float('-inf')
        )
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        # Weighted sum: (B, nh, T_q, hs)
        out = att @ (v * mask_kv[:, :, :, None].to(v.dtype))
        out = out.transpose(2, 3).contiguous().view(B, C, T_q)

        out = self.proj_drop(out_proj(out)) * mask_q.to(out.dtype)
        return out

    def forward(self, ego_feat, exo_feat, ego_mask, exo_mask, ablate=False):
        """
        ego_feat, exo_feat: (B, C, T)
        ego_mask, exo_mask: (B, 1, T) boolean
        ablate: if True, force gates to 0 (skip cross-view attention contribution)
        Returns: ego_enhanced, exo_enhanced (same shapes)
        """
        if ablate:
            return ego_feat, exo_feat

        # Ego attends to exo
        ego_cross = self._cross_attn(
            self.q_ego, self.k_exo, self.v_exo, self.proj_ego,
            ego_feat, exo_feat, ego_mask, exo_mask,
        )
        ego_enhanced = ego_feat + torch.sigmoid(self.gate_ego) * ego_cross

        # Exo attends to ego
        exo_cross = self._cross_attn(
            self.q_exo, self.k_ego, self.v_ego, self.proj_exo,
            exo_feat, ego_feat, exo_mask, ego_mask,
        )
        exo_enhanced = exo_feat + torch.sigmoid(self.gate_exo) * exo_cross

        return ego_enhanced, exo_enhanced


class CrossViewFusion(nn.Module):
    """
    Fuses ego_enhanced and exo_enhanced into a single stream.

    fused = ego + gate * proj(cat(ego, exo))

    Same pattern as the pose gated late fusion, but for ego/exo streams.
    """

    def __init__(self, fpn_dim, dropout=0.1, gate_init=-1.0):
        super().__init__()
        self.proj = nn.Sequential(
            MaskedConv1D(fpn_dim * 2, fpn_dim, 1, stride=1, padding=0, bias=False),
            nn.GroupNorm(1, fpn_dim),
            nn.ReLU(inplace=True),
        )
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        self.drop = nn.Dropout(dropout)

    def forward(self, ego_feat, exo_feat, mask, ablate=False):
        """
        ego_feat, exo_feat: (B, C, T)
        mask: (B, 1, T)
        ablate: if True, force gate to 0 (skip fusion, pass ego through)
        Returns: fused (B, C, T)
        """
        if ablate:
            return ego_feat * mask.to(ego_feat.dtype)

        cat_feat = torch.cat([ego_feat, exo_feat], dim=1)  # (B, 2C, T)

        proj_out = cat_feat
        for module in self.proj:
            if isinstance(module, MaskedConv1D):
                proj_out, _ = module(proj_out, mask)
            else:
                proj_out = module(proj_out)

        fused = ego_feat + torch.sigmoid(self.gate) * self.drop(proj_out)
        fused = fused * mask.to(fused.dtype)
        return fused


def build_crossview_modules(
    num_fpn_levels,
    fpn_dim,
    cross_view_n_head=4,
    cross_view_attn_pdrop=0.0,
    cross_view_proj_pdrop=0.1,
    cross_view_gate_init=-2.0,
    cross_view_fusion_dropout=0.1,
    cross_view_fusion_gate_init=-1.0,
):
    """Factory function to create cross-view attention + fusion module lists."""
    attn_modules = nn.ModuleList([
        CrossViewAttention(
            n_embd=fpn_dim,
            n_head=cross_view_n_head,
            attn_pdrop=cross_view_attn_pdrop,
            proj_pdrop=cross_view_proj_pdrop,
            gate_init=cross_view_gate_init,
        )
        for _ in range(num_fpn_levels)
    ])

    fusion_modules = nn.ModuleList([
        CrossViewFusion(
            fpn_dim=fpn_dim,
            dropout=cross_view_fusion_dropout,
            gate_init=cross_view_fusion_gate_init,
        )
        for _ in range(num_fpn_levels)
    ])

    return attn_modules, fusion_modules


def build_exo_backbone_neck(
    backbone_type, input_dim, embd_dim, n_head, embd_kernel_size,
    max_seq_len, backbone_arch, n_mha_win_size, scale_factor,
    embd_with_ln, train_dropout, train_droppath, use_abs_pe, use_rel_pe,
    fpn_type, fpn_dim, fpn_start_level, fpn_with_ln,
):
    """Factory function to create a separate exo backbone + neck."""
    if backbone_type == "convTransformer":
        if isinstance(n_mha_win_size, int):
            mha_win = [n_mha_win_size] * (1 + backbone_arch[-1])
        else:
            mha_win = n_mha_win_size
        exo_backbone = make_backbone(
            "convTransformer",
            **{
                "n_in": input_dim,
                "n_embd": embd_dim,
                "n_head": n_head,
                "n_embd_ks": embd_kernel_size,
                "max_len": max_seq_len,
                "arch": backbone_arch,
                "mha_win_size": mha_win,
                "scale_factor": scale_factor,
                "with_ln": embd_with_ln,
                "attn_pdrop": 0.0,
                "proj_pdrop": train_dropout,
                "path_pdrop": train_droppath,
                "use_abs_pe": use_abs_pe,
                "use_rel_pe": use_rel_pe,
            },
        )
    else:
        exo_backbone = make_backbone(
            "conv",
            **{
                "n_in": input_dim,
                "n_embd": embd_dim,
                "n_embd_ks": embd_kernel_size,
                "arch": backbone_arch,
                "scale_factor": scale_factor,
                "with_ln": embd_with_ln,
            },
        )

    if isinstance(embd_dim, (list, tuple)):
        _embd_dim = sum(embd_dim)
    else:
        _embd_dim = embd_dim

    exo_neck = make_neck(
        fpn_type,
        **{
            "in_channels": [_embd_dim] * (backbone_arch[-1] + 1),
            "out_channel": fpn_dim,
            "scale_factor": scale_factor,
            "start_level": fpn_start_level,
            "with_ln": fpn_with_ln,
        },
    )

    return exo_backbone, exo_neck


def build_pose_stream(
    pose_input_dim, pose_backbone_type, pose_embd_dim, pose_embd_with_ln,
    pose_neck_type, n_head, embd_kernel_size, max_seq_len, backbone_arch,
    n_mha_win_size, scale_factor, train_dropout, train_droppath,
    use_abs_pe, use_rel_pe, fpn_dim, fpn_start_level, fpn_with_ln,
    pose_gate_init, pose_fusion_dropout, num_fpn_levels,
):
    """Factory function to create pose backbone + neck + fusion modules."""
    assert pose_backbone_type in ("conv", "convTransformer")
    if pose_backbone_type == "convTransformer":
        if isinstance(n_mha_win_size, int):
            pose_mha_win_size = [n_mha_win_size] * (1 + backbone_arch[-1])
        else:
            pose_mha_win_size = n_mha_win_size
        pose_backbone = make_backbone(
            "convTransformer",
            **{
                "n_in": pose_input_dim,
                "n_embd": pose_embd_dim,
                "n_head": n_head,
                "n_embd_ks": embd_kernel_size,
                "max_len": max_seq_len,
                "arch": backbone_arch,
                "mha_win_size": pose_mha_win_size,
                "scale_factor": scale_factor,
                "with_ln": pose_embd_with_ln,
                "attn_pdrop": 0.0,
                "proj_pdrop": train_dropout,
                "path_pdrop": train_droppath,
                "use_abs_pe": use_abs_pe,
                "use_rel_pe": use_rel_pe,
            },
        )
    else:
        pose_backbone = make_backbone(
            "conv",
            **{
                "n_in": pose_input_dim,
                "n_embd": pose_embd_dim,
                "n_embd_ks": embd_kernel_size,
                "arch": backbone_arch,
                "scale_factor": scale_factor,
                "with_ln": pose_embd_with_ln,
            },
        )

    if isinstance(pose_embd_dim, (list, tuple)):
        pose_embd_dim = sum(pose_embd_dim)

    pose_neck = make_neck(
        pose_neck_type,
        **{
            "in_channels": [pose_embd_dim] * (backbone_arch[-1] + 1),
            "out_channel": fpn_dim,
            "scale_factor": scale_factor,
            "start_level": fpn_start_level,
            "with_ln": fpn_with_ln,
        },
    )

    pose_fusion_proj = nn.ModuleList([
        nn.Sequential(
            MaskedConv1D(fpn_dim * 2, fpn_dim, 1, stride=1, padding=0, bias=False),
            nn.GroupNorm(1, fpn_dim),
            nn.ReLU(inplace=True),
        )
        for _ in range(num_fpn_levels)
    ])

    pose_gates = nn.Parameter(
        torch.full((num_fpn_levels,), float(pose_gate_init))
    )

    pose_fusion_drop = nn.Dropout(pose_fusion_dropout)

    return pose_backbone, pose_neck, pose_fusion_proj, pose_gates, pose_fusion_drop


def fuse_pose_gated(video_fpn_feats, video_fpn_masks, pose_fpn_feats, pose_fpn_masks,
                    pose_fusion_proj, pose_gates, pose_fusion_drop, ablate=False):
    """Gated late fusion for pose (shared implementation).

    Args:
        ablate: if True, skip pose fusion (gates forced to 0)
    """
    fused_feats = []
    fused_masks = []

    for lvl, (v_feat, v_mask, p_feat, p_mask) in enumerate(
        zip(video_fpn_feats, video_fpn_masks, pose_fpn_feats, pose_fpn_masks)
    ):
        if ablate:
            fused_feats.append(v_feat)
            fused_masks.append(v_mask)
            continue

        if p_feat.shape[-1] != v_feat.shape[-1]:
            p_feat = F.interpolate(p_feat, size=v_feat.shape[-1], mode="nearest")
            p_mask = F.interpolate(p_mask.float(), size=v_feat.shape[-1], mode="nearest").bool()

        p_feat = p_feat * p_mask.to(p_feat.dtype)
        cat_feat = torch.cat([v_feat, p_feat], dim=1)

        proj_out = cat_feat
        for module in pose_fusion_proj[lvl]:
            if isinstance(module, MaskedConv1D):
                proj_out, _ = module(proj_out, v_mask)
            else:
                proj_out = module(proj_out)

        gate = torch.sigmoid(pose_gates[lvl])
        fused = v_feat + gate * pose_fusion_drop(proj_out)
        fused = fused * v_mask.to(fused.dtype)

        fused_feats.append(fused)
        fused_masks.append(v_mask)

    return tuple(fused_feats), tuple(fused_masks)


def run_crossview_fpn(ego_fpn_feats, ego_fpn_masks, exo_fpn_feats, exo_fpn_masks,
                      cross_view_attn, cross_view_fusion, num_fpn_levels,
                      ablate_attn=False, ablate_fusion=False):
    """Run cross-view attention + fusion at each FPN level.

    Args:
        ablate_attn: if True, skip cross-view attention (gates forced to 0)
        ablate_fusion: if True, skip cross-view fusion (gate forced to 0)
    """
    fused_fpn_feats = []
    fused_fpn_masks = []
    for lvl in range(num_fpn_levels):
        ego_enh, exo_enh = cross_view_attn[lvl](
            ego_fpn_feats[lvl], exo_fpn_feats[lvl],
            ego_fpn_masks[lvl], exo_fpn_masks[lvl],
            ablate=ablate_attn,
        )
        fused = cross_view_fusion[lvl](
            ego_enh, exo_enh, ego_fpn_masks[lvl],
            ablate=ablate_fusion,
        )
        fused_fpn_feats.append(fused)
        fused_fpn_masks.append(ego_fpn_masks[lvl])

    return tuple(fused_fpn_feats), tuple(fused_fpn_masks)

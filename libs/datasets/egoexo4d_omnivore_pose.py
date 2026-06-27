import os
import json
import copy
import random
import logging
import numpy as np
from collections import defaultdict

import torch
from torch.utils.data import Dataset

from .datasets import register_dataset

logger = logging.getLogger(__name__)

BODY_JOINTS = [
    "nose", "left-eye", "right-eye", "left-ear", "right-ear",
    "left-shoulder", "right-shoulder", "left-elbow", "right-elbow",
    "left-wrist", "right-wrist", "left-hip", "right-hip",
    "left-knee", "right-knee", "left-ankle", "right-ankle"
]

HAND_JOINTS_RIGHT = [
    "right_wrist",
    "right_thumb_1", "right_thumb_2", "right_thumb_3", "right_thumb_4",
    "right_index_1", "right_index_2", "right_index_3", "right_index_4",
    "right_middle_1", "right_middle_2", "right_middle_3", "right_middle_4",
    "right_ring_1", "right_ring_2", "right_ring_3", "right_ring_4",
    "right_pinky_1", "right_pinky_2", "right_pinky_3", "right_pinky_4"
]
HAND_JOINTS_LEFT = [j.replace("right_", "left_") for j in HAND_JOINTS_RIGHT]

# ── Body joint index definitions for kinematic features ──
# COCO 17-joint ordering (same as BODY_JOINTS above)
_J = {name: i for i, name in enumerate(BODY_JOINTS)}

# Angle triplets: (parent, joint, child) — angle at the middle joint
BODY_ANGLE_TRIPLETS = [
    (_J["left-shoulder"],  _J["left-elbow"],   _J["left-wrist"]),     # left elbow
    (_J["right-shoulder"], _J["right-elbow"],  _J["right-wrist"]),    # right elbow
    (_J["left-hip"],       _J["left-knee"],    _J["left-ankle"]),     # left knee
    (_J["right-hip"],      _J["right-knee"],   _J["right-ankle"]),    # right knee
    (_J["left-elbow"],     _J["left-shoulder"], _J["left-hip"]),      # left shoulder angle
    (_J["right-elbow"],    _J["right-shoulder"], _J["right-hip"]),    # right shoulder angle
    (_J["left-shoulder"],  _J["left-hip"],      _J["left-knee"]),     # left hip angle
    (_J["right-shoulder"], _J["right-hip"],     _J["right-knee"]),    # right hip angle
]

# Pairwise distance pairs — skill-relevant distances
BODY_DISTANCE_PAIRS = [
    (_J["left-wrist"],  _J["right-wrist"]),    # hand spread
    (_J["left-ankle"],  _J["right-ankle"]),     # stance width
    (_J["left-wrist"],  _J["left-hip"]),        # left reach
    (_J["right-wrist"], _J["right-hip"]),       # right reach
    (_J["nose"],        _J["left-hip"]),         # head-to-hip (posture)
    (_J["nose"],        _J["right-hip"]),        # head-to-hip (posture)
    (_J["left-wrist"],  _J["left-ankle"]),       # left hand-foot distance
    (_J["right-wrist"], _J["right-ankle"]),      # right hand-foot distance
]


def _compute_angle(a, b, c):
    """Angle at joint b (in radians) given 3D positions a, b, c. Shape: (T, 3) each."""
    ba = a - b
    bc = c - b
    cos_angle = np.sum(ba * bc, axis=-1) / (
        np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1) + 1e-8
    )
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.arccos(cos_angle)  # (T,)


def compute_body_kinematics(raw_seq, normalize=True):
    """
    Compute kinematic features from raw body pose sequence.

    Args:
        raw_seq: (T, 51) raw 3D body keypoints (17 joints × 3 coords), already normalized.
        normalize: whether to normalize angles to [0, 1] and clip distances.

    Returns:
        kin_feats: (T, D_kin) kinematic features.
            - 8 joint angles (normalized to [0,1])
            - 8 pairwise distances
            - 51 velocities (first differences of raw coords)
        Total D_kin = 8 + 8 + 51 = 67
    """
    T = raw_seq.shape[0]
    joints_3d = raw_seq.reshape(T, 17, 3)  # (T, 17, 3)

    # ── Joint angles ──
    angles = np.zeros((T, len(BODY_ANGLE_TRIPLETS)), dtype=np.float32)
    for k, (i_a, i_b, i_c) in enumerate(BODY_ANGLE_TRIPLETS):
        angles[:, k] = _compute_angle(joints_3d[:, i_a], joints_3d[:, i_b], joints_3d[:, i_c])
    if normalize:
        angles = angles / np.pi  # [0, 1]

    # ── Pairwise distances ──
    distances = np.zeros((T, len(BODY_DISTANCE_PAIRS)), dtype=np.float32)
    for k, (i_a, i_b) in enumerate(BODY_DISTANCE_PAIRS):
        distances[:, k] = np.linalg.norm(joints_3d[:, i_a] - joints_3d[:, i_b], axis=-1)

    # ── Velocity (first differences) ──
    velocity = np.zeros_like(raw_seq)  # (T, 51)
    if T > 1:
        velocity[1:] = raw_seq[1:] - raw_seq[:-1]

    return np.concatenate([angles, distances, velocity], axis=-1)  # (T, 67)


# ── Hand joint indices (within a single 21-joint hand, MediaPipe / POTTER ordering) ──
HAND_WRIST = 0
HAND_FINGER_TRIPLETS = [  # (proximal, middle, distal) — angle at the middle joint = curl
    (1, 2, 3),     # thumb
    (5, 6, 7),     # index
    (9, 10, 11),   # middle
    (13, 14, 15),  # ring
    (17, 18, 19),  # pinky
]
HAND_FINGERTIPS = [4, 8, 12, 16, 20]


def compute_hand_kinematics(raw_seq, normalize=True):
    """
    Compute kinematic features from raw hand pose sequence.

    Args:
        raw_seq: (T, 126) raw 3D hand keypoints in the layout:
                 right hand (21 joints × 3) | left hand (21 joints × 3).
                 Assumed wrist-relative (output of _normalize_pose_sequence(kind='hand')).
        normalize: if True, scale angles to [0, 1] (divide by π).

    Returns:
        kin_feats: (T, 30) features per timestep:
            - 5 finger-curl angles per hand × 2 hands = 10
            - 5 fingertip-to-wrist distances per hand × 2 hands = 10
            - 5 fingertip-velocity magnitudes per hand × 2 hands = 10
    """
    T = raw_seq.shape[0]
    h_right = raw_seq[:, :63].reshape(T, 21, 3)
    h_left = raw_seq[:, 63:].reshape(T, 21, 3)

    parts = []
    for h in (h_right, h_left):
        curls = np.zeros((T, len(HAND_FINGER_TRIPLETS)), dtype=np.float32)
        for k, (i_a, i_b, i_c) in enumerate(HAND_FINGER_TRIPLETS):
            curls[:, k] = _compute_angle(h[:, i_a], h[:, i_b], h[:, i_c])
        if normalize:
            curls = curls / np.pi  # [0, 1]

        wrist = h[:, HAND_WRIST]                      # (T, 3)
        tips = h[:, HAND_FINGERTIPS]                  # (T, 5, 3)
        tip_dist = np.linalg.norm(tips - wrist[:, None, :], axis=2)  # (T, 5)

        tip_vel = np.zeros((T, 5), dtype=np.float32)
        if T > 1:
            tip_vel[1:] = np.linalg.norm(tips[1:] - tips[:-1], axis=2)

        parts.append(np.concatenate([curls, tip_dist, tip_vel], axis=1))

    return np.concatenate(parts, axis=1).astype(np.float32)  # (T, 30)


@register_dataset("egoexo4d_omnivore_pose")
class EgoExo4DOmnivorePoseDataset(Dataset):
    def __init__(
        self,
        is_training,
        is_validation,
        split,
        feat_folder,
        json_file,
        feat_stride,
        num_frames,
        default_fps,
        downsample_rate,
        max_seq_len,
        trunc_thresh,
        crop_ratio,
        input_dim,
        num_classes,
        file_prefix,
        file_ext,
        force_upsampling,
        egoexo_type,
        pose_root,
        pose_type="body",              # body | hand | body_hand
        pose_max_frame_dist=8,         # nearest frame tolerance
        pose_require_file=False,       # if True: drop videos without pose file
        pose_normalize=True,
        pose_use_kinematics=True,      # if True: append angles/distances/velocity
        log_pose_coverage=False,       # if True: scan & log per-scenario coverage at init
        concat_views=False,            # if True: concatenate all view features per take
        concat_norm="l2",              # per-view normalization before concat: "l2", "zscore", "none"
        cross_view=False,              # if True: return ego + exo_avg as separate tensors (both only)
        load_hand_pose=False,          # if True: ALSO load hand pose into separate fields
                                       # (hand_feats, hand_valid). The body stream is unchanged.
                                       # Used by ActionFormerSkillHandTimestamps.
        hand_use_kinematics=True,      # if load_hand_pose: append the 30-dim hand kinematics
        hand_pose_root=None,           # if set: load hand pose from a DIFFERENT root than
                                       # body pose. Useful e.g. for training with GT body
                                       # pose ('OriginalData/.../ego_pose') but predicted
                                       # hand pose ('PreprocessedData/.../predicted_ego_pose')
                                       # since GT hand annotations are sparse (~5–10% of
                                       # stride-16 tokens) while predicted hand is dense.
                                       # If None, falls back to `pose_root`.
    ):
        if not isinstance(feat_folder, (list, tuple)):
            feat_folder = (feat_folder,)
        assert all([os.path.exists(folder) for folder in feat_folder])
        assert os.path.exists(json_file)
        assert isinstance(split, (tuple, list))
        assert crop_ratio is None or len(crop_ratio) == 2
        assert pose_type in ("body", "hand", "body_hand")

        self.feat_folder = feat_folder
        self.file_prefix = file_prefix if file_prefix is not None else ""
        self.file_ext = file_ext
        self.json_file = json_file

        self.split = split
        self.is_training = is_training

        self.feat_stride = feat_stride
        self.num_frames = num_frames
        self.input_dim = input_dim
        self.default_fps = default_fps
        self.downsample_rate = downsample_rate
        self.max_seq_len = max_seq_len
        self.trunc_thresh = trunc_thresh
        self.crop_ratio = crop_ratio
        self.num_classes = num_classes
        self.egoexo_type = egoexo_type
        self.concat_views = bool(concat_views)
        self.cross_view = bool(cross_view)
        assert not (self.concat_views and self.cross_view), (
            "concat_views and cross_view are mutually exclusive"
        )
        assert concat_norm in ("l2", "zscore", "none"), (
            f"concat_norm must be 'l2', 'zscore', or 'none', got '{concat_norm}'"
        )
        self.concat_norm = concat_norm

        self.pose_root = pose_root
        self.pose_type = pose_type
        self.pose_max_frame_dist = int(pose_max_frame_dist)
        self.pose_require_file = bool(pose_require_file)
        self.pose_normalize = bool(pose_normalize)
        self.pose_use_kinematics = bool(pose_use_kinematics)

        # Raw pose dims
        if pose_type == "body":
            self._raw_pose_dim = 51
        elif pose_type == "hand":
            self._raw_pose_dim = 126
        else:
            self._raw_pose_dim = 177

        # Kinematics only supported for body (and body part of body_hand)
        if self.pose_use_kinematics and pose_type in ("body", "body_hand"):
            # body kinematics: 8 angles + 8 distances + 51 velocity = 67
            self._kin_dim = 67
        else:
            self._kin_dim = 0

        self.pose_dim = self._raw_pose_dim + self._kin_dim

        # ── Hand-pose side stream (separate from the body pose loaded above) ──
        self.load_hand_pose = bool(load_hand_pose)
        self.hand_use_kinematics = bool(hand_use_kinematics)
        # raw hand: 21 joints × 3 coords × 2 hands = 126
        # kinematics: 5 curls + 5 tip-dist + 5 tip-vel per hand × 2 = 30
        self.hand_raw_dim = 126
        self.hand_kin_dim = 30 if self.hand_use_kinematics else 0
        self.hand_dim = self.hand_raw_dim + self.hand_kin_dim
        # Hand pose root — fall back to body pose_root if not specified.
        self.hand_pose_root = hand_pose_root if hand_pose_root is not None else pose_root

        # Hook for subclasses to override (e.g. HoloAssist with 20 tasks).
        # See `_build_scenario_name_dict` below for the EgoExo4D default.
        self.scenario_name_dict = self._build_scenario_name_dict()
        self.scenario_id_to_name = {v: k for k, v in self.scenario_name_dict.items()}

        self.label_dict = None
        self._pose_cache = {}  # (subset, take_uid, kind) -> entry | None
        self._feat_cache = {}  # filepath -> np.ndarray (float32)

        dict_db, label_dict = self._load_json_db(self.json_file)
        # Inference-only fallback for manifests without timestamp annotations
        if len(label_dict) == 0:
            if is_training:
                raise AssertionError(
                    f"No labels found in {self.json_file}. "
                    "Training requires timestamp annotations with label_id."
                )
            label_dict = {
                "good_executions": 0,
                "tips_for_improvement": 1,
            }

        assert len(label_dict) == num_classes, (
            f"Label dict size ({len(label_dict)}) != num_classes ({num_classes}) "
            f"for {self.json_file}"
        )
        self.data_list = dict_db
        self.label_dict = label_dict

        if self.concat_views:
            assert self.egoexo_type in ('exo', 'both'), (
                f"concat_views requires egoexo_type 'exo' or 'both', got '{self.egoexo_type}'"
            )
            logger.info(
                "concat_views enabled (egoexo_type=%s): %d grouped entries",
                self.egoexo_type, len(self.data_list),
            )

        if self.cross_view:
            assert self.egoexo_type == 'both', (
                f"cross_view requires egoexo_type 'both', got '{self.egoexo_type}'"
            )
            logger.info(
                "cross_view enabled: %d grouped entries (ego + exo_avg)",
                len(self.data_list),
            )

        self.db_attributes = {
            "dataset_name": "egoexo4d_pose",
            "empty_label_ids": [],
            "pose_dim": self.pose_dim,
        }

        # ── Pre-load all feature files into memory ──
        self._preload_features()

        # ── Approach A: Pose coverage logging (gated — heavy at init) ──
        if log_pose_coverage:
            self._log_pose_coverage()

    def get_attributes(self):
        return self.db_attributes

    def __len__(self):
        return len(self.data_list)

    def _log_pose_coverage(self):
        """Scan all videos and log per-scenario pose file coverage."""
        scenario_total = defaultdict(int)
        scenario_has_pose = defaultdict(int)
        scenario_total_tokens = defaultdict(int)
        scenario_valid_tokens = defaultdict(int)

        # Sample up to 50 videos per scenario for token-level coverage
        scenario_sampled = defaultdict(int)
        MAX_SAMPLE = 50

        for item in self.data_list:
            scen_id = item["scenario_id"]
            scen_name = self.scenario_id_to_name[scen_id]
            scenario_total[scen_name] += 1

            has_file = self._has_pose_file(item["take_uid"], item["pose_subset"])
            if has_file:
                scenario_has_pose[scen_name] += 1

            # Token-level coverage (sampled)
            if scenario_sampled[scen_name] < MAX_SAMPLE and has_file:
                scenario_sampled[scen_name] += 1
                feat_stride = self.feat_stride * self.downsample_rate
                # estimate num_tokens from duration
                n_tok = max(1, int(item["duration"] * item["fps"] / feat_stride))
                n_tok = min(n_tok, self.max_seq_len)
                _, pv = self._load_pose_sequence(
                    item["take_uid"], item["pose_subset"], n_tok, feat_stride
                )
                scenario_total_tokens[scen_name] += n_tok
                scenario_valid_tokens[scen_name] += int(pv.sum())

        total_vids = len(self.data_list)
        total_with_pose = sum(scenario_has_pose.values())

        print(f"\n{'='*70}")
        print(f"  POSE COVERAGE REPORT  (split={self.split}, type={self.pose_type}, "
              f"kin={self.pose_use_kinematics}, dim={self.pose_dim})")
        print(f"{'='*70}")
        print(f"  {'Scenario':<20s} {'Videos':>8s} {'w/ Pose':>8s} {'File%':>7s} {'Token%':>8s}")
        print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*7} {'-'*8}")

        for scen_name in sorted(scenario_total.keys()):
            n_total = scenario_total[scen_name]
            n_pose = scenario_has_pose[scen_name]
            file_pct = 100.0 * n_pose / max(n_total, 1)

            t_total = scenario_total_tokens[scen_name]  # defaultdict → 0 if missing
            t_valid = scenario_valid_tokens[scen_name]
            tok_pct = 100.0 * t_valid / max(t_total, 1) if t_total > 0 else 0.0

            print(f"  {scen_name:<20s} {n_total:>8d} {n_pose:>8d} {file_pct:>6.1f}% {tok_pct:>7.1f}%")

        file_pct_total = 100.0 * total_with_pose / max(total_vids, 1)
        t_total_all = sum(scenario_total_tokens.values())
        t_valid_all = sum(scenario_valid_tokens.values())
        tok_pct_all = 100.0 * t_valid_all / max(t_total_all, 1) if t_total_all > 0 else 0.0

        print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*7} {'-'*8}")
        print(f"  {'TOTAL':<20s} {total_vids:>8d} {total_with_pose:>8d} {file_pct_total:>6.1f}% {tok_pct_all:>7.1f}%")
        print(f"{'='*70}\n")

    def _build_scenario_name_dict(self):
        """Return the {scenario_name -> int_id} mapping used by the
        adaptive-suppression head's per-scenario embedding.

        The default is the 8-class Ego-Exo4D taxonomy. Subclasses (e.g.
        HoloAssistOmnivorePoseDataset) override this to provide a different
        taxonomy — must be set BEFORE _load_json_db runs since it asserts
        scenario membership when ingesting each entry.
        """
        return {
            "Rock Climbing": 0,
            "Cooking":       1,
            "Music":         2,
            "Basketball":    3,
            "Health":        4,
            "Soccer":        5,
            "Dance":         6,
            "Bike Repair":   7,
        }

    def _pose_subset_from_split(self, subset_name):
        """
        Find which pose subdirectory actually contains this split's data.

        EgoExo4D's ego_pose annotations are stored in train/val/test dirs,
        but our custom train/val/test split (from convert_egoexo4d_trainvaltest.py)
        doesn't align — e.g. our "validation" videos come from the original
        training set, so their pose files live under train/, not val/.

        We return the canonical guess first, but _resolve_pose_subset() will
        try all directories at file-lookup time.
        """
        m = {
            "training": "train",
            "validation": "val",
            "testing": "test",
            "train": "train",
            "val": "val",
            "test": "test",
        }
        return m.get(str(subset_name).lower(), None)

    def _resolve_pose_subset(self, take_uid, pose_subset, kind, root=None):
        """
        Try the guessed pose_subset first, then fall back to all other
        subdirectories (train/val/test) to find the actual pose file.
        Returns the correct subset string, or None if not found anywhere.

        ``root`` lets callers override the pose root directory; defaults to
        ``self.pose_root`` for backward compatibility.
        """
        if root is None:
            root = self.pose_root
        # Try the guessed subset first
        if pose_subset is not None:
            p = os.path.join(root, pose_subset, kind, "annotation", f"{take_uid}.json")
            if os.path.exists(p):
                return pose_subset

        # Fall back: try all subdirectories
        for candidate in ("train", "val", "test"):
            if candidate == pose_subset:
                continue  # already tried
            p = os.path.join(root, candidate, kind, "annotation", f"{take_uid}.json")
            if os.path.exists(p):
                return candidate

        return None

    @staticmethod
    def _take_uid_from_key(key):
        """Extract take_uid from a database key (e.g., '{uid}_exo2' -> '{uid}')."""
        for suffix in ('_ego', '_exo1', '_exo2', '_exo3', '_exo4'):
            if key.endswith(suffix):
                return key[:-len(suffix)]
        return key

    def _load_json_db(self, json_file):
        with open(json_file, "r") as fid:
            json_data = json.load(fid)
        json_db = json_data["database"]

        if self.label_dict is None:
            label_dict = {}
            for _, value in json_db.items():
                if "annotations" not in value:
                    continue
                for act in value["annotations"]:
                    label_dict[act["label"]] = act["label_id"]
        else:
            label_dict = self.label_dict

        if self.concat_views or self.cross_view:
            return self._load_json_db_concat(json_db, label_dict)

        dict_db = tuple()
        for key, value in json_db.items():
            if value["subset"].lower() not in self.split:
                continue

            if self.egoexo_type == "ego" and value["egoexo_type"] != "ego":
                continue
            if self.egoexo_type == "exo" and value["egoexo_type"] != "exo":
                continue

            feat_files = [
                os.path.join(folder, self.file_prefix + key + self.file_ext)
                for folder in self.feat_folder
            ]
            if not all([os.path.exists(file) for file in feat_files]):
                continue

            if self.default_fps is not None:
                fps = self.default_fps
            else:
                fps = value["fps"]

            duration = value["duration"]

            if ("annotations" in value) and (len(value["annotations"]) > 0):
                num_acts = len(value["annotations"])
                reg_points = np.zeros([num_acts], dtype=np.float32)
                labels = np.zeros([num_acts], dtype=np.int64)
                for idx, act in enumerate(value["annotations"]):
                    reg_points[idx] = act["reg_point"]
                    labels[idx] = label_dict[act["label"]]
            else:
                reg_points = None
                labels = None

            take_uid = key
            if "_ego" in key:
                take_uid = key.rsplit("_ego", 1)[0]
            elif "_exo" in key:
                take_uid = key.rsplit("_exo", 1)[0]
            elif "_" in key:
                take_uid = key.rsplit("_", 1)[0]

            pose_subset = self._pose_subset_from_split(value["subset"])

            if self.pose_require_file:
                if not self._has_pose_file(take_uid, pose_subset):
                    continue

            scenario_name = value["scenario_name"]
            assert scenario_name in self.scenario_name_dict, (
                f"Unknown scenario '{scenario_name}' for video '{key}'. "
                f"Known scenarios: {list(self.scenario_name_dict.keys())}"
            )
            scenario_id = self.scenario_name_dict[scenario_name]

            dict_db += ({
                "id": key,
                "orig_video_file": value["video_file"],
                "fps": fps,
                "duration": duration,
                "reg_points": reg_points,
                "labels": labels,
                "scenario_id": scenario_id,
                "take_uid": take_uid,
                "pose_subset": pose_subset,
                "egoexo_type": value["egoexo_type"],
            },)

        return dict_db, label_dict

    def _load_json_db_concat(self, json_db, label_dict):
        """
        Load database with multi-view feature concatenation.

        One entry per take_uid with a sibling_ids list for feature loading.
        Pose is shared across views (same take_uid), so loaded once.

        For egoexo_type='exo':  concat exo1+exo2+exo3+exo4  (4 views)
        For egoexo_type='both': concat ego+exo1+exo2+exo3+exo4 (5 views)
        """
        # Step 1: Group all database keys by take_uid
        take_groups = defaultdict(dict)  # take_uid -> {view_name: (key, value)}
        for key, value in json_db.items():
            if value["subset"].lower() not in self.split:
                continue
            take_uid = self._take_uid_from_key(key)
            if key.endswith('_ego'):
                view_name = 'ego'
            else:
                view_name = key[len(take_uid) + 1:]  # e.g., 'exo1'
            take_groups[take_uid][view_name] = (key, value)

        # Step 2: Define expected views per mode
        if self.egoexo_type == 'exo':
            expected_views = ['exo1', 'exo2', 'exo3', 'exo4']
        else:  # 'both'
            expected_views = ['ego', 'exo1', 'exo2', 'exo3', 'exo4']

        # Step 3: Build entries — one per take
        dict_db = tuple()
        n_skipped_missing_views = 0
        n_skipped_missing_feats = 0
        n_skipped_missing_pose = 0

        for take_uid, views in take_groups.items():
            # Check all expected views exist in database
            if not all(v in views for v in expected_views):
                n_skipped_missing_views += 1
                continue

            # Collect ordered sibling IDs
            sibling_ids = [views[v][0] for v in expected_views]

            # Check all feature files exist for all siblings
            all_feats_exist = True
            for sid in sibling_ids:
                feat_files = [
                    os.path.join(folder, self.file_prefix + sid + self.file_ext)
                    for folder in self.feat_folder
                ]
                if not all(os.path.exists(f) for f in feat_files):
                    all_feats_exist = False
                    break
            if not all_feats_exist:
                n_skipped_missing_feats += 1
                continue

            # Use first view's metadata
            ref_key, ref_value = views[expected_views[0]]

            if self.default_fps is not None:
                fps = self.default_fps
            else:
                fps = ref_value["fps"]

            duration = ref_value["duration"]

            if ("annotations" in ref_value) and (len(ref_value["annotations"]) > 0):
                num_acts = len(ref_value["annotations"])
                reg_points = np.zeros([num_acts], dtype=np.float32)
                labels = np.zeros([num_acts], dtype=np.int64)
                for idx, act in enumerate(ref_value["annotations"]):
                    reg_points[idx] = act["reg_point"]
                    labels[idx] = label_dict[act["label"]]
            else:
                reg_points = None
                labels = None

            pose_subset = self._pose_subset_from_split(ref_value["subset"])

            if self.pose_require_file:
                if not self._has_pose_file(take_uid, pose_subset):
                    n_skipped_missing_pose += 1
                    continue

            scenario_name = ref_value["scenario_name"]
            assert scenario_name in self.scenario_name_dict, (
                f"Unknown scenario '{scenario_name}' for take '{take_uid}'. "
                f"Known scenarios: {list(self.scenario_name_dict.keys())}"
            )
            scenario_id = self.scenario_name_dict[scenario_name]

            dict_db += ({
                "id": ref_key,
                "sibling_ids": sibling_ids,
                "orig_video_file": ref_value["video_file"],
                "fps": fps,
                "duration": duration,
                "reg_points": reg_points,
                "labels": labels,
                "scenario_id": scenario_id,
                "take_uid": take_uid,
                "pose_subset": pose_subset,
                "egoexo_type": ref_value["egoexo_type"],
            },)

        logger.info(
            "concat_views: %d takes loaded, %d skipped (missing views), "
            "%d skipped (missing features), %d skipped (missing pose)",
            len(dict_db), n_skipped_missing_views,
            n_skipped_missing_feats, n_skipped_missing_pose,
        )

        return dict_db, label_dict

    def _pose_path(self, take_uid, pose_subset, kind, root=None):
        if pose_subset is None:
            return None
        if root is None:
            root = self.pose_root
        return os.path.join(
            root, pose_subset, kind, "annotation", f"{take_uid}.json"
        )

    def _has_pose_file(self, take_uid, pose_subset):
        if self.pose_type == "body":
            return self._resolve_pose_subset(take_uid, pose_subset, "body") is not None
        if self.pose_type == "hand":
            return self._resolve_pose_subset(take_uid, pose_subset, "hand") is not None
        return (
            self._resolve_pose_subset(take_uid, pose_subset, "body") is not None
            or self._resolve_pose_subset(take_uid, pose_subset, "hand") is not None
        )

    def _choose_person_frame(self, frame_data):
        if not isinstance(frame_data, list) or len(frame_data) == 0:
            return None
        best = None
        best_count = -1
        for item in frame_data:
            if not isinstance(item, dict):
                continue
            if "annotation3D" not in item:
                continue
            ann3d = item["annotation3D"]
            if not isinstance(ann3d, dict):
                continue
            count = len(ann3d)
            if count > best_count:
                best_count = count
                best = item
        if best is None:
            return frame_data[0] if isinstance(frame_data[0], dict) else None
        return best

    def _extract_pose_vector(self, frame_data, kind):
        chosen = self._choose_person_frame(frame_data)
        if chosen is None:
            return None
        if "annotation3D" not in chosen:
            return None
        ann3d = chosen["annotation3D"]
        if not isinstance(ann3d, dict):
            return None

        if kind == "body":
            joints = BODY_JOINTS
        else:
            joints = HAND_JOINTS_RIGHT + HAND_JOINTS_LEFT

        out = []
        for j in joints:
            if j in ann3d:
                out.extend([ann3d[j]["x"], ann3d[j]["y"], ann3d[j]["z"]])
            else:
                out.extend([0.0, 0.0, 0.0])
        return np.asarray(out, dtype=np.float32)

    def _load_pose_entry(self, take_uid, pose_subset, kind, root=None):
        # Cache key includes the root so body / hand can read from different
        # pose-root directories without colliding.
        cache_root = root if root is not None else self.pose_root
        key = (cache_root, pose_subset, take_uid, kind)
        if key in self._pose_cache:
            return self._pose_cache[key]

        # Resolve to actual directory (may differ from guessed subset)
        actual_subset = self._resolve_pose_subset(take_uid, pose_subset, kind, root=root)
        if actual_subset is None:
            self._pose_cache[key] = None
            return None

        path = self._pose_path(take_uid, actual_subset, kind, root=root)
        if (path is None) or (not os.path.exists(path)):
            self._pose_cache[key] = None
            return None

        try:
            with open(path, "r") as f:
                raw = json.load(f)
        except Exception:
            self._pose_cache[key] = None
            return None

        frames = []
        vecs = []
        for k, frame_data in raw.items():
            try:
                frame_idx = int(k)
            except Exception:
                continue
            vec = self._extract_pose_vector(frame_data, kind)
            if vec is None:
                continue
            frames.append(frame_idx)
            vecs.append(vec)

        if len(frames) == 0:
            self._pose_cache[key] = None
            return None

        order = np.argsort(np.asarray(frames))
        frames = np.asarray(frames, dtype=np.int32)[order]
        vecs = np.stack(vecs, axis=0).astype(np.float32)[order]
        entry = {"frames": frames, "vectors": vecs}
        self._pose_cache[key] = entry
        return entry

    def _nearest_pose_vector(self, entry, center_frame):
        if entry is None:
            return None
        frames = entry["frames"]
        vectors = entry["vectors"]
        if frames.size == 0:
            return None

        idx = int(np.searchsorted(frames, center_frame))
        candidates = []
        if idx < frames.size:
            candidates.append(idx)
        if idx > 0:
            candidates.append(idx - 1)
        if len(candidates) == 0:
            return None

        best_i = min(candidates, key=lambda i: abs(int(frames[i]) - int(center_frame)))
        if abs(int(frames[best_i]) - int(center_frame)) > self.pose_max_frame_dist:
            return None
        return vectors[best_i]

    def _normalize_pose_sequence(self, pose_sequence, kind):
        if pose_sequence is None or len(pose_sequence) == 0:
            return pose_sequence

        seq = pose_sequence.astype(np.float32, copy=True)

        if kind == "body":
            left_hip_idx, right_hip_idx = 11, 12
            left_shoulder_idx, right_shoulder_idx = 5, 6
            n_joints = 17

            left_hip = seq[:, left_hip_idx * 3:(left_hip_idx + 1) * 3]
            right_hip = seq[:, right_hip_idx * 3:(right_hip_idx + 1) * 3]
            root = (left_hip + right_hip) / 2.0

            for j in range(n_joints):
                seq[:, j * 3:(j + 1) * 3] -= root

            ls = seq[:, left_shoulder_idx * 3:(left_shoulder_idx + 1) * 3]
            rs = seq[:, right_shoulder_idx * 3:(right_shoulder_idx + 1) * 3]
            shoulder_center = (ls + rs) / 2.0
            torso_len = np.linalg.norm(shoulder_center, axis=1, keepdims=True)
            torso_len = np.clip(torso_len, 1e-2, None)

            for j in range(n_joints):
                seq[:, j * 3:(j + 1) * 3] /= torso_len

            return seq

        if kind == "hand":
            # right hand: joints 0..20
            right_wrist = seq[:, 0:3].copy()
            for j in range(21):
                seq[:, j * 3:(j + 1) * 3] -= right_wrist

            # left hand: joints 21..41
            left_wrist = seq[:, 21 * 3:21 * 3 + 3].copy()
            for j in range(21, 42):
                seq[:, j * 3:(j + 1) * 3] -= left_wrist

            right_tip = seq[:, 12 * 3:12 * 3 + 3]      # right_middle_4
            left_tip = seq[:, 33 * 3:33 * 3 + 3]       # left_middle_4

            right_span = np.linalg.norm(right_tip, axis=1, keepdims=True)
            left_span = np.linalg.norm(left_tip, axis=1, keepdims=True)
            right_span = np.clip(right_span, 1e-3, None)
            left_span = np.clip(left_span, 1e-3, None)

            for j in range(21):
                seq[:, j * 3:(j + 1) * 3] /= right_span
            for j in range(21, 42):
                seq[:, j * 3:(j + 1) * 3] /= left_span

            return seq

        # body_hand: normalize each block independently
        b = seq[:, :51]
        h = seq[:, 51:]
        b = self._normalize_pose_sequence(b, "body")
        h = self._normalize_pose_sequence(h, "hand")
        return np.concatenate([b, h], axis=-1)

    def _load_pose_sequence(self, take_uid, pose_subset, num_tokens, feat_stride):
        raw_dim = self._raw_pose_dim
        pose_raw = np.zeros((num_tokens, raw_dim), dtype=np.float32)
        pose_valid = np.zeros((num_tokens,), dtype=np.bool_)

        if (take_uid is None) or (pose_subset is None):
            return np.zeros((num_tokens, self.pose_dim), dtype=np.float32), pose_valid

        centers = np.round(
            np.arange(num_tokens, dtype=np.float32) * float(feat_stride) + 0.5 * float(self.num_frames)
        ).astype(np.int64)

        if self.pose_type in ("body", "hand"):
            entry = self._load_pose_entry(take_uid, pose_subset, self.pose_type)
            if entry is None:
                return np.zeros((num_tokens, self.pose_dim), dtype=np.float32), pose_valid

            for i, c in enumerate(centers):
                v = self._nearest_pose_vector(entry, int(c))
                if v is None:
                    continue
                pose_raw[i] = v
                pose_valid[i] = True

            if self.pose_normalize and pose_valid.any():
                pose_raw[pose_valid] = self._normalize_pose_sequence(
                    pose_raw[pose_valid], self.pose_type
                )

            # Append kinematic features if enabled
            # Compute on the full sequence (not just valid) so velocity
            # reflects true temporal differences, then zero out invalid.
            if self._kin_dim > 0 and self.pose_type == "body":
                kin = compute_body_kinematics(pose_raw, normalize=True)  # (T, 67)
                kin[~pose_valid] = 0.0
                pose_seq = np.concatenate([pose_raw, kin], axis=-1)
            else:
                pose_seq = pose_raw

            return pose_seq, pose_valid

        # body_hand
        body_entry = self._load_pose_entry(take_uid, pose_subset, "body")
        hand_entry = self._load_pose_entry(take_uid, pose_subset, "hand")

        body_valid = np.zeros((num_tokens,), dtype=np.bool_)
        hand_valid = np.zeros((num_tokens,), dtype=np.bool_)

        for i, c in enumerate(centers):
            vb = self._nearest_pose_vector(body_entry, int(c))
            vh = self._nearest_pose_vector(hand_entry, int(c))

            if vb is not None:
                pose_raw[i, :51] = vb
                body_valid[i] = True
            if vh is not None:
                pose_raw[i, 51:] = vh
                hand_valid[i] = True

        pose_valid = np.logical_or(body_valid, hand_valid)

        if self.pose_normalize:
            if body_valid.any():
                pose_raw[body_valid, :51] = self._normalize_pose_sequence(
                    pose_raw[body_valid, :51], "body"
                )
            if hand_valid.any():
                pose_raw[hand_valid, 51:] = self._normalize_pose_sequence(
                    pose_raw[hand_valid, 51:], "hand"
                )

        # Append kinematic features for body part
        # Compute on full sequence for correct velocity, then zero invalid
        if self._kin_dim > 0:
            kin = compute_body_kinematics(pose_raw[:, :51], normalize=True)  # (T, 67)
            kin[~body_valid] = 0.0
            pose_seq = np.concatenate([pose_raw, kin], axis=-1)
        else:
            pose_seq = pose_raw

        return pose_seq, pose_valid

    def _load_hand_pose_sequence(self, take_uid, pose_subset, num_tokens, feat_stride):
        """Load hand-only pose into a (num_tokens, self.hand_dim) array + (num_tokens,)
        validity mask. Independent of `self.pose_type` — used by the parallel hand
        stream when load_hand_pose=True.
        """
        out = np.zeros((num_tokens, self.hand_dim), dtype=np.float32)
        valid = np.zeros((num_tokens,), dtype=np.bool_)

        if (take_uid is None) or (pose_subset is None):
            return out, valid

        entry = self._load_pose_entry(
            take_uid, pose_subset, "hand", root=self.hand_pose_root
        )
        if entry is None:
            return out, valid

        centers = np.round(
            np.arange(num_tokens, dtype=np.float32) * float(feat_stride)
            + 0.5 * float(self.num_frames)
        ).astype(np.int64)

        raw = np.zeros((num_tokens, self.hand_raw_dim), dtype=np.float32)
        for i, c in enumerate(centers):
            v = self._nearest_pose_vector(entry, int(c))
            if v is None:
                continue
            raw[i] = v
            valid[i] = True

        if self.pose_normalize and valid.any():
            raw[valid] = self._normalize_pose_sequence(raw[valid], "hand")

        if self.hand_kin_dim > 0:
            # Compute kinematics on the full sequence (so velocities reflect true
            # frame-to-frame deltas), then zero out invalid tokens.
            kin = compute_hand_kinematics(raw, normalize=True)  # (T, 30)
            kin[~valid] = 0.0
            out[:, : self.hand_raw_dim] = raw
            out[:, self.hand_raw_dim :] = kin
        else:
            out[:] = raw

        return out, valid

    def _truncate_feats_with_pose(
        self,
        data_dict,
        feat_offset,
        max_num_trials=200,
        has_action=True
    ):
        feat_len = data_dict["feats"].shape[1]
        target_len = self.max_seq_len

        if feat_len <= self.max_seq_len:
            if self.crop_ratio is None:
                return data_dict
            target_len = random.randint(
                max(round(self.crop_ratio[0] * feat_len), 1),
                min(round(self.crop_ratio[1] * feat_len), feat_len),
            )
            if target_len == feat_len:
                return data_dict

        data_dict = copy.deepcopy(data_dict)
        reg_points = data_dict["reg_points"]

        st = 0
        ed = target_len
        seg_idx = None

        for _ in range(max_num_trials):
            st = random.randint(0, feat_len - target_len)
            ed = st + target_len

            if reg_points is None or reg_points.numel() == 0:
                seg_idx = None
                break

            seg_idx = torch.logical_and(
                reg_points >= (st - feat_offset),
                reg_points <= (ed + feat_offset),
            )

            if has_action:
                if seg_idx.sum().item() > 0:
                    break
            else:
                break

        data_dict["feats"] = data_dict["feats"][:, st:ed].clone()
        data_dict["pose_feats"] = data_dict["pose_feats"][:, st:ed].clone()
        data_dict["pose_valid"] = data_dict["pose_valid"][st:ed].clone()
        # cross-view features: truncate identically if present
        if "feats_ego" in data_dict:
            data_dict["feats_ego"] = data_dict["feats_ego"][:, st:ed].clone()
        if "feats_exo" in data_dict:
            data_dict["feats_exo"] = data_dict["feats_exo"][:, st:ed].clone()
        # parallel hand-pose stream: truncate identically if present
        if "hand_feats" in data_dict:
            data_dict["hand_feats"] = data_dict["hand_feats"][:, st:ed].clone()
        if "hand_valid" in data_dict:
            data_dict["hand_valid"] = data_dict["hand_valid"][st:ed].clone()

        if reg_points is not None:
            if seg_idx is None:
                data_dict["reg_points"] = reg_points - st
            else:
                data_dict["reg_points"] = reg_points[seg_idx].clone() - st
                if data_dict["labels"] is not None:
                    data_dict["labels"] = data_dict["labels"][seg_idx].clone()

        return data_dict

    def _normalize_view_feats(self, feat):
        """
        Normalize a single view's features according to self.concat_norm.
        feat: (T, C) array
        Returns: (T, C) normalized array
        """
        if self.concat_norm == "l2":
            norms = np.linalg.norm(feat, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-6)
            return feat / norms
        elif self.concat_norm == "zscore":
            # Per-feature z-score across the time dimension
            mu = feat.mean(axis=0, keepdims=True)     # (1, C)
            std = feat.std(axis=0, keepdims=True)      # (1, C)
            std = np.maximum(std, 1e-6)
            return (feat - mu) / std
        else:  # "none"
            return feat

    def _preload_features(self):
        """Pre-load all .npy feature files into memory at init time."""
        paths = set()
        for item in self.data_list:
            if self.concat_views or self.cross_view:
                for sid in item["sibling_ids"]:
                    for folder in self.feat_folder:
                        paths.add(os.path.join(folder, self.file_prefix + sid + self.file_ext))
            else:
                for folder in self.feat_folder:
                    paths.add(os.path.join(folder, self.file_prefix + item["id"] + self.file_ext))

        for p in paths:
            self._feat_cache[p] = np.load(p).astype(np.float32)

        total_mb = sum(a.nbytes for a in self._feat_cache.values()) / (1024 * 1024)
        logger.info("Pre-loaded %d feature files (%.1f MB) into memory", len(self._feat_cache), total_mb)

    def _load_npy_cached(self, filepath):
        """Load a .npy file with in-memory caching. Returns float32 array."""
        if filepath not in self._feat_cache:
            self._feat_cache[filepath] = np.load(filepath).astype(np.float32)
        return self._feat_cache[filepath]

    def _load_concat_view_feats(self, video_item):
        """
        Load and concatenate features from all sibling views.
        Each view is normalized independently before concatenation
        to prevent distribution mismatch between ego and exo features.
        Returns (T, C*num_views) array.
        """
        sibling_ids = video_item["sibling_ids"]
        view_feats = []
        for sid in sibling_ids:
            filenames = [
                os.path.join(folder, self.file_prefix + sid + self.file_ext)
                for folder in self.feat_folder
            ]
            feat = np.concatenate(
                [self._load_npy_cached(name) for name in filenames], axis=1
            )
            feat = feat.squeeze()
            feat = self._normalize_view_feats(feat)
            view_feats.append(feat)

        # All views from same take should have same T; truncate to min for safety
        min_t = min(f.shape[0] for f in view_feats)
        view_feats = [f[:min_t] for f in view_feats]

        return np.concatenate(view_feats, axis=1)  # (T, C*num_views)

    def _load_cross_view_feats(self, video_item):
        """
        Load ego and exo features separately for cross-view attention.
        Returns:
            ego_feats: (T, C) array — ego view features
            exo_feats: (T, C) array — average of exo1-exo4 features
        """
        sibling_ids = video_item["sibling_ids"]
        ego_id = sibling_ids[0]    # ego
        exo_ids = sibling_ids[1:]  # exo1, exo2, exo3, exo4

        def _load_single(sid):
            filenames = [
                os.path.join(folder, self.file_prefix + sid + self.file_ext)
                for folder in self.feat_folder
            ]
            feat = np.concatenate(
                [self._load_npy_cached(name) for name in filenames], axis=1
            )
            return feat.squeeze()

        ego_feats = _load_single(ego_id)

        exo_feats_list = [_load_single(sid) for sid in exo_ids]
        min_t = min(f.shape[0] for f in exo_feats_list)
        exo_feats_list = [f[:min_t] for f in exo_feats_list]
        exo_feats = np.mean(np.stack(exo_feats_list, axis=0), axis=0)

        min_t = min(ego_feats.shape[0], exo_feats.shape[0])
        ego_feats = ego_feats[:min_t]
        exo_feats = exo_feats[:min_t]

        return ego_feats, exo_feats

    def __getitem__(self, idx):
        video_item = self.data_list[idx]

        # Load video features
        if self.cross_view:
            ego_feats_np, exo_feats_np = self._load_cross_view_feats(video_item)
            ego_feats_np = ego_feats_np[::self.downsample_rate, :]
            exo_feats_np = exo_feats_np[::self.downsample_rate, :]
            feats = ego_feats_np  # use ego as primary for T calculation
        elif self.concat_views:
            feats = self._load_concat_view_feats(video_item)
        else:
            filenames = [
                os.path.join(folder, self.file_prefix + video_item["id"] + self.file_ext)
                for folder in self.feat_folder
            ]
            feats = np.concatenate([self._load_npy_cached(name) for name in filenames], axis=1)
            feats = feats.squeeze()
            if self.concat_norm != "none":
                feats = self._normalize_view_feats(feats)

        if not self.cross_view:
            if feats.ndim == 3 and feats.shape[1] == 1:
                feats = feats.squeeze(1)
            if feats.ndim != 2:
                raise ValueError(f"Expected feature shape (T,C), got {feats.shape} for {video_item['id']}")
            feats = feats[::self.downsample_rate, :]

        feat_stride = self.feat_stride * self.downsample_rate
        feat_offset = 0.5 * self.num_frames / feat_stride

        num_tokens = feats.shape[0]
        pose_seq, pose_valid = self._load_pose_sequence(
            take_uid=video_item["take_uid"],
            pose_subset=video_item["pose_subset"],
            num_tokens=num_tokens,
            feat_stride=feat_stride,
        )

        feats = torch.from_numpy(np.ascontiguousarray(feats.transpose()))               # C_video x T
        pose_feats = torch.from_numpy(np.ascontiguousarray(pose_seq.transpose()))       # C_pose x T
        pose_valid = torch.from_numpy(pose_valid.astype(np.bool_))                      # T

        # ── Optional hand-pose side stream ──
        if self.load_hand_pose:
            hand_seq, hand_valid_np = self._load_hand_pose_sequence(
                take_uid=video_item["take_uid"],
                pose_subset=video_item["pose_subset"],
                num_tokens=num_tokens,
                feat_stride=feat_stride,
            )
            hand_feats = torch.from_numpy(np.ascontiguousarray(hand_seq.transpose()))   # C_hand x T
            hand_valid = torch.from_numpy(hand_valid_np.astype(np.bool_))               # T
        else:
            hand_feats = None
            hand_valid = None

        if video_item["reg_points"] is not None:
            reg_points = torch.from_numpy(
                video_item["reg_points"] * video_item["fps"] / feat_stride - feat_offset
            )
            labels = torch.from_numpy(video_item["labels"])
        else:
            reg_points, labels = None, None

        if video_item.get("egoexo_type") == "exo":
            view_type = 1
        else:
            view_type = 0

        data_dict = {
            "video_id": video_item["id"],
            "view_type": view_type,
            "orig_video_file": video_item["orig_video_file"],
            "feats": feats,
            "pose_feats": pose_feats,
            "pose_valid": pose_valid,
            "reg_points": reg_points,
            "labels": labels,
            "fps": video_item["fps"],
            "duration": video_item["duration"],
            "feat_stride": feat_stride,
            "feat_num_frames": self.num_frames,
            "scenario_id": video_item["scenario_id"],
        }

        if self.cross_view:
            data_dict["feats_ego"] = torch.from_numpy(
                np.ascontiguousarray(ego_feats_np.transpose()))   # C x T
            data_dict["feats_exo"] = torch.from_numpy(
                np.ascontiguousarray(exo_feats_np.transpose()))   # C x T

        if hand_feats is not None:
            data_dict["hand_feats"] = hand_feats
            data_dict["hand_valid"] = hand_valid

        if self.is_training and (reg_points is not None):
            data_dict = self._truncate_feats_with_pose(
                data_dict=data_dict,
                feat_offset=feat_offset,
            )

        return data_dict

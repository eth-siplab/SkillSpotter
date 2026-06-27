#!/usr/bin/env python3
"""
Create a demonstrator-proficiency dataset manifest and Omnivore feature clips.

This script is intentionally separate from convert_egoexo4d_trainvaltest.py so
the demonstration-task preprocessing remains untouched.

Outputs:
  1) A JSON in EgoExo4D "database" format that includes both ego and exo views.
  2) .npy Omnivore features (T x C) for each included view key:
       {take_uid}_ego, {take_uid}_exo1, {take_uid}_exo2, {take_uid}_exo3, {take_uid}_exo4
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


@dataclass
class BuildStats:
    takes_total: int = 0
    takes_train: int = 0
    takes_test: int = 0
    takes_missing_metadata: int = 0
    takes_bad_fps: int = 0
    entries_written: int = 0
    entries_skipped_missing_feature: int = 0
    entries_skipped_missing_video_path: int = 0
    entries_skipped_decode_error: int = 0
    entries_skipped_no_frames: int = 0
    entries_shorter_than_expected: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build demonstrator-only EgoExo4D json + Omnivore npy features"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/path/to/EgoExo4D",
        help="EgoExo4D root directory",
    )
    parser.add_argument(
        "--train_json",
        type=str,
        default="annotations/proficiency_demonstrator_train.json",
        help="Path relative to --base_dir or absolute",
    )
    parser.add_argument(
        "--val_json",
        type=str,
        default="annotations/proficiency_demonstrator_val.json",
        help="Path relative to --base_dir or absolute",
    )
    parser.add_argument(
        "--takes_json",
        type=str,
        default="takes.json",
        help="Path relative to --base_dir or absolute",
    )
    parser.add_argument(
        "--omnivore_src_dir",
        type=str,
        default="features/omnivore_video",
        help="Directory with source Omnivore .pt files (relative or absolute)",
    )
    parser.add_argument(
        "--omnivore_out_dir",
        type=str,
        default="features/omnivore_features_demonstrator",
        help="Directory to write output .npy features (relative or absolute)",
    )
    parser.add_argument(
        "--out_json",
        type=str,
        default="annotations/egoexo4d_demonstrator.json",
        help="Output dataset json path (relative to --base_dir or absolute)",
    )
    parser.add_argument(
        "--views",
        nargs="+",
        default=["ego", "exo1", "exo2", "exo3", "exo4"],
        help="Views to include",
    )
    parser.add_argument(
        "--clip_size",
        type=int,
        default=32,
        help="Clip size used in source feature extraction",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=16,
        help="Stride used in source feature extraction",
    )
    parser.add_argument(
        "--min_fps",
        type=float,
        default=10.0,
        help="Minimum allowed FPS for a take",
    )
    parser.add_argument(
        "--max_fps",
        type=float,
        default=100.0,
        help="Maximum allowed FPS for a take",
    )
    parser.add_argument(
        "--check_downscaled_video_exists",
        action="store_true",
        help="If set, verify downscaled/448 video path exists; skip entry if missing",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .npy files",
    )
    return parser.parse_args()


def resolve_path(base_dir: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(base_dir, path)


def load_annotations(path: str) -> List[dict]:
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict) and ("annotations" in data) and isinstance(data["annotations"], list):
        return data["annotations"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported annotation format: {path}")


def load_takes_metadata(path: str) -> Dict[str, dict]:
    with open(path, "r") as f:
        takes = json.load(f)
    return {t["take_uid"]: t for t in takes}


def to_downscaled_448_path(video_rel_path: str) -> str:
    parts = video_rel_path.split("/")
    if len(parts) < 2:
        return video_rel_path
    parts.insert(-1, "downscaled/448")
    return "/".join(parts)


def camera_name_from_video_path(view_key: str, video_rel_path: str) -> str:
    name = os.path.basename(video_rel_path).split(".")[0]
    if view_key == "ego":
        return name[:4]
    return name


def find_source_feature_file(
    src_dir: str,
    take_uid: str,
    view_key: str,
    video_rel_path: str,
) -> Optional[str]:
    cam_name = camera_name_from_video_path(view_key, video_rel_path)
    if view_key == "ego":
        pattern = os.path.join(src_dir, f"{take_uid}_{cam_name}*")
    else:
        pattern = os.path.join(src_dir, f"{take_uid}_{cam_name}_*")

    matches = sorted(glob.glob(pattern))
    if len(matches) == 0:
        return None
    if len(matches) == 1:
        return matches[0]

    rgb_matches = [m for m in matches if "_rgb." in m or m.endswith("_rgb.pt")]
    if len(rgb_matches) == 1:
        return rgb_matches[0]
    if len(rgb_matches) > 1:
        return sorted(rgb_matches)[0]

    # Fallback: deterministic selection
    return matches[0]


def compute_window_count(
    total_frames: int,
    fps: float,
    clip_size: int,
    stride: int,
) -> Tuple[int, float]:
    """
    Returns:
      n_windows: expected number of feature windows
      duration_sec_with_padding: duration after frame padding logic
    """
    if total_frames < clip_size:
        return 0, float(total_frames) / fps if fps > 0 else 0.0

    append_frames = 0
    remainder = (total_frames - clip_size) % stride
    if remainder:
        append_frames = stride - remainder

    padded_frames = total_frames + append_frames
    duration_sec = float(padded_frames) / fps
    n_windows = (padded_frames - clip_size) // stride + 1
    return int(n_windows), duration_sec


def build_entry(
    take_uid: str,
    view_key: str,
    video_rel_path: str,
    subset: str,
    fps: float,
    duration_sec: float,
    original_duration_sec: float,
    take_ann: dict,
) -> dict:
    egoexo_type = "ego" if view_key == "ego" else "exo"
    return {
        "video_file": to_downscaled_448_path(video_rel_path),
        "subset": subset,  # "training" or "testing"
        "egoexo_type": egoexo_type,
        "duration": round(float(duration_sec), 2),
        "original_duration": round(float(original_duration_sec), 2),
        "fps": round(float(fps), 2),
        # No timestamp supervision for demonstrator task in this manifest.
        "annotations": [],
        "good_executions": [],
        "tips_for_improvement": [],
        "task_name": take_ann["task_name"],
        "scenario_name": take_ann["scenario_name"],
        "take_uid": take_uid,
        "origin_participant_id": str(take_ann["origin_participant_id"]),
        "proficiency_score": take_ann["proficiency_score"],
    }


def progress_wrap(iterable, total: int, desc: str):
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True)


def main() -> None:
    args = parse_args()

    base_dir = args.base_dir
    train_json = resolve_path(base_dir, args.train_json)
    val_json = resolve_path(base_dir, args.val_json)
    takes_json = resolve_path(base_dir, args.takes_json)
    src_dir = resolve_path(base_dir, args.omnivore_src_dir)
    out_dir = resolve_path(base_dir, args.omnivore_out_dir)
    out_json = resolve_path(base_dir, args.out_json)

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(out_json), exist_ok=True)

    print("=" * 80)
    print("Create EgoExo4D Demonstrator Omnivore Dataset")
    print("=" * 80)
    print(f"base_dir:          {base_dir}")
    print(f"train_json:        {train_json}")
    print(f"val_json:          {val_json}")
    print(f"takes_json:        {takes_json}")
    print(f"omnivore_src_dir:  {src_dir}")
    print(f"omnivore_out_dir:  {out_dir}")
    print(f"out_json:          {out_json}")
    print(f"views:             {args.views}")
    print(f"clip/stride:       {args.clip_size}/{args.stride}")
    print(f"fps range:         [{args.min_fps}, {args.max_fps}]")
    print(f"overwrite:         {args.overwrite}")
    print(f"check videos:      {args.check_downscaled_video_exists}")
    print()

    takes_meta = load_takes_metadata(takes_json)
    train_anns = load_annotations(train_json)
    val_anns = load_annotations(val_json)

    stats = BuildStats()
    stats.takes_train = len(train_anns)
    stats.takes_test = len(val_anns)
    stats.takes_total = stats.takes_train + stats.takes_test

    all_anns: List[Tuple[str, dict]] = [("training", a) for a in train_anns] + [
        ("testing", a) for a in val_anns
    ]
    if tqdm is None:
        print("tqdm not available; running without progress bar.")
    missing_meta_uids: List[str] = []
    missing_feature_by_view: Counter = Counter()
    written_by_view: Counter = Counter()
    written_by_subset: Counter = Counter()
    seen_cids = set()

    database: Dict[str, dict] = {}

    ann_iter = progress_wrap(all_anns, total=len(all_anns), desc="Processing takes")
    for subset, ann in ann_iter:
        take_uid = ann["take_uid"]

        meta = takes_meta.get(take_uid)
        if meta is None:
            stats.takes_missing_metadata += 1
            missing_meta_uids.append(take_uid)
            continue

        frames = int(meta["timesync_end_idx"]) - int(meta["timesync_start_idx"])
        original_duration = float(meta["duration_sec"])
        if original_duration <= 0:
            stats.takes_bad_fps += 1
            continue
        fps = frames / original_duration
        if not (args.min_fps <= fps <= args.max_fps):
            stats.takes_bad_fps += 1
            continue

        n_windows_expected, duration_padded = compute_window_count(
            total_frames=frames,
            fps=fps,
            clip_size=args.clip_size,
            stride=args.stride,
        )
        if n_windows_expected <= 0:
            stats.entries_skipped_no_frames += 1
            continue

        video_paths = ann["video_paths"]
        if not isinstance(video_paths, dict):
            raise TypeError(f"Annotation for take {take_uid} has non-dict video_paths: {type(video_paths)}")

        for view_key in args.views:
            if view_key not in video_paths:
                stats.entries_skipped_missing_video_path += 1
                continue
            video_rel_path = video_paths[view_key]

            cid = f"{take_uid}_{view_key}"
            if cid in seen_cids:
                # Prefer first occurrence deterministically (should not happen in normal data).
                continue

            src_feature = find_source_feature_file(src_dir, take_uid, view_key, video_rel_path)
            if src_feature is None:
                stats.entries_skipped_missing_feature += 1
                missing_feature_by_view[view_key] += 1
                continue

            try:
                feat = torch.load(src_feature, map_location="cpu")
                if isinstance(feat, torch.Tensor):
                    feat = feat.numpy()
                feat = np.asarray(feat)
            except Exception:
                stats.entries_skipped_decode_error += 1
                continue

            # Squeeze singleton dims: [T, 1, C] -> [T, C]
            if feat.ndim == 3 and feat.shape[1] == 1:
                feat = feat[:, 0, :]
            if feat.ndim == 1:
                feat = feat[None, :]
            if feat.ndim != 2:
                stats.entries_skipped_decode_error += 1
                continue

            n_windows = min(int(feat.shape[0]), n_windows_expected)
            if n_windows <= 0:
                stats.entries_skipped_no_frames += 1
                continue
            if n_windows < n_windows_expected:
                stats.entries_shorter_than_expected += 1

            out_feat_path = os.path.join(out_dir, f"{cid}.npy")
            if args.overwrite or (not os.path.exists(out_feat_path)):
                np.save(out_feat_path, feat[:n_windows].astype(np.float32))

            duration_for_entry = duration_padded
            if n_windows < n_windows_expected:
                padded_frames = (n_windows - 1) * args.stride + args.clip_size
                duration_for_entry = padded_frames / fps

            entry = build_entry(
                take_uid=take_uid,
                view_key=view_key,
                video_rel_path=video_rel_path,
                subset=subset,
                fps=fps,
                duration_sec=duration_for_entry,
                original_duration_sec=original_duration,
                take_ann=ann,
            )

            if args.check_downscaled_video_exists:
                video_abs = os.path.join(base_dir, entry["video_file"])
                if not os.path.exists(video_abs):
                    continue

            database[cid] = entry
            seen_cids.add(cid)
            stats.entries_written += 1
            written_by_view[view_key] += 1
            written_by_subset[subset] += 1

        if tqdm is not None and hasattr(ann_iter, "set_postfix"):
            ann_iter.set_postfix(
                written=stats.entries_written,
                miss_feat=stats.entries_skipped_missing_feature,
                miss_view=stats.entries_skipped_missing_video_path,
            )

    with open(out_json, "w") as f:
        json.dump({"version": "v1", "database": database}, f)

    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Takes total (ann):                 {stats.takes_total}")
    print(f"  train takes:                     {stats.takes_train}")
    print(f"  val takes:                       {stats.takes_test}")
    print(f"Takes missing metadata:            {stats.takes_missing_metadata}")
    print(f"Takes with invalid FPS:            {stats.takes_bad_fps}")
    print(f"Entries written:                   {stats.entries_written}")
    print(f"Entries skipped (missing view):    {stats.entries_skipped_missing_video_path}")
    print(f"Entries skipped (missing feature): {stats.entries_skipped_missing_feature}")
    print(f"Entries skipped (decode error):    {stats.entries_skipped_decode_error}")
    print(f"Entries skipped (no frames):       {stats.entries_skipped_no_frames}")
    print(f"Entries shorter than expected:     {stats.entries_shorter_than_expected}")
    print()
    print("Written by subset:")
    for subset, cnt in sorted(written_by_subset.items()):
        print(f"  {subset}: {cnt}")
    print("Written by view:")
    for view, cnt in sorted(written_by_view.items()):
        print(f"  {view}: {cnt}")

    if missing_feature_by_view:
        print("Missing source features by view:")
        for view, cnt in sorted(missing_feature_by_view.items()):
            print(f"  {view}: {cnt}")

    if missing_meta_uids:
        print(f"Missing metadata take_uids (first 20/{len(missing_meta_uids)}):")
        for uid in missing_meta_uids[:20]:
            print(f"  {uid}")

    print()
    print(f"Wrote JSON: {out_json}")
    print(f"Feature dir: {out_dir}")


if __name__ == "__main__":
    main()

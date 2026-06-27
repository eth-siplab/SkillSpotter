"""
Predict ego body pose using the EgoExo4D baseline model.

This script runs the official EgoExo4D ego body pose baseline
(BoDiffusion/AvatarPoser-based transformer) on all takes that have
camera_pose (aria trajectory) data. The model takes aria camera
trajectories as input and predicts 17 COCO body joints in 3D.

Output is saved in the same JSON format as the GT ego_pose annotations
so it can be used as a drop-in replacement via a different pose_root.

Prerequisites:
    1. Clone the baseline repo:
       git clone https://github.com/EGO4D/ego-exo4d-egopose.git
       cd ego-exo4d-egopose/bodypose
       conda env create -f environment.yml

    2. Download pretrained weights:
       pip install gdown
       gdown 1XpY7aa7I7XFNDM6tJPcyS17xPsDlW0g7
       mkdir -p results/EgoExo4D/models/
       mv 100000_G.pth results/EgoExo4D/models/

    3. Run:
       # Single GPU
       python tools/predict_ego_pose.py --device cuda:0

       # Multi-GPU (specific GPUs, processes takes in parallel)
       python tools/predict_ego_pose.py --gpus 0 1 2 3 4 5 6 7
"""

import argparse
import json
import os
import sys
import time
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
import multiprocessing as mp

import torch

try:
    import orjson
except ImportError:
    orjson = None


# ── COCO 17-joint ordering (same as EgoExo4D GT annotations) ──
BODY_JOINTS = [
    "nose",
    "left-eye", "right-eye",
    "left-ear", "right-ear",
    "left-shoulder", "right-shoulder",
    "left-elbow", "right-elbow",
    "left-wrist", "right-wrist",
    "left-hip", "right-hip",
    "left-knee", "right-knee",
    "left-ankle", "right-ankle",
]


# ======================================================================
# Config loading (handles // comments in baseline JSON configs)
# ======================================================================

def load_json_with_comments(path):
    """
    Load a JSON file that may contain // comments (like the baseline configs).
    Replicates the baseline's own utils_option.parse() logic.
    """
    json_str = ""
    with open(path, "r") as f:
        for line in f:
            # Strip everything after // (baseline convention)
            line = line.split("//")[0] + "\n"
            json_str += line
    return json.loads(json_str, object_pairs_hook=OrderedDict)


def load_json_file(path):
    """Fast JSON loader (orjson if available, else stdlib json)."""
    if orjson is not None:
        with open(path, "rb") as f:
            return orjson.loads(f.read())
    with open(path, "r") as f:
        return json.load(f)


def dump_json_file(path, obj):
    """Fast JSON writer (orjson if available, else stdlib json)."""
    if orjson is not None:
        with open(path, "wb") as f:
            f.write(orjson.dumps(obj))
        return
    with open(path, "w") as f:
        json.dump(obj, f, separators=(",", ":"))


# ======================================================================
# Model loading — always uses the baseline repo
# ======================================================================

def load_model_from_repo(baseline_repo, checkpoint_path, device="cuda:0"):
    """
    Load the baseline model using the repo's own code.
    This guarantees architecture match with the pretrained weights.

    We replicate the essential setup from the baseline's utils_option.parse()
    which populates opt['path'], opt['is_train'], etc. before the model
    constructor accesses them.
    """
    # Add baseline repo to path
    sys.path.insert(0, baseline_repo)
    from models.select_model import define_Model

    # Load the original inference config (with // comment handling)
    opt_path = os.path.join(baseline_repo, "options", "inference_egoexo.json")
    opt = load_json_with_comments(opt_path)

    # Override checkpoint path
    opt["path"]["pretrained_netG"] = checkpoint_path

    # Set GPU — extract index from device string (e.g. "cuda:7" → 7)
    # This ensures DataParallel (if used) targets the correct device
    if "cuda" in device:
        gpu_idx = int(device.split(":")[-1]) if ":" in device else 0
    else:
        gpu_idx = None
    opt["gpu_ids"] = [gpu_idx] if gpu_idx is not None else None

    # ── Populate keys that parse() normally sets ──
    # (model_base.__init__ and other code expect these)
    opt["is_train"] = False
    opt["opt_path"] = opt_path
    opt["merge_bn"] = False
    opt["merge_bn_startpoint"] = -1
    opt["scale"] = 1
    opt["num_gpu"] = 1
    opt["find_unused_parameters"] = True
    opt["dist"] = False

    # Build path entries that parse() constructs from root + task
    root = opt["path"].get("root", "results")
    task = opt.get("task", "EgoExo4D")
    task_path = os.path.join(baseline_repo, root, task)
    opt["path"]["task"] = task_path
    opt["path"]["log"] = task_path
    opt["path"]["options"] = os.path.join(task_path, "options")
    opt["path"]["models"] = os.path.join(task_path, "models")
    opt["path"]["images"] = os.path.join(task_path, "inference")
    opt["path"]["pretrained"] = checkpoint_path

    # model_egoexo.load() also tries to load an EMA network (netE).
    # We don't have / need an EMA checkpoint — set to None so load() skips it.
    opt["path"].setdefault("pretrained_netE", None)

    # Ensure train defaults exist (some code paths check these)
    if "train" not in opt:
        opt["train"] = OrderedDict()
    train_defaults = {
        "G_optimizer_reuse": False,
        "G_param_strict": True,
        "E_param_strict": True,
        "E_decay": 0,
    }
    for k, v in train_defaults.items():
        opt["train"].setdefault(k, v)

    # model_egoexo.__init__ reads window_size from netG, but it's only
    # defined under datasets.test in the config. Copy it over.
    ds_window = opt.get("datasets", {}).get("test", {}).get("window_size", 20)
    opt["netG"].setdefault("window_size", ds_window)

    # model_egoexo.load() checks opt['netG']['video_model'] to decide
    # strict loading.  Body-pose models are NOT video models, so set False
    # → strict=True (standard weight loading).
    opt["netG"].setdefault("video_model", False)

    # Create model (architecture only — weights NOT loaded yet)
    model = define_Model(opt)

    # Load pretrained weights — CRITICAL: define_Model only builds the
    # architecture. init_test() calls self.load(test=True) which loads
    # the checkpoint from opt['path']['pretrained_netG'].
    # Without this, the model runs with random weights!
    model.init_test()

    # Unwrap DataParallel if the baseline wrapped it — we do single-GPU inference
    if isinstance(model.netG, torch.nn.DataParallel):
        model.netG = model.netG.module

    model.netG = model.netG.to(device)
    model.device = torch.device(device)
    model.netG.eval()

    # Extract window_size from config
    window_size = opt.get("datasets", {}).get("test", {}).get("window_size", 20)

    # Extract coord setting from config
    coord = opt.get("datasets", {}).get("test", {}).get("coord", None)

    print(f"  Model loaded from baseline repo")
    print(f"  Config window_size: {window_size}")
    print(f"  Config coord: {coord}")

    return model, opt, window_size, coord


# ======================================================================
# Camera trajectory extraction (matches baseline's translate_poses)
# ======================================================================

def load_camera_trajectory(camera_pose_path, coord=None):
    """
    Load aria camera trajectory from camera_pose JSON.
    Replicates the baseline's Dataset_EgoExo.translate_poses() trajectory logic.

    The camera_pose file has structure:
    {
        "metadata": {"take_name": ..., "take_uid": ...},
        "aria01": {
            "camera_intrinsics": [[3x3]],
            "camera_extrinsics": {"frame_idx": [[3x4]], ...}
        },
        "cam01": {...}, ...
    }

    The baseline computes the trajectory as T_world_camera[:3, 3] where:
      - coord == 'global': T_world_camera = inv(extrinsic)
      - coord == 'aria':   T_world_camera = first_extrinsic @ inv(extrinsic)
      - coord == None:     T_world_camera = extrinsic (raw)

    IMPORTANT: The pretrained model was trained with coord=null (per-frame
    normalization using raw extrinsics). For inference:
    1. Always feed the model coord=null trajectory (what it was trained on)
    2. If you want output in a different coord system, use the returned
       per-frame transforms to convert predictions afterward.

    Returns:
        trajectory: np.array of shape (N_frames, 3) — model input trajectory
        frame_indices: sorted list of integer frame indices
        transforms: np.array of shape (N_frames, 4, 4) — per-frame 4x4
            transforms to convert from model output coord (null/camera)
            to the requested coord system. If coord is None, these are
            identity matrices (no transform needed).
    """
    data = load_json_file(camera_pose_path)

    # Find aria camera key
    aria_key = None
    for k in data.keys():
        if "aria" in k.lower():
            aria_key = k
            break
    if aria_key is None:
        raise ValueError(f"No aria camera found in {camera_pose_path}")

    extrinsics = data[aria_key]["camera_extrinsics"]

    # Sort frame indices numerically
    frame_keys = sorted(extrinsics.keys(), key=int)
    n_frames = len(frame_keys)

    if n_frames == 0:
        return np.zeros((0, 3), dtype=np.float32), [], np.zeros((0, 4, 4), dtype=np.float64)

    # Build the first-frame transform for 'aria' coord mode
    first_ext = np.eye(4, dtype=np.float64)
    first_ext[:3, :] = np.array(extrinsics[frame_keys[0]], dtype=np.float64)

    # Always compute the model-input trajectory with coord=None (raw extrinsic)
    # since the pretrained model was trained with coord=null.
    # Additionally compute per-frame transforms to convert predictions to the
    # requested output coord system.
    trajectory = np.zeros((n_frames, 3), dtype=np.float32)
    transforms = np.zeros((n_frames, 4, 4), dtype=np.float64)

    for i, fk in enumerate(frame_keys):
        ext_3x4 = np.array(extrinsics[fk], dtype=np.float64)
        ext_4x4 = np.eye(4, dtype=np.float64)
        ext_4x4[:3, :] = ext_3x4

        # Model input: always raw extrinsic (coord=null, what model was trained with)
        trajectory[i] = ext_4x4[:3, 3].astype(np.float32)

        # Per-frame transform: camera-space → requested output coord
        if coord == "global":
            # camera-space → world: inv(extrinsic)
            transforms[i] = np.linalg.inv(ext_4x4)
        elif coord == "aria":
            # camera-space → aria-relative: first_ext @ inv(extrinsic)
            transforms[i] = first_ext @ np.linalg.inv(ext_4x4)
        else:
            # No transform needed — predictions already in camera space
            transforms[i] = np.eye(4, dtype=np.float64)

    return trajectory, [int(x) for x in frame_keys], transforms


# ======================================================================
# Prediction using the baseline model's own test() method
# ======================================================================

@torch.no_grad()
def predict_poses(model, trajectory, device="cuda:0"):
    """
    Predict body poses using the baseline model's test() method.
    This replicates the exact inference logic from the baseline including
    its sliding window handling.

    Args:
        model: baseline ModelEgoExo4D object (has feed_data/test/current_prediction)
        trajectory: (N_frames, 3) numpy array of camera positions
        device: torch device

    Returns:
        predictions: (N_frames, 17, 3) numpy array of joint positions
    """
    traj_tensor = torch.from_numpy(trajectory).float()

    # The baseline expects (1, T, 3) for inference
    data = {
        "cond": traj_tensor.unsqueeze(0),  # (1, T, 3)
    }
    model.feed_data(data, inference=True)
    model.test(inference=True)
    result = model.current_prediction()

    # result['position'] is (T, 17, 3) tensor
    predictions = result["position"].cpu().numpy()
    if predictions.ndim == 3:
        return predictions  # (T, 17, 3)
    else:
        return predictions.reshape(-1, 17, 3)


def transform_predictions(predictions, transforms):
    """
    Transform predicted joint positions from model output coordinate system
    (camera/null) to the requested output coordinate system (global/aria).

    The model predicts in per-frame camera space (coord=null). To get
    global coordinates, each frame's joints are transformed by the
    corresponding per-frame 4x4 matrix.

    Args:
        predictions: (T, 17, 3) joint positions in camera space
        transforms:  (T, 4, 4) per-frame transforms (camera → output coord)

    Returns:
        transformed: (T, 17, 3) joint positions in output coord system
    """
    T, J, _ = predictions.shape
    assert transforms.shape[0] == T, (
        f"Transform count {transforms.shape[0]} != prediction frames {T}"
    )

    # Check if transforms are all identity (no-op, coord=None)
    identity = np.eye(4, dtype=np.float64)
    if np.allclose(transforms[0], identity):
        return predictions  # skip if no transform needed

    transformed = np.zeros_like(predictions)
    for i in range(T):
        R = transforms[i, :3, :3]  # (3, 3)
        t = transforms[i, :3, 3]   # (3,)
        # Apply: p_out = R @ p_camera + t  for each joint
        transformed[i] = (R @ predictions[i].T).T + t

    return transformed


# ======================================================================
# Output formatting (match GT annotation format)
# ======================================================================

def format_as_gt_json(predictions, frame_indices):
    """
    Format predictions in the same JSON format as GT ego_pose annotations.

    GT format:
    {
        "frame_idx_str": [
            {
                "annotation3D": {
                    "joint_name": {"x": float, "y": float, "z": float},
                    ...
                }
            }
        ],
        ...
    }
    """
    output = {}
    for i, fidx in enumerate(frame_indices):
        if i >= predictions.shape[0]:
            break

        joints_3d = {}
        for j, joint_name in enumerate(BODY_JOINTS):
            joints_3d[joint_name] = {
                "x": float(predictions[i, j, 0]),
                "y": float(predictions[i, j, 1]),
                "z": float(predictions[i, j, 2]),
            }

        output[str(fidx)] = [{"annotation3D": joints_3d}]

    return output


# ======================================================================
# Main pipeline
# ======================================================================

def process_split(
    split,
    ego_pose_root,
    output_root,
    model,
    device,
    coord,
    overwrite=False,
):
    """Process all takes in a given split."""
    camera_pose_dir = os.path.join(ego_pose_root, split, "camera_pose")
    output_dir = os.path.join(output_root, split, "body", "annotation")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.isdir(camera_pose_dir):
        print(f"  Skipping {split}: {camera_pose_dir} not found")
        return 0, 0, 0

    files = sorted([f for f in os.listdir(camera_pose_dir) if f.endswith(".json")])
    print(f"  {split}: {len(files)} takes with camera_pose data")

    n_success = 0
    n_skip = 0
    n_error = 0
    t_load = 0.0
    t_infer = 0.0
    t_save = 0.0

    for fname in tqdm(files, desc=f"  {split}"):
        take_uid = fname.replace(".json", "")
        output_path = os.path.join(output_dir, fname)

        if os.path.exists(output_path) and not overwrite:
            n_skip += 1
            continue

        try:
            # Load aria camera trajectory
            cam_pose_path = os.path.join(camera_pose_dir, fname)
            t0 = time.time()
            trajectory, frame_indices, per_frame_transforms = load_camera_trajectory(
                cam_pose_path, coord=coord
            )
            t_load += (time.time() - t0)

            if trajectory.shape[0] < 2:
                print(f"    WARNING: {take_uid} has {trajectory.shape[0]} frames, skipping")
                n_error += 1
                continue

            # Predict body poses (model always gets coord=null trajectory)
            t1 = time.time()
            predictions = predict_poses(model, trajectory, device=device)
            t_infer += (time.time() - t1)

            # Transform from camera space to requested output coord system
            predictions = transform_predictions(predictions, per_frame_transforms)

            # Sanity check
            assert predictions.shape[0] == len(frame_indices), (
                f"Prediction length mismatch: {predictions.shape[0]} vs {len(frame_indices)}"
            )

            # Format and save
            output_json = format_as_gt_json(predictions, frame_indices)
            t2 = time.time()
            dump_json_file(output_path, output_json)
            t_save += (time.time() - t2)

            n_success += 1

        except Exception as e:
            print(f"    ERROR processing {take_uid}: {e}")
            n_error += 1
            continue

    if n_success > 0:
        t_total = t_load + t_infer + t_save
        if t_total > 0:
            print(
                f"  {split} timing: load={t_load:.1f}s ({100*t_load/t_total:.1f}%), "
                f"infer={t_infer:.1f}s ({100*t_infer/t_total:.1f}%), "
                f"save={t_save:.1f}s ({100*t_save/t_total:.1f}%)"
            )

    return n_success, n_skip, n_error


def gpu_worker(gpu_id, worker_idx, take_assignments, args, coord):
    """
    Worker function for multi-GPU processing.
    Each worker loads its own model on the assigned GPU and processes its chunk.
    """
    device = f"cuda:{gpu_id}"
    worker_tag = f"GPU {gpu_id} / W{worker_idx}"
    print(f"\n[{worker_tag}] Starting worker with {len(take_assignments)} takes")

    model, opt, config_window_size, _ = load_model_from_repo(
        args.baseline_repo, args.checkpoint, device=device
    )
    print(f"[{worker_tag}] Model loaded, coord={coord}")

    n_success = 0
    n_error = 0
    n_skip = 0
    t_load = 0.0
    t_infer = 0.0
    t_save = 0.0
    t0 = time.time()

    for idx, (split, fname) in enumerate(take_assignments):
        take_uid = fname.replace(".json", "")
        output_dir = os.path.join(args.output_root, split, "body", "annotation")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, fname)

        if os.path.exists(output_path) and not args.overwrite:
            n_skip += 1
            continue

        try:
            cam_pose_path = os.path.join(
                args.ego_pose_root, split, "camera_pose", fname
            )
            t_load0 = time.time()
            trajectory, frame_indices, per_frame_transforms = load_camera_trajectory(
                cam_pose_path, coord=coord
            )
            t_load += (time.time() - t_load0)

            if trajectory.shape[0] < 2:
                n_error += 1
                continue

            t_infer0 = time.time()
            predictions = predict_poses(model, trajectory, device=device)
            t_infer += (time.time() - t_infer0)

            # Transform from camera space to requested output coord system
            predictions = transform_predictions(predictions, per_frame_transforms)

            assert predictions.shape[0] == len(frame_indices), (
                f"Prediction length mismatch: {predictions.shape[0]} vs {len(frame_indices)}"
            )

            output_json = format_as_gt_json(predictions, frame_indices)
            t_save0 = time.time()
            dump_json_file(output_path, output_json)
            t_save += (time.time() - t_save0)

            n_success += 1

            # Log progress periodically
            if (n_success % 50 == 0) or (idx < 3):
                elapsed = time.time() - t0
                rate = (n_success + n_skip + n_error) / elapsed
                print(f"  [{worker_tag}] {idx+1}/{len(take_assignments)} "
                      f"ok={n_success} err={n_error} skip={n_skip} "
                      f"({rate:.1f} takes/s)")

        except Exception as e:
            print(f"  [{worker_tag}] ERROR {take_uid}: {e}")
            n_error += 1

    elapsed = time.time() - t0
    print(f"\n[{worker_tag}] DONE: {n_success} ok, {n_error} errors, "
          f"{n_skip} skipped in {elapsed:.1f}s")
    t_total = t_load + t_infer + t_save
    if t_total > 0:
        print(
            f"[{worker_tag}] timing: load={t_load:.1f}s ({100*t_load/t_total:.1f}%), "
            f"infer={t_infer:.1f}s ({100*t_infer/t_total:.1f}%), "
            f"save={t_save:.1f}s ({100*t_save/t_total:.1f}%)"
        )
    return n_success, n_skip, n_error


def main():
    parser = argparse.ArgumentParser(
        description="Predict ego body pose using EgoExo4D baseline model"
    )
    parser.add_argument(
        "--ego_pose_root",
        type=str,
        default="/path/to/EgoExo4D/annotations/ego_pose",
        help="Root of ego_pose annotations (contains train/val/test with camera_pose/)",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="/path/to/EgoExo4D/predicted_ego_pose",
        help="Output directory (same structure as ego_pose_root)",
    )
    parser.add_argument(
        "--baseline_repo",
        type=str,
        default="/path/to/ego-exo4d-egopose/bodypose",
        help="Path to cloned ego-exo4d-egopose/bodypose repo (REQUIRED)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/path/to/ego-exo4d-egopose/bodypose/results/EgoExo4D/models/100000_G.pth",
        help="Path to pretrained model checkpoint (100000_G.pth)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="Splits to process (default: train val; test has no GT to compare)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for inference (single-GPU mode, ignored if --gpus is set)",
    )
    parser.add_argument(
        "--gpus",
        nargs="+",
        type=int,
        default=None,
        help="List of GPU IDs for parallel processing (e.g., --gpus 0 2 5 7). "
             "Overrides --device. If not set, uses single-GPU mode with --device.",
    )
    parser.add_argument(
        "--workers_per_gpu",
        type=int,
        default=1,
        help="Number of worker processes per GPU. "
             "Use >1 to overlap CPU JSON I/O and GPU inference on each GPU.",
    )
    parser.add_argument(
        "--coord",
        type=str,
        default=None,
        help="Coordinate system override (default: use baseline config value). "
             "Options: null/camera (raw extrinsics), aria (relative to first frame), "
             "global (world frame)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing predictions",
    )
    args = parser.parse_args()

    # Validate baseline repo
    if not os.path.isdir(args.baseline_repo):
        print(f"ERROR: Baseline repo not found at: {args.baseline_repo}")
        print("\nSetup instructions:")
        print("  git clone https://github.com/EGO4D/ego-exo4d-egopose.git")
        print("  cd ego-exo4d-egopose/bodypose")
        print("  conda env create -f environment.yml")
        print("  gdown 1XpY7aa7I7XFNDM6tJPcyS17xPsDlW0g7")
        print("  mkdir -p results/EgoExo4D/models/")
        print("  mv 100000_G.pth results/EgoExo4D/models/")
        sys.exit(1)

    if not os.path.isfile(args.checkpoint):
        print(f"ERROR: Checkpoint not found at: {args.checkpoint}")
        print("\nDownload with:")
        print("  pip install gdown")
        print("  gdown 1XpY7aa7I7XFNDM6tJPcyS17xPsDlW0g7")
        sys.exit(1)

    print("=" * 60)
    print("EgoExo4D Ego Body Pose Prediction")
    print("=" * 60)
    print(f"  ego_pose_root:  {args.ego_pose_root}")
    print(f"  output_root:    {args.output_root}")
    print(f"  baseline_repo:  {args.baseline_repo}")
    print(f"  checkpoint:     {args.checkpoint}")
    print(f"  splits:         {args.splits}")
    print(f"  device:         {args.device}")
    print(f"  gpus:           {args.gpus}")
    print(f"  workers/gpu:    {args.workers_per_gpu}")
    print(f"  coord override: {args.coord}")
    print(f"  overwrite:      {args.overwrite}")
    print(f"  json engine:    {'orjson' if orjson is not None else 'json'}")
    print()

    # Gather all (split, filename) pairs to process
    all_tasks = []
    for split in args.splits:
        camera_pose_dir = os.path.join(args.ego_pose_root, split, "camera_pose")
        if not os.path.isdir(camera_pose_dir):
            print(f"  Skipping {split}: {camera_pose_dir} not found")
            continue
        files = sorted([f for f in os.listdir(camera_pose_dir) if f.endswith(".json")])
        print(f"  {split}: {len(files)} takes with camera_pose data")
        for fname in files:
            all_tasks.append((split, fname))

    print(f"\n  Total tasks: {len(all_tasks)}")

    if len(all_tasks) == 0:
        print("  No tasks to process!")
        return

    # Determine coord system: need to load config to get default if not overridden
    # For single-GPU we load model first; for multi-GPU we need coord before fork
    if args.coord is not None:
        coord = args.coord
        print(f"  Using coord system (override): {coord}")
    else:
        # Peek at the baseline config to get default coord
        opt_path = os.path.join(args.baseline_repo, "options", "inference_egoexo.json")
        opt = load_json_with_comments(opt_path)
        coord = opt.get("datasets", {}).get("test", {}).get("coord", None)
        print(f"  Using coord system (from config): {coord}")

    if args.workers_per_gpu < 1:
        print("ERROR: --workers_per_gpu must be >= 1")
        sys.exit(1)

    gpu_ids = args.gpus  # None if single-GPU mode

    # Allow multiprocessing on a single GPU by deriving it from --device.
    if gpu_ids is None and args.workers_per_gpu > 1:
        if not args.device.startswith("cuda"):
            print("ERROR: --workers_per_gpu > 1 requires a CUDA device.")
            sys.exit(1)
        gpu_idx = int(args.device.split(":")[-1]) if ":" in args.device else 0
        gpu_ids = [gpu_idx]

    use_mp = gpu_ids is not None and (len(gpu_ids) > 1 or args.workers_per_gpu > 1)

    if use_mp:
        # Build worker slots: workers_per_gpu processes for each GPU id.
        worker_slots = []
        for gid in gpu_ids:
            for w in range(args.workers_per_gpu):
                worker_slots.append((gid, w))

        print(
            f"\n  Distributing {len(all_tasks)} takes across "
            f"{len(worker_slots)} workers on GPUs {gpu_ids}"
        )

        # Round-robin assignment by worker slot (works even with repeated GPU ids).
        chunks = [[] for _ in worker_slots]
        for i, task in enumerate(all_tasks):
            chunks[i % len(worker_slots)].append(task)

        for i, (gid, w) in enumerate(worker_slots):
            print(f"    Worker GPU {gid} / W{w}: {len(chunks[i])} takes")

        processes = []
        for i, (gid, w) in enumerate(worker_slots):
            if len(chunks[i]) == 0:
                continue
            p = mp.Process(
                target=gpu_worker,
                args=(gid, w, chunks[i], args, coord),
            )
            p.start()
            processes.append(p)

        failed = 0
        for p in processes:
            p.join()
            if p.exitcode != 0:
                failed += 1

        print(f"\n{'=' * 60}")
        if failed == 0:
            print("DONE (multiprocessing). Check per-worker logs above for timing/results.")
        else:
            print(f"FAILED: {failed} worker process(es) exited with non-zero status.")
            sys.exit(1)

    else:
        # Single-GPU mode: use --gpus [X] if given (single element), else --device
        device = f"cuda:{gpu_ids[0]}" if (gpu_ids and len(gpu_ids) == 1) else args.device
        print(f"\nLoading model from baseline repo on {device}...")
        model, opt, config_window_size, config_coord = load_model_from_repo(
            args.baseline_repo, args.checkpoint, device=device
        )
        print(f"  Using coord system: {coord}")
        print()

        # ── Process each split ──
        total_success = 0
        total_skip = 0
        total_error = 0

        for split in args.splits:
            print(f"{'─' * 40}")
            n_success, n_skip, n_error = process_split(
                split=split,
                ego_pose_root=args.ego_pose_root,
                output_root=args.output_root,
                model=model,
                device=device,
                coord=coord,
                overwrite=args.overwrite,
            )
            total_success += n_success
            total_skip += n_skip
            total_error += n_error
            print(f"  → {n_success} predicted, {n_skip} skipped, {n_error} errors")

        print(f"\n{'=' * 60}")
        print(f"DONE: {total_success} predicted, {total_skip} skipped, "
              f"{total_error} errors")

    print(f"Output: {args.output_root}")
    print(f"\nTo use predicted pose with your model, set in config:")
    print(f"  pose_root: '{args.output_root}'")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

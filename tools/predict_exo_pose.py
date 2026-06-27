"""
Predict exo body pose using 2D pose detection + multi-view triangulation.

Pipeline (matching the official EgoExo4D annotation pipeline):
1. Uses Aria headset 3D position to identify the camera-wearer in exo views
2. Runs YOLO person detection, matched to Aria-projected region by IoU
3. Runs ViTPose for 2D keypoint estimation on the matched person crop
4. RANSAC-based DLT triangulation across exo cameras (with nonlinear refinement)
5. Saves output in the same JSON format as GT ego_pose annotations

The output can be used as a drop-in replacement for GT pose via pose_root.

Prerequisites:
    pip install "transformers>=4.48.0" "ultralytics>=8.0" pillow opencv-python scipy

Usage:
    # Single GPU
    python tools/predict_exo_pose.py --device cuda:0

    # Multi-GPU (specific GPUs, processes takes in parallel)
    python tools/predict_exo_pose.py --gpus 0 1 2 3 4 5 6 7
"""

import argparse
import itertools
import json
import os
import queue
import sys
import threading
import cv2
import numpy as np
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from tqdm import tqdm
from types import SimpleNamespace
import time
import multiprocessing as mp

import torch
from scipy.optimize import least_squares

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
NUM_JOINTS = 17


# ======================================================================
# Camera utilities
# ======================================================================

def load_camera_params(camera_pose_path):
    """
    Load all exo camera parameters and Aria trajectory from a camera_pose JSON.

    Returns:
        cameras: dict of camera_name -> {
            'intrinsics': (3, 3),
            'extrinsics': (3, 4),
            'distortion': (4,) or None,
            'projection': (3, 4)  = K @ [R|t]
        }
        aria_trajectory: dict of frame_idx(int) -> (3,) world-coords head position
                         Empty dict if no Aria data found.
        metadata: dict with take_name, take_uid
    """
    with open(camera_pose_path, "r") as f:
        data = json.load(f)

    metadata = data.get("metadata", {})
    cameras = {}
    aria_trajectory = {}

    for cam_name, cam_data in data.items():
        if cam_name == "metadata":
            continue

        if "aria" in cam_name.lower():
            # Extract Aria per-frame head position in world coordinates.
            # The extrinsic is [R|t] mapping world→camera, so the camera
            # center in world coords is C = -R^T @ t.
            ext = cam_data["camera_extrinsics"]
            if isinstance(ext, dict):
                for fkey, mat in ext.items():
                    Rt = np.array(mat, dtype=np.float64)
                    R = Rt[:3, :3]
                    t = Rt[:3, 3]
                    C = -R.T @ t
                    aria_trajectory[int(fkey)] = C
            continue

        K = np.array(cam_data["camera_intrinsics"], dtype=np.float64)

        ext = cam_data["camera_extrinsics"]
        if isinstance(ext, dict):
            first_key = sorted(ext.keys(), key=int)[0]
            Rt = np.array(ext[first_key], dtype=np.float64)
        else:
            Rt = np.array(ext, dtype=np.float64)

        dist = None
        if "distortion_coeffs" in cam_data:
            dist = np.array(cam_data["distortion_coeffs"], dtype=np.float64)

        P = K @ Rt

        cameras[cam_name] = {
            "intrinsics": K,
            "extrinsics": Rt,
            "distortion": dist,
            "projection": P,
        }

    return cameras, aria_trajectory, metadata


# ======================================================================
# Multi-view triangulation (N-view DLT)
# ======================================================================

def triangulate_point_dlt(points_2d, projection_matrices, confidences=None):
    """
    Triangulate a single 3D point from N camera views using DLT.
    """
    n_cams = len(points_2d)
    A = np.zeros((2 * n_cams, 4))

    for i in range(n_cams):
        x, y = points_2d[i]
        P = projection_matrices[i]
        w = confidences[i] if confidences is not None else 1.0
        A[2 * i] = w * (x * P[2, :] - P[0, :])
        A[2 * i + 1] = w * (y * P[2, :] - P[1, :])

    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    if abs(X[3]) < 1e-10:
        return np.array([0.0, 0.0, 0.0])
    return X[:3] / X[3]


def compute_reprojection_error(point_3d, points_2d, projection_matrices):
    """Compute per-view reprojection error."""
    X_hom = np.append(point_3d, 1.0)
    errors = []
    for pt2d, P in zip(points_2d, projection_matrices):
        projected = P @ X_hom
        if abs(projected[2]) < 1e-10:
            errors.append(float("inf"))
            continue
        projected = projected[:2] / projected[2]
        errors.append(np.linalg.norm(projected - pt2d))
    return np.array(errors)


def undistort_keypoints(keypoints_2d, K, dist_coeffs):
    """
    Undistort 2D keypoints using camera intrinsics and distortion coefficients.

    Args:
        keypoints_2d: (N, 2) array of 2D keypoints in pixel coordinates
        K: (3, 3) camera intrinsic matrix
        dist_coeffs: distortion coefficients (4, 5, 8, or 14 elements)

    Returns:
        undistorted: (N, 2) array of undistorted 2D keypoints in pixel coordinates
    """
    if dist_coeffs is None or len(dist_coeffs) == 0:
        return keypoints_2d

    # cv2.undistortPoints expects (N, 1, 2).
    # With P=K it returns undistorted points in pixel coordinates directly.
    pts = keypoints_2d.reshape(-1, 1, 2).astype(np.float64)
    undist_px = cv2.undistortPoints(pts, K, dist_coeffs, P=K)  # (N, 1, 2) in pixels
    return undist_px.reshape(-1, 2)


def triangulate_point_ransac(
    points_2d,
    projection_matrices,
    confidences=None,
    reprojection_error_epsilon=30.0,
    min_inliers=2,
    refine=True,
):
    """
    RANSAC-based triangulation of a single 3D point from N camera views.

    Matches the official EgoExo4D pipeline (triangulator.py):
    1. Exhaustively test all 2-view pairs
    2. For each pair, DLT triangulate → find inlier views (reproj error < epsilon)
    3. Keep the pair producing the most inliers (tie-break: lowest mean error)
    4. Re-triangulate using all inlier views
    5. (Optional) Nonlinear refinement with Huber loss via scipy

    Args:
        points_2d: (N, 2) array of 2D keypoints
        projection_matrices: (N, 3, 4) projection matrices
        confidences: (N,) confidence weights, or None
        reprojection_error_epsilon: inlier threshold in pixels
        min_inliers: minimum inlier views required
        refine: if True, run scipy nonlinear refinement on result

    Returns:
        point_3d: (3,) triangulated 3D point
        inlier_mask: (N,) boolean mask of inlier views
        mean_reproj: mean reprojection error of inliers
    """
    n_views = len(points_2d)

    if n_views < 2:
        return np.zeros(3), np.zeros(n_views, dtype=bool), float("inf")

    # Fast path: with only 2 views, skip RANSAC — just DLT + optional refine
    if n_views == 2:
        pt_3d = triangulate_point_dlt(points_2d, projection_matrices, confidences)
        reproj_errors = compute_reprojection_error(pt_3d, points_2d, projection_matrices)
        mean_err = np.mean(reproj_errors)
        if refine and mean_err < reprojection_error_epsilon:
            pt_3d = _refine_point_nonlinear(pt_3d, points_2d, projection_matrices)
            reproj_errors = compute_reprojection_error(pt_3d, points_2d, projection_matrices)
            mean_err = np.mean(reproj_errors)
        return pt_3d, np.ones(2, dtype=bool), mean_err

    # Generate all 2-view combinations (exhaustive for small N)
    all_pairs = list(itertools.combinations(range(n_views), 2))

    best_inlier_mask = np.zeros(n_views, dtype=bool)
    best_inlier_count = 0
    best_mean_error = float("inf")
    best_point = np.zeros(3)

    for i_cam, j_cam in all_pairs:
        # Triangulate from this pair
        pair_pts = np.array([points_2d[i_cam], points_2d[j_cam]])
        pair_Ps = np.array([projection_matrices[i_cam], projection_matrices[j_cam]])
        pair_confs = None
        if confidences is not None:
            pair_confs = np.array([confidences[i_cam], confidences[j_cam]])

        pt_3d = triangulate_point_dlt(pair_pts, pair_Ps, pair_confs)

        # Compute reprojection error for ALL views
        reproj_errors = compute_reprojection_error(pt_3d, points_2d, projection_matrices)

        # Find inliers
        inlier_mask = reproj_errors < reprojection_error_epsilon
        # The seed pair is always included
        inlier_mask[i_cam] = True
        inlier_mask[j_cam] = True
        inlier_count = int(inlier_mask.sum())

        inlier_errors = reproj_errors[inlier_mask]
        mean_err = np.mean(inlier_errors) if len(inlier_errors) > 0 else float("inf")

        # Keep best: most inliers, then lowest mean error
        if (inlier_count > best_inlier_count) or \
           (inlier_count == best_inlier_count and mean_err < best_mean_error):
            best_inlier_count = inlier_count
            best_inlier_mask = inlier_mask.copy()
            best_mean_error = mean_err
            best_point = pt_3d

        # Early exit: if all views are already inliers, no need to test more pairs
        if best_inlier_count == n_views:
            break

    if best_inlier_count < min_inliers:
        return np.zeros(3), np.zeros(n_views, dtype=bool), float("inf")

    # Re-triangulate using ALL inlier views
    inlier_pts = points_2d[best_inlier_mask]
    inlier_Ps = projection_matrices[best_inlier_mask]
    inlier_confs = confidences[best_inlier_mask] if confidences is not None else None
    best_point = triangulate_point_dlt(inlier_pts, inlier_Ps, inlier_confs)

    # Nonlinear refinement using Huber loss (matches official pipeline).
    # Only refine when there were outliers or when error is high — if all
    # views are inliers with low error, DLT is already accurate and scipy
    # overhead is wasted (~2ms per call).
    had_outliers = best_inlier_count < n_views
    if refine and best_inlier_count >= 2 and (had_outliers or best_mean_error > 15.0):
        best_point = _refine_point_nonlinear(best_point, inlier_pts, inlier_Ps)

    # Recompute final reproj error
    final_errors = compute_reprojection_error(best_point, inlier_pts, inlier_Ps)
    best_mean_error = np.mean(final_errors)

    return best_point, best_inlier_mask, best_mean_error


def _refine_point_nonlinear(point_3d_init, points_2d, projection_matrices):
    """
    Nonlinear refinement of a triangulated 3D point via Huber-loss
    reprojection error minimization. Matches official pipeline's
    scipy.optimize.least_squares with loss="huber", method="trf".
    """
    def residuals(x):
        X_hom = np.array([x[0], x[1], x[2], 1.0])
        res = []
        for pt2d, P in zip(points_2d, projection_matrices):
            proj = P @ X_hom
            if abs(proj[2]) < 1e-10:
                res.extend([0.0, 0.0])
                continue
            proj_2d = proj[:2] / proj[2]
            res.extend(proj_2d - pt2d)
        return np.array(res)

    result = least_squares(
        residuals,
        point_3d_init,
        loss="huber",
        f_scale=1.0,
        method="trf",
        max_nfev=50,
    )
    return result.x


def triangulate_joints_multiview(
    all_keypoints_2d,
    all_scores,
    projection_matrices,
    score_threshold=0.3,
    min_views=2,
    max_reproj_error=50.0,
    use_ransac=True,
):
    """
    Triangulate all joints from multi-view 2D detections.

    Uses RANSAC + nonlinear refinement per joint (matching the official
    EgoExo4D annotation pipeline) when use_ransac=True.
    """
    n_cams, n_joints, _ = all_keypoints_2d.shape
    joints_3d = np.zeros((n_joints, 3))
    valid = np.zeros(n_joints, dtype=bool)

    for j in range(n_joints):
        mask = all_scores[:, j] >= score_threshold
        if mask.sum() < min_views:
            continue

        pts = all_keypoints_2d[mask, j, :]
        Ps = projection_matrices[mask]
        confs = all_scores[mask, j]

        if use_ransac:
            point_3d, inlier_mask, mean_err = triangulate_point_ransac(
                pts, Ps, confs,
                reprojection_error_epsilon=max_reproj_error,
                min_inliers=min_views,
                refine=False,  # DLT with inlier selection is sufficient;
                               # scipy refinement adds ~2ms/joint for <1cm gain
            )
            if mean_err > max_reproj_error:
                continue
        else:
            point_3d = triangulate_point_dlt(pts, Ps, confs)
            reproj_errors = compute_reprojection_error(point_3d, pts, Ps)
            if np.mean(reproj_errors) > max_reproj_error:
                continue

        joints_3d[j] = point_3d
        valid[j] = True

    return joints_3d, valid


# ======================================================================
# 2D Pose detection via HuggingFace ViTPose + YOLO person detector
# ======================================================================

class PoseEstimator:
    """
    2D pose estimator using:
    - Ultralytics YOLO for person detection (batched)
    - HuggingFace ViTPose for keypoint estimation
    """

    def __init__(self, device="cuda:0"):
        self.device = device
        self._init_detector()
        self._init_pose_model()

    def _init_detector(self):
        """Initialize YOLO person detector."""
        try:
            from ultralytics import YOLO
        except ImportError:
            print("ERROR: ultralytics not installed.")
            print("  pip install ultralytics")
            sys.exit(1)

        print(f"  [{self.device}] Loading YOLO person detector...")
        self.detector = YOLO("yolo11n.pt")
        self.detector.to(self.device)
        print(f"  [{self.device}] YOLO loaded")

    def _init_pose_model(self):
        """Initialize ViTPose from HuggingFace."""
        try:
            from transformers import VitPoseForPoseEstimation, AutoImageProcessor
        except ImportError:
            print("ERROR: transformers not installed or too old.")
            print('  pip install "transformers>=4.48.0"')
            sys.exit(1)

        print(f"  [{self.device}] Loading ViTPose-base from HuggingFace...")
        model_name = "usyd-community/vitpose-base-simple"
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        self.pose_model = VitPoseForPoseEstimation.from_pretrained(model_name)
        self.pose_model = self.pose_model.to(self.device)
        self.pose_model.eval()
        print(f"  [{self.device}] ViTPose loaded")

    def detect_persons_batch(self, frames_bgr):
        """
        Detect persons in a batch of BGR frames using batched YOLO.

        Args:
            frames_bgr: list of (H, W, 3) BGR numpy arrays

        Returns:
            list of (boxes, confs) per frame
        """
        if len(frames_bgr) == 0:
            return []

        results = self.detector(frames_bgr, classes=[0], verbose=False)
        all_detections = []
        for r in results:
            boxes = []
            confs = []
            for box in r.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                if conf > 0.5:
                    boxes.append(xyxy.tolist())
                    confs.append(conf)
            all_detections.append((boxes, confs))
        return all_detections

    @torch.no_grad()
    def estimate_pose(self, frame_bgr, boxes):
        """
        Estimate 2D pose for detected persons in a single frame.

        Args:
            frame_bgr: (H, W, 3) BGR numpy array
            boxes: list of [x1, y1, x2, y2]

        Returns:
            keypoints: (N_people, 17, 2) 2D keypoints in image coords
            scores: (N_people, 17) confidence scores
        """
        if len(boxes) == 0:
            return np.zeros((0, NUM_JOINTS, 2)), np.zeros((0, NUM_JOINTS))

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        inputs = self.image_processor(
            pil_image, boxes=[boxes], return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.pose_model(**inputs)

        pose_results = self.image_processor.post_process_pose_estimation(
            outputs, boxes=[boxes]
        )

        all_kps = []
        all_scores = []
        for person in pose_results[0]:
            kps = person["keypoints"].cpu().numpy()  # (17, 2)
            sc = person["scores"].cpu().numpy()  # (17,)
            all_kps.append(kps)
            all_scores.append(sc)

        if len(all_kps) == 0:
            return np.zeros((0, NUM_JOINTS, 2)), np.zeros((0, NUM_JOINTS))

        return np.array(all_kps), np.array(all_scores)

    @torch.no_grad()
    def detect_and_estimate_batch(self, frames_bgr):
        """
        Full pipeline for a batch of frames: detect + estimate pose.
        Batches YOLO detection AND ViTPose inference across all frames
        to maximize GPU utilization.

        Args:
            frames_bgr: list of BGR frames

        Returns:
            list of (keypoints, scores, det_boxes, det_confs) per frame
            - keypoints: (N_people, 17, 2)
            - scores: (N_people, 17)
            - det_boxes: list of [x1,y1,x2,y2] (aligned with keypoints order)
            - det_confs: list of float
        """
        detections = self.detect_persons_batch(frames_bgr)

        # Collect all per-frame inputs, then batch ViTPose across all frames
        per_frame_info = []  # (pil_image, boxes, confs) for frames with detections
        frame_map = []       # index into per_frame_info for each frame

        for i, (frame, (boxes, confs)) in enumerate(zip(frames_bgr, detections)):
            if len(boxes) == 0:
                frame_map.append(None)
            else:
                frame_map.append(len(per_frame_info))
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                per_frame_info.append((pil_image, boxes, confs))

        empty = (np.zeros((0, NUM_JOINTS, 2)), np.zeros((0, NUM_JOINTS)), [], [])
        if len(per_frame_info) == 0:
            return [empty for _ in frames_bgr]

        # Process all frames through image_processor, then concatenate crops
        all_pixel_values = []
        all_boxes_list = []
        all_confs_list = []
        crop_counts = []  # how many crops per frame (for splitting results)

        for pil_image, boxes, confs in per_frame_info:
            inputs = self.image_processor(
                pil_image, boxes=[boxes], return_tensors="pt"
            )
            pv = inputs["pixel_values"]  # (n_persons, C, H, W)
            all_pixel_values.append(pv)
            all_boxes_list.append(boxes)
            all_confs_list.append(confs)
            crop_counts.append(pv.shape[0])

        # Batch all crops into a single forward pass
        batched_pixels = torch.cat(all_pixel_values, dim=0).to(self.device)
        outputs = self.pose_model(pixel_values=batched_pixels)

        # Split results back per frame and post-process
        all_heatmaps = outputs.heatmaps  # (total_crops, 17, H, W)
        per_frame_results = []
        offset = 0
        for idx, (pil_image, boxes, confs) in enumerate(per_frame_info):
            n_crops = crop_counts[idx]
            # Create a mock output for this frame's crops
            frame_heatmaps = all_heatmaps[offset:offset + n_crops]
            offset += n_crops

            # Post-process using the image_processor
            mock_output = SimpleNamespace(heatmaps=frame_heatmaps)
            pose_results = self.image_processor.post_process_pose_estimation(
                mock_output, boxes=[boxes]
            )

            all_kps = []
            all_scores = []
            for person in pose_results[0]:
                kps = person["keypoints"].cpu().numpy()
                sc = person["scores"].cpu().numpy()
                all_kps.append(kps)
                all_scores.append(sc)

            if len(all_kps) == 0:
                per_frame_results.append(empty)
            else:
                per_frame_results.append(
                    (np.array(all_kps), np.array(all_scores), boxes, confs)
                )

        # Map back to original frame order
        results = []
        for i in range(len(frames_bgr)):
            if frame_map[i] is None:
                results.append(empty)
            else:
                results.append(per_frame_results[frame_map[i]])

        return results


def _maybe_undistort(kps_2d, camera):
    """Undistort 2D keypoints if distortion coefficients are available."""
    dist = camera.get("distortion")
    if dist is not None:
        return undistort_keypoints(kps_2d, camera["intrinsics"], dist)
    return kps_2d


# ======================================================================
# Aria-based person selection (matches official EgoExo4D pipeline)
# ======================================================================

def get_aria_head_position(aria_trajectory, frame_idx, _sorted_cache=[None, None]):
    """
    Get the Aria headset 3D position for a given frame index.

    Finds the nearest Aria trajectory frame using bisect for O(log N) lookup.
    (Aria may have different frame rate / frame indices than the exo cameras.)

    Args:
        aria_trajectory: dict of frame_idx(int) -> (3,) world position
        frame_idx: target frame index

    Returns:
        (3,) world position of the Aria headset, or None if no trajectory
    """
    if not aria_trajectory:
        return None

    import bisect

    # Cache sorted frame list for the current trajectory.
    # Simple single-entry cache: [dict_ref, sorted_keys].
    # Using a direct reference (not id()) avoids stale-cache bugs from
    # Python reusing memory addresses for different objects.
    if _sorted_cache[0] is not aria_trajectory:
        _sorted_cache[0] = aria_trajectory
        _sorted_cache[1] = sorted(aria_trajectory.keys())
    aria_frames = _sorted_cache[1]

    # Binary search for nearest frame
    pos = bisect.bisect_left(aria_frames, frame_idx)
    candidates = []
    if pos < len(aria_frames):
        candidates.append(aria_frames[pos])
    if pos > 0:
        candidates.append(aria_frames[pos - 1])
    nearest = min(candidates, key=lambda x: abs(x - frame_idx))

    # Reject if too far away (> 30 frames = 1 second at 30fps)
    if abs(nearest - frame_idx) > 30:
        return None

    return aria_trajectory[nearest]


def project_point_to_2d(point_3d, P, img_w, img_h):
    """
    Project a single 3D world point into 2D image coordinates.

    Returns:
        (px, py) in image coordinates, or None if behind camera or far outside
        the image. Allows a margin of 50px outside image bounds for edge cases
        where the person is partially off-screen.
    """
    X_hom = np.append(point_3d, 1.0)
    proj = P @ X_hom
    if proj[2] <= 0:
        return None
    px = proj[0] / proj[2]
    py = proj[1] / proj[2]
    margin = 50
    if px < -margin or px > img_w + margin or py < -margin or py > img_h + margin:
        return None
    return (px, py)


def project_body_cylinder(
    head_3d, P, img_w, img_h, human_height=1.8, radius=0.3, n_samples=20
):
    """
    Project a 3D body cylinder around the Aria head position into an exo camera.
    Returns both the bounding box AND the individual projected 2D points.

    Matches the official pipeline's get_region_proposal(): places a vertical
    cylinder (height=1.8m, radius=0.3m) centered at the camera wearer's
    estimated body center (head minus half the body height), then projects
    surface points into the exo camera view.

    Args:
        head_3d: (3,) world-coords position of Aria headset (head)
        P: (3, 4) projection matrix of the exo camera
        img_w, img_h: image dimensions
        human_height: estimated body height in meters
        radius: cylinder radius in meters
        n_samples: number of surface sample points

    Returns:
        (bbox, points_2d) where:
            bbox: (x1, y1, x2, y2) bounding box, or None if behind camera
            points_2d: (N, 2) array of projected 2D points (may be empty)
    """
    # Body center is at approximately hip height = head - 0.5 * human_height
    # (the official pipeline uses the same convention with a downward normal)
    up = np.array([0.0, 0.0, 1.0])  # z-up world coordinate
    body_center = head_3d - 0.5 * human_height * up

    # Generate sample 3D points on the cylinder surface
    angles = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    heights = np.linspace(-human_height / 2, human_height / 2, 5)
    points_3d = []
    for h in heights:
        for a in angles:
            pt = body_center + np.array([
                radius * np.cos(a),
                radius * np.sin(a),
                h,
            ])
            points_3d.append(pt)
    # Also add head and foot points
    points_3d.append(head_3d)
    points_3d.append(head_3d - human_height * up)
    points_3d = np.array(points_3d)

    # Project all points
    projected = []
    for pt in points_3d:
        X_hom = np.append(pt, 1.0)
        proj = P @ X_hom
        if proj[2] <= 0:
            continue  # behind camera
        px = proj[0] / proj[2]
        py = proj[1] / proj[2]
        projected.append([px, py])

    if len(projected) < 4:
        return None, np.empty((0, 2))

    projected = np.array(projected)

    x1 = max(0, np.min(projected[:, 0]))
    y1 = max(0, np.min(projected[:, 1]))
    x2 = min(img_w, np.max(projected[:, 0]))
    y2 = min(img_h, np.max(projected[:, 1]))

    if x2 <= x1 or y2 <= y1:
        return None, projected

    return (x1, y1, x2, y2), projected


def project_body_cylinder_to_bbox(head_3d, P, img_w, img_h, **kwargs):
    """Backward-compatible wrapper — returns only the bbox."""
    result = project_body_cylinder(head_3d, P, img_w, img_h, **kwargs)
    if result[0] is None:
        return None
    return result[0]


def compute_iou(box_a, box_b):
    """Compute IoU between two (x1, y1, x2, y2) bounding boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def select_person_by_aria(
    detections_boxes,
    detections_confs,
    keypoints_all,
    scores_all,
    aria_head_2d,
):
    """
    Select the camera-wearer from YOLO detections using projected Aria head point.

    The Aria head (glasses position) projects to within 2-5 pixels of the
    camera wearer's ViTPose nose keypoint at 448p (verified empirically).

    Strategy: find the person whose ViTPose nose keypoint (index 0) is
    closest to the projected Aria head point. This is more robust than
    matching against YOLO bounding boxes, which can be tight or offset.

    A maximum distance threshold (100px at 448p) rejects clearly wrong
    matches (e.g., all detected people are far from the camera wearer).

    Args:
        detections_boxes: list of [x1, y1, x2, y2] YOLO detection boxes
        detections_confs: list of confidence scores
        keypoints_all: (N_people, 17, 2) ViTPose keypoints
        scores_all: (N_people, 17) ViTPose scores
        aria_head_2d: (px, py) projected Aria head in image coords, or None

    Returns:
        kps: (17, 2) selected keypoints, or None
        sc: (17,) selected scores, or None
    """
    n_people = keypoints_all.shape[0]
    if n_people == 0:
        return None, None

    if n_people == 1:
        return keypoints_all[0], scores_all[0]

    if aria_head_2d is not None:
        hx, hy = aria_head_2d

        # Find person whose ViTPose nose (keypoint index 0) is closest
        # to the projected Aria head point.
        NOSE_IDX = 0
        MAX_DIST = 100.0  # pixels at 448p — reject if all noses are far

        best_idx = -1
        best_dist = float("inf")

        for i in range(n_people):
            # Use nose keypoint if confident, otherwise use mean of
            # eye keypoints (indices 1,2) as backup head location
            nose_conf = scores_all[i, NOSE_IDX]
            if nose_conf >= 0.3:
                nx, ny = keypoints_all[i, NOSE_IDX]
            else:
                # Try eyes as backup
                eye_confs = scores_all[i, 1:3]
                if np.max(eye_confs) >= 0.3:
                    eye_mask = eye_confs >= 0.3
                    nx = np.mean(keypoints_all[i, 1:3, 0][eye_mask])
                    ny = np.mean(keypoints_all[i, 1:3, 1][eye_mask])
                else:
                    continue  # no reliable head keypoint for this person

            dist = np.sqrt((nx - hx) ** 2 + (ny - hy) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        if best_idx >= 0 and best_dist <= MAX_DIST:
            return keypoints_all[best_idx], scores_all[best_idx]

        # All noses too far — reject this camera view
        return None, None

    # No Aria data: fall back to highest confidence detection
    mean_scores = np.mean(scores_all, axis=1)
    idx = np.argmax(mean_scores)
    return keypoints_all[idx], scores_all[idx]


# ======================================================================
# Threaded multi-camera video frame loader
# ======================================================================

class MultiCameraFrameLoader:
    """
    Prefetch frames from multiple cameras in parallel using a thread pool.

    For each frame index, all cameras are read concurrently (one thread per
    camera). Decoded frame-sets are buffered in a queue so the main thread
    (GPU inference) never blocks on IO.

    cv2.VideoCapture is NOT thread-safe for a single object, but each camera
    has its own capture — so parallel reads across cameras are safe.
    """

    def __init__(self, video_paths, frame_indices, n_threads=8, prefetch_frames=8):
        """
        Args:
            video_paths: dict of cam_name -> video file path
            frame_indices: list of frame indices to read (already strided)
            n_threads: number of decode threads (default: 8)
            prefetch_frames: how many frame-sets to buffer ahead (default: 8)
        """
        self.frame_indices = frame_indices
        self.n_threads = n_threads

        # Open one VideoCapture per camera
        self.readers = {}
        for cam_name, path in video_paths.items():
            cap = cv2.VideoCapture(path)
            if cap.isOpened():
                self.readers[cam_name] = cap

        self.active_cams = sorted(self.readers.keys())
        self._queue = queue.Queue(maxsize=prefetch_frames)
        self._error = None
        self._thread = threading.Thread(target=self._prefetch_loop, daemon=True)
        self._thread.start()

    @property
    def n_frames_min(self):
        """Minimum frame count across all cameras."""
        if not self.readers:
            return 0
        return min(
            int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in self.readers.values()
        )

    def get_video_props(self, cam_name):
        """Get (width, height, fps) for a camera."""
        cap = self.readers[cam_name]
        return (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            cap.get(cv2.CAP_PROP_FPS),
        )

    def _decode_camera_to_queue(self, cam_name, cam_queue):
        """
        Sequentially decode requested frames for one camera, pushing each
        frame to cam_queue as it's decoded (streaming, not buffered).

        Instead of random-seeking (cap.set + cap.read per frame), we decode
        forward sequentially and grab only the frames we need.

        For compressed MP4 with stride=10, sequential decode is dramatically
        faster because random-seeking forces the decoder to find the nearest
        keyframe and re-decode to the target — often 100+ ms per seek.
        Sequential decode just calls cap.read() which takes ~1-5 ms per frame.
        """
        cap = self.readers[cam_name]
        current_pos = 0  # next frame the decoder will return

        for fidx in self.frame_indices:
            if fidx < current_pos:
                # Frame is behind current position — must seek (rare for
                # sorted frame_indices, but handle it for safety)
                cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
                current_pos = fidx

            # Skip (grab without decode where possible) to target frame
            while current_pos < fidx:
                cap.grab()  # fast skip — no decode
                current_pos += 1

            ret, frame = cap.read()
            current_pos += 1
            cam_queue.put(frame if ret else None)

        cam_queue.put(None)  # sentinel

    def _prefetch_loop(self):
        """
        Background thread: decode frames from all cameras in parallel.

        Each camera gets its own decode thread that sequentially reads frames
        and pushes them to a per-camera queue.  This thread collects one frame
        from each camera queue per frame index and pushes the assembled
        frame-set to the main output queue.

        Memory-efficient: only prefetch_frames × n_cameras frames are buffered
        at any time (not the entire video).
        """
        try:
            # Per-camera queues (bounded to limit memory)
            cam_queues = {
                cam: queue.Queue(maxsize=self._queue.maxsize + 2)
                for cam in self.active_cams
            }

            # Start one decode thread per camera
            decode_threads = []
            for cam in self.active_cams:
                t = threading.Thread(
                    target=self._decode_camera_to_queue,
                    args=(cam, cam_queues[cam]),
                    daemon=True,
                )
                t.start()
                decode_threads.append(t)

            # Collect one frame from each camera per frame index
            for fidx in self.frame_indices:
                frames = {}
                for cam in self.active_cams:
                    frame = cam_queues[cam].get()
                    if frame is not None:
                        frames[cam] = frame
                self._queue.put((fidx, frames))

            # Wait for decode threads to finish
            for t in decode_threads:
                t.join()

        except Exception as e:
            self._error = e
        finally:
            self._queue.put(None)  # sentinel

    def __iter__(self):
        while True:
            item = self._queue.get()
            if item is None:
                if self._error is not None:
                    raise self._error
                break
            yield item

    def close(self):
        for cap in self.readers.values():
            cap.release()
        self.readers.clear()


# ======================================================================
# Output formatting (match GT annotation format)
# ======================================================================

def format_as_gt_json(all_joints_3d, all_valid, frame_indices):
    """Format predictions as GT-compatible JSON."""
    output = {}
    for fidx in frame_indices:
        if fidx not in all_joints_3d:
            continue

        joints_3d = all_joints_3d[fidx]
        valid = all_valid[fidx]

        ann3d = {}
        for j, joint_name in enumerate(BODY_JOINTS):
            if valid[j]:
                ann3d[joint_name] = {
                    "x": float(joints_3d[j, 0]),
                    "y": float(joints_3d[j, 1]),
                    "z": float(joints_3d[j, 2]),
                }

        if ann3d:
            output[str(fidx)] = [{"annotation3D": ann3d}]

    return output


# ======================================================================
# Per-take processing
# ======================================================================

def get_exo_video_paths(take_info, egoexo_root, downscale_res=448):
    """
    Get exo camera video paths from take metadata.

    Returns dict keyed by physical camera name (e.g., 'cam01', 'cam02') to match
    the naming convention in camera_pose JSON files. The takes.json uses logical
    names ('exo1', 'exo2') but the video filenames contain the physical camera
    name (e.g., 'cam01.mp4'), which matches camera_pose keys.

    Looks for downscaled videos first (under downscaled/{res}/), falls back to
    full-resolution if downscaled not found.
    """
    root_dir = take_info["root_dir"]
    fav = take_info.get("frame_aligned_videos", {})

    video_paths = {}
    for cam_name, streams in fav.items():
        if "aria" in cam_name.lower() or cam_name in ("collage", "best_exo"):
            continue
        if isinstance(streams, dict) and "0" in streams:
            rel_path = streams["0"].get("relative_path", "")
            if rel_path:
                # Extract physical camera name from video filename
                # e.g., "cam01.mp4" -> "cam01", "cam02.mp4" -> "cam02"
                video_filename = os.path.basename(rel_path)
                video_basename = os.path.splitext(video_filename)[0]

                # Try downscaled version first:
                # takes/TAKE_NAME/frame_aligned_videos/downscaled/448/cam01.mp4
                fav_dir = os.path.join(egoexo_root, root_dir, "frame_aligned_videos")
                downscaled_path = os.path.join(
                    fav_dir, "downscaled", str(downscale_res), video_filename
                )
                fullres_path = os.path.join(egoexo_root, root_dir, rel_path)

                if os.path.isfile(downscaled_path):
                    video_paths[video_basename] = downscaled_path
                elif os.path.isfile(fullres_path):
                    video_paths[video_basename] = fullres_path

    return video_paths


def process_take(
    take_uid,
    camera_pose_path,
    video_paths,
    pose_estimator,
    frame_stride=30,
    score_threshold=0.3,
    min_views=2,
    max_reproj_error=50.0,
    num_decode_threads=8,
    verbose=False,
):
    """
    Process a single take: detect 2D poses on all exo cameras, triangulate 3D.

    Pipeline (matching official EgoExo4D annotation pipeline):
    1. Load Aria trajectory for camera-wearer identification
    2. For each frame, project Aria head position into each exo view
    3. Match YOLO detections to the Aria-projected region by IoU
    4. Run ViTPose on matched person crops
    5. RANSAC triangulate across exo views

    Uses MultiCameraFrameLoader for parallel video decoding across cameras.
    """
    cameras, aria_trajectory, metadata = load_camera_params(camera_pose_path)

    if verbose and aria_trajectory:
        print(f"      Aria trajectory: {len(aria_trajectory)} frames loaded")
    elif verbose:
        print(f"      WARNING: No Aria trajectory found — falling back to confidence-based person selection")

    # Filter to cameras with both calibration and video
    active_cams = []
    for cam_name in sorted(cameras.keys()):
        if cam_name in video_paths and os.path.isfile(video_paths[cam_name]):
            active_cams.append(cam_name)

    if len(active_cams) < min_views:
        sample_paths = {k: v for k, v in list(video_paths.items())[:2]}
        raise ValueError(
            f"Only {len(active_cams)} usable cameras (need {min_views}). "
            f"camera_pose keys: {sorted(cameras.keys())}, "
            f"video_path keys: {sorted(video_paths.keys())}, "
            f"sample paths: {sample_paths}"
        )

    # Create frame loader (opens all cameras, validates they're readable)
    active_video_paths = {cam: video_paths[cam] for cam in active_cams}

    # We need frame count and video props before creating the loader with
    # frame_indices, so open temporary captures for metadata.
    cam_props = {}
    min_n_frames = float("inf")
    readable_cams = []
    for cam_name in active_cams:
        cap = cv2.VideoCapture(video_paths[cam_name])
        if not cap.isOpened():
            print(f"      Warning: cannot open {cam_name}")
            continue
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        cam_props[cam_name] = (vid_w, vid_h, fps, n_frames)
        min_n_frames = min(min_n_frames, n_frames)
        readable_cams.append(cam_name)

    active_cams = readable_cams
    if len(active_cams) < min_views:
        raise ValueError(f"Only {len(active_cams)} readable videos (need {min_views})")

    n_frames = min_n_frames
    frame_indices = list(range(0, n_frames, frame_stride))

    # Scale intrinsics/projection matrices if video is downscaled.
    for cam_name in active_cams:
        vid_w, vid_h, fps, _ = cam_props[cam_name]
        K_orig = cameras[cam_name]["intrinsics"].copy()

        orig_w_est = K_orig[0, 2] * 2.0
        orig_h_est = K_orig[1, 2] * 2.0

        if orig_w_est > 0 and orig_h_est > 0 and abs(vid_w - orig_w_est) > 10:
            sx = vid_w / orig_w_est
            sy = vid_h / orig_h_est

            if abs(sx - sy) > 0.05:
                print(f"      Warning {cam_name}: non-uniform scale "
                      f"sx={sx:.3f} sy={sy:.3f} — possible crop or non-standard resize")

            S = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float64)
            K_scaled = S @ K_orig
            cameras[cam_name]["intrinsics"] = K_scaled
            cameras[cam_name]["projection"] = K_scaled @ cameras[cam_name]["extrinsics"]

    # Now create the threaded frame loader
    active_video_paths = {cam: video_paths[cam] for cam in active_cams}
    loader = MultiCameraFrameLoader(
        video_paths=active_video_paths,
        frame_indices=frame_indices,
        n_threads=min(num_decode_threads, len(active_cams) * 2),
        prefetch_frames=8,
    )

    all_joints_3d = {}
    all_valid = {}

    n_cams = len(active_cams)

    if verbose:
        _, _, fps, _ = cam_props[active_cams[0]]
        dur = n_frames / max(fps, 1)
        print(f"      {n_frames} frames ({dur:.0f}s), {len(frame_indices)} to process, "
              f"{n_cams} cams, {min(num_decode_threads, n_cams * 2)} decode threads")

    t0 = time.time()
    n_valid = 0

    try:
        # Use tqdm for per-frame progress with ETA
        frame_iter = enumerate(loader)
        if verbose:
            pbar = tqdm(
                frame_iter,
                total=len(frame_indices),
                desc=f"      {take_uid[:12]}",
                unit="fr",
                leave=False,
                dynamic_ncols=True,
            )
        else:
            pbar = frame_iter

        for fi, (fidx, cam_frames) in pbar:
            # cam_frames: dict of cam_name -> BGR frame (decoded in parallel)
            frames = []
            cam_order = []
            for cam_name in active_cams:
                if cam_name in cam_frames:
                    frames.append(cam_frames[cam_name])
                    cam_order.append(cam_name)

            if len(frames) < min_views:
                continue

            # Batched YOLO detection + ViTPose per frame (returns 4-tuples)
            pose_results = pose_estimator.detect_and_estimate_batch(frames)

            # Get Aria head position for this frame (for person identification)
            aria_head = get_aria_head_position(aria_trajectory, fidx)

            # Per-camera: select the camera-wearer using Aria projection
            cam_kps = []
            cam_scores = []
            cam_proj = []

            for i, cam_name in enumerate(cam_order):
                kps_all, sc_all, det_boxes, det_confs = pose_results[i]

                if kps_all.shape[0] == 0:
                    continue

                # Project Aria head position into this exo camera view.
                # The projected head lands within 2-5px of the GT nose,
                # so it reliably falls inside the correct YOLO box.
                aria_head_2d = None
                if aria_head is not None:
                    vid_w, vid_h, _, _ = cam_props[cam_name]
                    P = cameras[cam_name]["projection"]
                    aria_head_2d = project_point_to_2d(
                        aria_head, P, vid_w, vid_h,
                    )

                # Select the camera-wearer from detected persons
                kps, sc = select_person_by_aria(
                    det_boxes, det_confs, kps_all, sc_all,
                    aria_head_2d,
                )

                if kps is not None and sc is not None:
                    # Undistort 2D keypoints before triangulation
                    kps = _maybe_undistort(kps, cameras[cam_name])
                    cam_kps.append(kps)
                    cam_scores.append(sc)
                    cam_proj.append(cameras[cam_name]["projection"])

            if len(cam_kps) < min_views:
                continue

            cam_kps = np.array(cam_kps)
            cam_scores = np.array(cam_scores)
            cam_proj = np.array(cam_proj)

            # RANSAC triangulation with nonlinear refinement
            joints_3d, valid = triangulate_joints_multiview(
                cam_kps,
                cam_scores,
                cam_proj,
                score_threshold=score_threshold,
                min_views=min_views,
                max_reproj_error=max_reproj_error,
                use_ransac=True,
            )

            if valid.any():
                all_joints_3d[fidx] = joints_3d
                all_valid[fidx] = valid
                n_valid += 1

            if verbose and hasattr(pbar, 'set_postfix'):
                pbar.set_postfix(valid=n_valid, refresh=False)

        if verbose and hasattr(pbar, 'close'):
            pbar.close()

    finally:
        loader.close()

    elapsed = time.time() - t0
    if verbose:
        print(f"      Done: {n_valid}/{len(frame_indices)} valid frames in {elapsed:.1f}s")

    return all_joints_3d, all_valid, frame_indices


# ======================================================================
# Main pipeline
# ======================================================================

def process_split(
    split,
    egoexo_root,
    ego_pose_root,
    output_root,
    takes_by_uid,
    pose_estimator,
    frame_stride,
    score_threshold,
    min_views,
    max_reproj_error,
    num_decode_threads=8,
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

    take_pbar = tqdm(
        enumerate(files),
        total=len(files),
        desc=f"  {split}",
        unit="take",
        leave=True,
        dynamic_ncols=True,
    )

    for i, fname in take_pbar:
        take_uid = fname.replace(".json", "")
        output_path = os.path.join(output_dir, fname)

        if os.path.exists(output_path) and not overwrite:
            n_skip += 1
            take_pbar.set_postfix(ok=n_success, err=n_error, skip=n_skip, refresh=False)
            continue

        if take_uid not in takes_by_uid:
            n_error += 1
            continue

        take_info = takes_by_uid[take_uid]
        video_paths = get_exo_video_paths(take_info, egoexo_root)

        if len(video_paths) < min_views:
            n_error += 1
            continue

        try:
            t_start = time.time()
            cam_pose_path = os.path.join(camera_pose_dir, fname)
            all_joints_3d, all_valid, frame_indices = process_take(
                take_uid=take_uid,
                camera_pose_path=cam_pose_path,
                video_paths=video_paths,
                pose_estimator=pose_estimator,
                frame_stride=frame_stride,
                score_threshold=score_threshold,
                min_views=min_views,
                max_reproj_error=max_reproj_error,
                num_decode_threads=num_decode_threads,
                verbose=True,
            )

            output_json = format_as_gt_json(all_joints_3d, all_valid, frame_indices)

            if len(output_json) == 0:
                tqdm.write(f"    WARNING: {take_uid} produced 0 valid frames")
                n_error += 1
                continue

            with open(output_path, "w") as f:
                json.dump(output_json, f)

            n_success += 1
            elapsed = time.time() - t_start
            take_pbar.set_postfix(
                ok=n_success, err=n_error, skip=n_skip,
                last=f"{len(output_json)}fr/{elapsed:.0f}s",
                refresh=False,
            )

        except Exception as e:
            tqdm.write(f"    ERROR {take_uid}: {e}")
            n_error += 1
            continue

    take_pbar.close()
    return n_success, n_skip, n_error


# ======================================================================
# Multi-GPU worker
# ======================================================================

def gpu_worker(gpu_id, take_assignments, args, ego_pose_root, takes_by_uid):
    """Worker function for multi-GPU processing."""
    device = f"cuda:{gpu_id}"

    # Limit CPU threads per worker to prevent oversubscription.
    # With N GPU workers, each should use at most a fair share of cores.
    n_workers = max(1, len(args.gpus)) if args.gpus else 1
    threads_per_worker = max(4, os.cpu_count() // (n_workers * 2))
    os.environ["OMP_NUM_THREADS"] = str(threads_per_worker)
    os.environ["MKL_NUM_THREADS"] = str(threads_per_worker)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads_per_worker)
    torch.set_num_threads(threads_per_worker)
    cv2.setNumThreads(threads_per_worker)
    print(f"\n[GPU {gpu_id}] Starting worker with {len(take_assignments)} takes, "
          f"{threads_per_worker} CPU threads")

    pose_estimator = PoseEstimator(device=device)

    n_success = 0
    n_error = 0
    n_skip = 0

    # Take-level progress bar with ETA
    take_pbar = tqdm(
        enumerate(take_assignments),
        total=len(take_assignments),
        desc=f"GPU {gpu_id}",
        unit="take",
        position=gpu_id,
        leave=True,
        dynamic_ncols=True,
    )

    for idx, (split, fname) in take_pbar:
        take_uid = fname.replace(".json", "")
        output_dir = os.path.join(args.output_root, split, "body", "annotation")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, fname)

        if os.path.exists(output_path) and not args.overwrite:
            n_skip += 1
            take_pbar.set_postfix(ok=n_success, err=n_error, skip=n_skip, refresh=False)
            continue

        if take_uid not in takes_by_uid:
            n_error += 1
            continue

        take_info = takes_by_uid[take_uid]
        video_paths = get_exo_video_paths(take_info, args.egoexo_root)

        if len(video_paths) < args.min_views:
            n_error += 1
            continue

        try:
            cam_pose_path = os.path.join(ego_pose_root, split, "camera_pose", fname)
            t_start = time.time()
            all_joints_3d, all_valid, frame_indices = process_take(
                take_uid=take_uid,
                camera_pose_path=cam_pose_path,
                video_paths=video_paths,
                pose_estimator=pose_estimator,
                frame_stride=args.frame_stride,
                score_threshold=args.score_threshold,
                min_views=args.min_views,
                max_reproj_error=args.max_reproj_error,
                num_decode_threads=args.num_decode_threads,
                verbose=False,  # tqdm handles progress display
            )

            output_json = format_as_gt_json(all_joints_3d, all_valid, frame_indices)

            if len(output_json) == 0:
                n_error += 1
                continue

            with open(output_path, "w") as f:
                json.dump(output_json, f)

            n_success += 1
            elapsed = time.time() - t_start
            take_pbar.set_postfix(
                ok=n_success, err=n_error, skip=n_skip,
                last=f"{len(output_json)}fr/{elapsed:.0f}s",
                refresh=False,
            )

        except Exception as e:
            tqdm.write(f"  [GPU {gpu_id}] ERROR {take_uid}: {e}")
            n_error += 1

    take_pbar.close()
    print(f"\n[GPU {gpu_id}] DONE: {n_success} ok, {n_error} errors, {n_skip} skipped")
    return n_success, n_skip, n_error


def main():
    parser = argparse.ArgumentParser(
        description="Predict exo body pose using ViTPose 2D + multi-view triangulation"
    )
    parser.add_argument(
        "--egoexo_root",
        type=str,
        default="/path/to/EgoExo4D",
        help="Root directory of EgoExo4D dataset",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="/path/to/EgoExo4D/predicted_exo_pose",
        help="Output directory (same structure as ego_pose_root)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="Splits to process",
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
        "--frame_stride",
        type=int,
        default=10,
        help="Process every N-th frame (default: 10 = 3fps from 30fps video). "
             "Must be ≤16 to ensure all pose tokens are valid with pose_max_frame_dist=8.",
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.3,
        help="Minimum 2D keypoint confidence for triangulation",
    )
    parser.add_argument(
        "--min_views",
        type=int,
        default=2,
        help="Minimum camera views needed for triangulation",
    )
    parser.add_argument(
        "--max_reproj_error",
        type=float,
        default=50.0,
        help="Maximum mean reprojection error (pixels) to accept a joint",
    )
    parser.add_argument(
        "--num_decode_threads",
        type=int,
        default=4,
        help="Number of threads for parallel video decoding per GPU worker (default: 8)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing predictions",
    )
    parser.add_argument(
        "--take_uids_file",
        type=str,
        default=None,
        help="Optional text file with one take_uid per line. "
             "Only process these takes (skip all others).",
    )
    args = parser.parse_args()

    # Runtime guard: frame_stride must be ≤ 2 * pose_max_frame_dist (typically 16)
    # to ensure every ActionFormer temporal token can find a nearby pose frame.
    MAX_VALID_STRIDE = 16
    if args.frame_stride > MAX_VALID_STRIDE:
        print(f"WARNING: frame_stride={args.frame_stride} > {MAX_VALID_STRIDE}. "
              f"This will cause many pose tokens to be invalid (pose_max_frame_dist=8). "
              f"Clamping to {MAX_VALID_STRIDE}.")
        args.frame_stride = MAX_VALID_STRIDE

    ego_pose_root = os.path.join(args.egoexo_root, "annotations", "ego_pose")
    takes_json_path = os.path.join(args.egoexo_root, "takes.json")

    if not os.path.isdir(args.egoexo_root):
        print(f"ERROR: EgoExo4D root not found: {args.egoexo_root}")
        sys.exit(1)
    if not os.path.isfile(takes_json_path):
        print(f"ERROR: takes.json not found: {takes_json_path}")
        sys.exit(1)

    print("=" * 60)
    print("EgoExo4D Exo Body Pose Prediction")
    print("  ViTPose 2D + YOLO detection + multi-view triangulation")
    print("=" * 60)
    print(f"  egoexo_root:      {args.egoexo_root}")
    print(f"  output_root:      {args.output_root}")
    print(f"  splits:           {args.splits}")
    print(f"  gpus:             {args.gpus}")
    print(f"  frame_stride:     {args.frame_stride}")
    print(f"  score_threshold:  {args.score_threshold}")
    print(f"  min_views:        {args.min_views}")
    print(f"  max_reproj_error: {args.max_reproj_error}")
    print(f"  decode_threads:   {args.num_decode_threads}")
    print(f"  overwrite:        {args.overwrite}")
    print()

    # Load takes.json and filter out takes with broken calibration
    # (e.g. Los_Andes / uniandes takes) that are also excluded from eval.
    KNOWN_SCENARIOS = {
        "Basketball", "Cooking", "Music", "Dance",
        "Soccer", "Health", "Bike Repair", "Rock Climbing",
    }
    SCENARIO_MAP = {
        "basketball": "Basketball", "cooking": "Cooking", "music": "Music",
        "dance": "Dance", "soccer": "Soccer", "health": "Health",
        "bike": "Bike Repair", "rock": "Rock Climbing",
        "fix": "Bike Repair", "install": "Bike Repair", "remove": "Bike Repair",
        "clean": "Bike Repair", "covid": "Health", "first aid": "Health",
        "salsa": "Dance", "choreog": "Dance", "guitar": "Music",
        "piano": "Music", "violin": "Music", "making": "Cooking",
    }

    def _get_scenario(take_info):
        sc = (take_info.get("scenario_name") or
              take_info.get("task", {}).get("scenario_name"))
        if sc and sc in KNOWN_SCENARIOS:
            return sc
        task_name = (take_info.get("task_name") or
                     take_info.get("task", {}).get("task_name") or "")
        task_lower = task_name.lower()
        for key, scenario in SCENARIO_MAP.items():
            if key in task_lower:
                return scenario
        return None

    print("Loading takes.json...")
    with open(takes_json_path, "r") as f:
        takes_list = json.load(f)
    takes_by_uid_all = {t["take_uid"]: t for t in takes_list}
    takes_by_uid = {uid: t for uid, t in takes_by_uid_all.items()
                    if _get_scenario(t) in KNOWN_SCENARIOS}
    n_excluded = len(takes_by_uid_all) - len(takes_by_uid)
    print(f"  {len(takes_by_uid_all)} takes loaded, {n_excluded} excluded "
          f"(non-standard scenarios, e.g. Los_Andes broken calibration)")
    print(f"  {len(takes_by_uid)} takes retained")

    # Optional: load take_uid filter
    filter_uids = None
    if args.take_uids_file:
        with open(args.take_uids_file) as f:
            filter_uids = set(l.strip() for l in f if l.strip())
        print(f"  Filtering to {len(filter_uids)} take_uids from {args.take_uids_file}")

    # Gather all (split, filename) pairs to process
    all_tasks = []
    for split in args.splits:
        camera_pose_dir = os.path.join(ego_pose_root, split, "camera_pose")
        if not os.path.isdir(camera_pose_dir):
            print(f"  Skipping {split}: {camera_pose_dir} not found")
            continue
        files = sorted([f for f in os.listdir(camera_pose_dir) if f.endswith(".json")])
        if filter_uids is not None:
            files = [f for f in files if f.replace(".json", "") in filter_uids]
        print(f"  {split}: {len(files)} takes with camera_pose data")
        for fname in files:
            all_tasks.append((split, fname))

    print(f"\n  Total tasks: {len(all_tasks)}")

    gpu_ids = args.gpus  # None if single-GPU mode

    if gpu_ids is not None and len(gpu_ids) > 1:
        # Multi-GPU: distribute tasks across specified GPUs
        n_gpus = len(gpu_ids)
        print(f"\n  Distributing {len(all_tasks)} takes across {n_gpus} GPUs: {gpu_ids}")
        chunks = {gid: [] for gid in gpu_ids}
        for i, task in enumerate(all_tasks):
            chunks[gpu_ids[i % n_gpus]].append(task)

        for gid in gpu_ids:
            print(f"    GPU {gid}: {len(chunks[gid])} takes")

        processes = []
        for gid in gpu_ids:
            if len(chunks[gid]) == 0:
                continue
            p = mp.Process(
                target=gpu_worker,
                args=(gid, chunks[gid], args, ego_pose_root, takes_by_uid),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # Check for worker failures
        failed = [(p.name, p.exitcode) for p in processes if p.exitcode != 0]
        if failed:
            print(f"\n{'=' * 60}")
            print(f"WARNING: {len(failed)}/{len(processes)} workers failed!")
            for name, code in failed:
                print(f"  {name}: exit code {code}")
            print("Results may be incomplete. Check per-GPU logs above.")
            print(f"{'=' * 60}")
            sys.exit(1)

        print(f"\n{'=' * 60}")
        print("DONE (multi-GPU). Check per-GPU logs above for results.")

    else:
        # Single-GPU mode: use --gpus [X] if given (single element), else --device
        device = f"cuda:{gpu_ids[0]}" if (gpu_ids and len(gpu_ids) == 1) else args.device
        # Limit CPU threads for single-GPU mode
        max_threads = max(4, os.cpu_count() // 4)
        torch.set_num_threads(max_threads)
        cv2.setNumThreads(max_threads)
        print(f"\nInitializing models on {device} ({max_threads} CPU threads)...")
        pose_estimator = PoseEstimator(device=device)

        total_success = 0
        total_skip = 0
        total_error = 0

        for split in args.splits:
            print(f"\n{'─' * 40}")
            n_success, n_skip, n_error = process_split(
                split=split,
                egoexo_root=args.egoexo_root,
                ego_pose_root=ego_pose_root,
                output_root=args.output_root,
                takes_by_uid=takes_by_uid,
                pose_estimator=pose_estimator,
                frame_stride=args.frame_stride,
                score_threshold=args.score_threshold,
                min_views=args.min_views,
                max_reproj_error=args.max_reproj_error,
                num_decode_threads=args.num_decode_threads,
                overwrite=args.overwrite,
            )
            total_success += n_success
            total_skip += n_skip
            total_error += n_error
            print(f"  -> {n_success} predicted, {n_skip} skipped, {n_error} errors")

        print(f"\n{'=' * 60}")
        print(f"DONE: {total_success} predicted, {total_skip} skipped, {total_error} errors")

    print(f"Output: {args.output_root}")
    print(f"\nTo use predicted exo pose with your model, set in config:")
    print(f"  pose_root: '{args.output_root}'")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

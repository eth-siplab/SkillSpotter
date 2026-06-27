import os
import torch
import numpy as np
import warnings

# torchvision + decord are only needed for raw-video clip extraction
# (eval.py --save_clips). The core train/eval path does not require them, so
# they are imported lazily to keep the package importable without video libs.
try:
    import torchvision  # noqa: F401
    import decord
    # Configure Decord and Suppress Warnings globally for any script using this
    decord.bridge.set_bridge('torch')
    warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.io")
except ImportError:
    decord = None


def get_clips_folder_name(clip_len, fps_out, resize_hw, bg_ratio, bg_min_dist, curr_epoch=None):
    folder_name = (f"CL{clip_len}_FPS{fps_out}_HW{resize_hw[0]}x{resize_hw[1]}_"
                   f"BGRATIO{int(bg_ratio)}_BGMINDIST{int(bg_min_dist)}")

    if curr_epoch is not None:
        folder_name = f"epoch_{curr_epoch}/" + folder_name

    return folder_name



def extract_clips_from_video(
        video_path,
        clips_config,
        save_format='mp4',
        fps_out=15,
        clip_len=32,
        resize_hw=(448, 448)
):
    """
    Opens a video file ONCE and extracts multiple clips from it.

    Args:
        video_path (str): Full path to the source MP4.
        clips_config (list[dict]): List of dictionaries. Each must contain:
            - 'timestamp': float (Center time in seconds)
            - 'save_path': str (Full output path)
        fps_out (int): Target FPS.
        clip_len (int): Number of frames.
        resize_hw (tuple): (Height, Width).

    Returns:
        tuple: (success_count, list_of_errors)
    """
    errors = []
    success_count = 0

    # Filter: Only process clips that don't exist yet
    queue = [c for c in clips_config if not os.path.exists(c['save_path'])]

    # Optimization: If all clips exist, return immediately
    if len(queue) == 0:
        return len(clips_config), []

    try:
        # 1. OPEN VIDEO (The expensive operation)
        # ctx=decord.cpu(0) is safest for dataloaders/multiprocessing
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0), num_threads=1)

        avg_fps = vr.get_avg_fps()
        if avg_fps <= 0 or np.isnan(avg_fps): avg_fps = 30.0
        video_len = len(vr)

        # 2. ITERATE AND CUT
        for item in queue:
            try:
                timestamp = item['timestamp']
                save_path = item['save_path']

                # --- Math: Calculate Indices ---
                center_frame_idx = int(timestamp * avg_fps)
                stride = max(1, avg_fps / fps_out)
                start_frame = center_frame_idx - (clip_len * stride) / 2

                frame_indices = np.arange(clip_len) * stride + start_frame
                frame_indices = np.clip(frame_indices, 0, video_len - 1).astype(int)

                # --- Decode (Decord C++) ---
                # Returns (T, H, W, C)
                frames = vr.get_batch(frame_indices)

                # --- Resize (Torch CPU) ---
                # Permute to (T, C, H, W) for interpolate
                frames = frames.permute(0, 3, 1, 2).float()

                frames_resized = torch.nn.functional.interpolate(
                    frames,
                    size=resize_hw,
                    mode='bilinear',
                    align_corners=False
                )

                # --- Save (Torchvision ffmpeg) ---
                if save_format == 'mp4':
                    frames_out = frames_resized.permute(0, 2, 3, 1).byte()  # write_video needs (T, H, W, C)
                    torchvision.io.write_video(
                        save_path,
                        frames_out,
                        fps=fps_out,
                        video_codec='h264',
                        options={'crf': '23', 'threads': '1'}
                    )
                elif save_format == 'pt':
                    tensor_to_save = frames_resized.byte()
                    torch.save(tensor_to_save, save_path)
                else:
                    raise NotImplementedError(f"Unsupported save format: {save_format}")

                success_count += 1

            except Exception as e:
                errors.append(f"Clip {timestamp:.2f} failed: {e}")

    except Exception as e:
        return 0, [f"Video Open Failed {os.path.basename(video_path)}: {e}"]

    # Total success = (Skipped because existed) + (Newly created)
    total_success = (len(clips_config) - len(queue)) + success_count
    return total_success, errors
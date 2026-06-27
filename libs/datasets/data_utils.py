import os
import copy
import random
import numpy as np
import torch

# Raw-video decoding libraries (av / decord / PIL) are only used by the
# clip-reading helpers below. The Omnivore-feature train/eval pipeline loads
# pre-extracted .npy features and does not need them, so they are optional.
try:
    import av
    import decord
    import PIL.Image as Image
    from decord import VideoReader, cpu
    decord.bridge.set_bridge('torch')
except ImportError:
    av = None
    decord = None
    Image = None
    VideoReader = cpu = None


def trivial_batch_collator(batch):
    """
        A batch collator that does nothing
    """
    return batch

def worker_init_reset_seed(worker_id):
    """
        Reset random seed for each worker
    """
    seed = torch.initial_seed() % 2 ** 31
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def truncate_feats(
    data_dict,
    max_seq_len,
    trunc_thresh,
    offset,
    crop_ratio=None,
    max_num_trials=200,
    has_action=True,
    no_trunc=False
):
    """
    Truncate feats and time stamps in a dict item

    data_dict = {'video_id'        : str
                 'feats'           : Tensor C x T
                 'segments'        : Tensor N x 2 (in feature grid)
                 'labels'          : Tensor N
                 'fps'             : float
                 'feat_stride'     : int
                 'feat_num_frames' : in

    """
    # get the meta info
    feat_len = data_dict['feats'].shape[1]
    num_segs = data_dict['segments'].shape[0]

    # seq_len < max_seq_len
    if feat_len <= max_seq_len:
        # do nothing
        if crop_ratio == None:
            return data_dict
        # randomly crop the seq by setting max_seq_len to a value in [l, r]
        else:
            max_seq_len = random.randint(
                max(round(crop_ratio[0] * feat_len), 1),
                min(round(crop_ratio[1] * feat_len), feat_len),
            )
            # # corner case
            if feat_len == max_seq_len:
                return data_dict

    # otherwise, deep copy the dict
    data_dict = copy.deepcopy(data_dict)

    # try a few times till a valid truncation with at least one action
    for _ in range(max_num_trials):

        # sample a random truncation of the video feats
        st = random.randint(0, feat_len - max_seq_len)
        ed = st + max_seq_len
        window = torch.as_tensor([st, ed], dtype=torch.float32)

        # compute the intersection between the sampled window and all segments
        window = window[None].repeat(num_segs, 1)
        left = torch.maximum(window[:, 0] - offset, data_dict['segments'][:, 0])
        right = torch.minimum(window[:, 1] + offset, data_dict['segments'][:, 1])
        inter = (right - left).clamp(min=0)
        area_segs = torch.abs(
            data_dict['segments'][:, 1] - data_dict['segments'][:, 0])
        inter_ratio = inter / area_segs

        # only select those segments over the thresh
        seg_idx = (inter_ratio >= trunc_thresh)

        if no_trunc:
            # with at least one action and not truncating any actions
            seg_trunc_idx = torch.logical_and(
                (inter_ratio > 0.0), (inter_ratio < 1.0)
            )
            if (seg_idx.sum().item() > 0) and (seg_trunc_idx.sum().item() == 0):
                break
        elif has_action:
            # with at least one action
            if seg_idx.sum().item() > 0:
                break
        else:
            # without any constraints
            break

    # feats: C x T
    data_dict['feats'] = data_dict['feats'][:, st:ed].clone()
    # segments: N x 2 in feature grids
    data_dict['segments'] = torch.stack((left[seg_idx], right[seg_idx]), dim=1)
    # shift the time stamps due to truncation
    data_dict['segments'] = data_dict['segments'] - st
    # labels: N
    data_dict['labels'] = data_dict['labels'][seg_idx].clone()

    return data_dict


def truncate_feats_timestamps(
    data_dict,
    max_seq_len,
    trunc_thresh,
    offset,
    crop_ratio=None,
    max_num_trials=200,
    has_action=True,
    no_trunc=False
):
    """
    Truncate feats and time stamps in a dict item

    data_dict = {'video_id'        : str
                 'feats'           : Tensor C x T
                 'segments'        : Tensor N x 2 (in feature grid)
                 'labels'          : Tensor N
                 'fps'             : float
                 'feat_stride'     : int
                 'feat_num_frames' : in

    """
    # get the meta info
    feat_len = data_dict['feats'].shape[1]

    # seq_len < max_seq_len
    if feat_len <= max_seq_len:
        # do nothing
        if crop_ratio == None:
            return data_dict
        # randomly crop the seq by setting max_seq_len to a value in [l, r]
        else:
            max_seq_len = random.randint(
                max(round(crop_ratio[0] * feat_len), 1),
                min(round(crop_ratio[1] * feat_len), feat_len),
            )
            # # corner case
            if feat_len == max_seq_len:
                return data_dict

    # otherwise, deep copy the dict
    data_dict = copy.deepcopy(data_dict)

    # try a few times till a valid truncation with at least one action
    for _ in range(max_num_trials):

        # sample a random truncation of the video feats
        st = random.randint(0, feat_len - max_seq_len)
        ed = st + max_seq_len

        # compute the intersection between the sampled window and all segments
        seg_idx = torch.logical_and(
            (data_dict['reg_points'][:] >= st - offset),
            (data_dict['reg_points'][:] <= ed + offset)
        )

        if no_trunc:
            # check all reg_points are exactly inside the window
            seg_trunc_idx = torch.logical_and(
                (data_dict['reg_points'][:] < st - offset),
                (data_dict['reg_points'][:] > ed + offset)
            )
            if (seg_idx.sum().item() > 0) and (seg_trunc_idx.sum().item() == 0):
                break
        elif has_action:
            # with at least one action
            if seg_idx.sum().item() > 0:
                break
        else:
            # without any constraints
            break

    # feats: C x T
    data_dict['feats'] = data_dict['feats'][:, st:ed].clone()
    # cross-view features: truncate identically if present
    if 'feats_ego' in data_dict:
        data_dict['feats_ego'] = data_dict['feats_ego'][:, st:ed].clone()
    if 'feats_exo' in data_dict:
        data_dict['feats_exo'] = data_dict['feats_exo'][:, st:ed].clone()
    # segments: N x 2 in feature grids
    data_dict['reg_points'] = data_dict['reg_points'][seg_idx].clone()
    # shift the time stamps due to truncation
    data_dict['reg_points'] = data_dict['reg_points'] - st
    # labels: N
    data_dict['labels'] = data_dict['labels'][seg_idx].clone()

    return data_dict


def load_clip_pyav(
        path: str,
        center_s: float,
        clip_len: int = 32,
        fps_out: float = 16.0,
        resize_hw=(448, 448),  # Updated default to 448 for H200s
        pad_mode: str = "edge",
) -> torch.Tensor:
    """
    Load a short clip around center_s from `path` using PyAV.
    Resizes using Letterboxing (preserving aspect ratio) to target `resize_hw`.
    Returns (C, T, H, W) float32 in [0, 1].
    """

    # Helper function for Letterboxing
    def resize_and_pad(img_arr, target_h, target_w):
        # Input: img_arr is (H, W, 3) numpy array
        pil_img = Image.fromarray(img_arr)
        orig_w, orig_h = pil_img.size

        # 1. Calculate scaling factor
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        # 2. Resize preserving aspect ratio
        resized_img = pil_img.resize((new_w, new_h), Image.BILINEAR)

        # 3. Create black background
        new_img = Image.new("RGB", (target_w, target_h), (0, 0, 0))

        # 4. Paste resized image in the center
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        new_img.paste(resized_img, (x_offset, y_offset))

        return np.array(new_img)

    # 1. Build target timestamps
    half = (clip_len - 1) / (2 * fps_out)
    t_start = max(0.0, center_s - half)
    t_end = t_start + (clip_len - 1) / fps_out
    ts = np.linspace(t_start, t_end, clip_len)

    try:
        container = av.open(path)
        stream = container.streams.video[0]
    except Exception as e:
        # Robustness for corrupted videos
        print(f"Error opening {path}: {e}")
        return torch.zeros(3, clip_len, resize_hw[0], resize_hw[1])

    # 2. Seek
    seek_time = max(0.0, t_start - 1.0)
    # Handle case where time_base is None or weird
    tb = float(stream.time_base) if stream.time_base else 1 / stream.average_rate
    container.seek(int(seek_time / tb), any_frame=False, backward=True, stream=stream)

    frames = []
    ts_idx = 0
    target = ts[ts_idx]

    # Optimization: Pre-calculate target dims
    tgt_h, tgt_w = resize_hw

    # 3. Decode and Process
    try:
        for packet in container.demux(stream):
            for frame in packet.decode():
                if frame.time is None:
                    continue
                frame_t = float(frame.time)

                while ts_idx < len(ts) and frame_t >= target:
                    img = frame.to_rgb().to_ndarray()  # (H,W,3)

                    if resize_hw is not None:
                        # --- MODIFIED SECTION START ---
                        img = resize_and_pad(img, tgt_h, tgt_w)
                        # --- MODIFIED SECTION END ---

                    frames.append(img)
                    ts_idx += 1
                    if ts_idx >= len(ts):
                        break
                    target = ts[ts_idx]
                if ts_idx >= len(ts):
                    break
            if ts_idx >= len(ts):
                break
    except Exception as e:
        print(f"Error decoding {path}: {e}")
    finally:
        container.close()

    # 4. Pad frames if needed (start/end of video issues)
    if len(frames) == 0:
        # Return pure black clip
        frames = [np.zeros((tgt_h, tgt_w, 3), dtype=np.uint8)] * clip_len
    elif len(frames) < clip_len:
        if pad_mode == "edge":
            pad_frame = frames[-1]
        else:
            pad_frame = np.zeros((tgt_h, tgt_w, 3), dtype=np.uint8)

        while len(frames) < clip_len:
            frames.append(pad_frame.copy())

    # 5. To Tensor
    arr = np.stack(frames[:clip_len], axis=0)  # (T, H, W, 3)
    arr = torch.from_numpy(arr).permute(3, 0, 1, 2).float()  # / 255.0  # (C, T, H, W)

    return arr


def load_clip_decord(
        path: str,
        center_s: float,
        clip_len: int = 32,
        fps_out: float = 15.0,
        resize_hw=(448, 448)
):
    """
    High-performance video loader using Decord + PyTorch Resizing.
    Optimized for H200 training pipelines to prevent CPU bottlenecks.
    """
    try:
        # 1. INITIALIZE DECORD
        # ctx=cpu(0) keeps decoding on CPU (faster for dataloaders).
        # num_threads=1 is CRITICAL. Without this, Decord spawns threads inside
        # PyTorch workers, causing a deadlock (stuck at 0%).
        vr = VideoReader(path, ctx=cpu(0), num_threads=1)

        # 2. FRAME INDEXING
        avg_fps = vr.get_avg_fps()
        # Fallback for weird metadata
        if avg_fps <= 0: avg_fps = 30.0

        # Calculate stride to match target fps_out
        stride = max(1, avg_fps / fps_out)

        # Center the window
        center_frame_idx = int(center_s * avg_fps)
        start_frame = center_frame_idx - (clip_len * stride) / 2

        # Vectorized index generation
        frame_indices = np.arange(clip_len) * stride + start_frame

        # Clamp indices to valid video range (0 to len-1)
        # .astype(int) is required for Decord
        frame_indices = np.clip(frame_indices, 0, len(vr) - 1).astype(int)

        # 3. FAST BATCH DECODE
        # get_batch fetches all frames in parallel C++ threads
        # Returns Tensor: (Time, Height, Width, Channel)
        video_data = vr.get_batch(frame_indices)

        # Permute to (Time, Channel, Height, Width) for PyTorch Interpolate
        video_data = video_data.permute(0, 3, 1, 2).float()

        # 4. FAST RESIZE & LETTERBOX (PyTorch Accelerated)
        if resize_hw is not None:
            target_h, target_w = resize_hw
            t, c, h, w = video_data.shape

            # Calculate Scale (Aspect Ratio Preserved)
            scale = min(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)

            # Resize using Bilinear Interpolation
            # We treat 'Time' as the Batch dimension here, resizing all frames at once
            if new_w != w or new_h != h:
                video_data = torch.nn.functional.interpolate(
                    video_data,
                    size=(new_h, new_w),
                    mode='bilinear',
                    align_corners=False
                )

            # Letterbox Padding (if needed)
            if new_w != target_w or new_h != target_h:
                # Create black canvas (0.0 is black)
                canvas = torch.zeros(t, c, target_h, target_w, dtype=torch.float32)

                # Center offsets
                y_off = (target_h - new_h) // 2
                x_off = (target_w - new_w) // 2

                # Paste resized video into center
                canvas[:, :, y_off:y_off + new_h, x_off:x_off + new_w] = video_data
                video_data = canvas

        # 5. FINAL PERMUTE
        # Model expects: (Channel, Time, Height, Width)
        video_data = video_data.permute(1, 0, 2, 3)

        return video_data

    except Exception as e:
        # Return a black clip (safeguard against corrupted files)
        # print(f"Warning: Error loading {path}: {e}")
        return torch.zeros(3, clip_len, resize_hw[0], resize_hw[1])
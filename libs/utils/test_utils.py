import numpy as np
import os
import pickle
import time
import torch
import tqdm
import json
import multiprocessing as mp
from collections import defaultdict
from functools import partial

from .postprocessing import postprocess_results
from .test_registry import register_test_func
from .train_utils import AverageMeter
from .video_processing import extract_clips_from_video

# -------------------------------------------------------------
# 1. UNPACKING STRATEGIES
# -------------------------------------------------------------
def unpack_timestamps(output, results):
    """Strategy to unpack Single Timestamps"""
    for vid_idx in range(len(output)):
        if output[vid_idx]['timestamps'].shape[0] > 0:
            results['video-id'].extend([output[vid_idx]['video_id']] * output[vid_idx]['timestamps'].shape[0])
            results['orig_video_file'].extend([output[vid_idx]['orig_video_file']] * output[vid_idx]['timestamps'].shape[0])
            results['time'].append(output[vid_idx]['timestamps'])
            results['label'].append(output[vid_idx]['labels'])
            results['score'].append(output[vid_idx]['scores'])


# -------------------------------------------------------------
# 2. CLIP SAVER (Standalone Function)
# -------------------------------------------------------------
# --- DEFINE INITIALIZER ---
def _init_worker():
    """
    Worker initializer.
    Note: Setting env vars here is often too late for numpy/torch imports.
    We rely on the parent setting them, but we enforce torch runtime settings here.
    """
    import torch
    import cv2
    # Force Runtime restrictions (these work even after import)
    torch.set_num_threads(1)
    cv2.setNumThreads(0)

# --- HELPER FOR PARALLEL EXTRACTION ---
def _parallel_extraction_worker(task, fps_out, clip_len, resize_hw, save_format):
    """
    Worker function to unpack arguments and run extraction.
    Must be at module level to be picklable.
    """
    vid_path, batch_config = task

    # Run the existing logic
    # Returns (success_count, errors_list)
    count, errors = extract_clips_from_video(
        video_path=vid_path,
        clips_config=batch_config,
        save_format=save_format,
        fps_out=fps_out,
        clip_len=clip_len,
        resize_hw=resize_hw
    )

    if errors:
        pass

    return count

def clip_saver_callback(results, save_folder, egoexo_type, clips_cfg):
    """
    Saves Top-K predicted clips and generates a JSON manifest.
    Designed to be called by Rank 0 after merging DDP results.
    """
    print(f"\n[Saver] Starting export of predictions to {save_folder}...")

    # 1. Config Loading
    top_k_clips = clips_cfg['top_k_clips']
    min_score_clips = clips_cfg['min_score_clips']
    save_format = clips_cfg['save_format']
    fps_out = clips_cfg['fps_out']
    clip_len = clips_cfg['clip_len']
    resize_hw = clips_cfg['resize_hw']
    if isinstance(resize_hw, list): resize_hw = tuple(resize_hw)
    json_filename = clips_cfg['json_file']

    annotations_file = clips_cfg['orig_json_file']
    video_to_scenario = {}
    if annotations_file and os.path.exists(annotations_file):
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)

        # Build lookup (adjust based on your JSON structure)
        video_to_scenario = {
            vid_id: info.get('scenario_name')
            for vid_id, info in annotations['database'].items()
        }

        print(f"[Saver] Loaded scenario info for {len(video_to_scenario)} videos")

    # 2. Group predictions by Video ID
    grouped_preds = defaultdict(list)
    num_preds = len(results['video-id'])

    for i in range(num_preds):
        score = float(results['score'][i])
        if score < min_score_clips: continue

        vid_id = results['video-id'][i]
        grouped_preds[vid_id].append({
            'orig_video_file': results['orig_video_file'][i],
            'timestamp': float(results['time'][i]),
            'score': score,
            'label': int(results['label'][i])
        })

    # 3. Sort and Truncate (Global Top-K)
    tasks = []
    manifest = []

    for vid_id, preds in grouped_preds.items():
        # Sort by score descending
        preds.sort(key=lambda x: x['score'], reverse=True)
        top_preds = preds[:top_k_clips]
        orig_video_file = preds[0]['orig_video_file']

        # Locate Source Video
        video_roots = clips_cfg['takes_root']
        if not isinstance(video_roots, (list, tuple)): video_roots = [video_roots]

        vid_path = None
        for root in video_roots:
            candidate = os.path.join(root, orig_video_file)
            if os.path.exists(candidate):
                vid_path = candidate
                break
            if vid_path: break

        if not vid_path:
            raise FileNotFoundError(f"Warning: No video path found for {vid_id}")

        # Prepare extraction tasks
        batch_config = []
        for i, pred in enumerate(top_preds):
            # Filename: vid_rank_score.pt
            filename = f"{vid_id}_rank{i}_s{pred['score']:.3f}.{save_format}"
            out_path = os.path.join(save_folder, filename)

            batch_config.append({
                'timestamp': pred['timestamp'],
                'save_path': out_path
            })

            # Manifest Entry
            manifest.append({
                'video_id': vid_id,
                'rel_path': filename,
                'text': "",
                'label': pred['label'],
                'center_time': pred['timestamp'],
                'subset': 'testing',
                'egoexo_type': egoexo_type,
                'score': pred['score'],
                'scenario_name': video_to_scenario[vid_id]
            })

        tasks.append((vid_path, batch_config))

    # 4. Execute Extraction
    print(f"[Saver] Extracting clips from {len(tasks)} videos using {clips_cfg['num_workers']} workers...")

    # Set variables to ensure multithreading does not use up all available CPUs
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    # Create partial function with fixed arguments
    worker_fn = partial(
        _parallel_extraction_worker,
        fps_out=fps_out,
        clip_len=clip_len,
        resize_hw=resize_hw,
        save_format=save_format
    )

    success_count = 0
    ctx = mp.get_context('spawn')  # Use 'spawn' context to avoid CUDA re-init issues in workers

    with ctx.Pool(processes=clips_cfg['num_workers'], initializer=_init_worker) as pool:
        for count in tqdm.tqdm(pool.imap_unordered(worker_fn, tasks), total=len(tasks)):
            success_count += count

    # 5. Save Manifest
    manifest_path = os.path.join(save_folder, egoexo_type + '_' + json_filename)
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"[Saver] Done! Saved {success_count} clips to {manifest_path}")

# -------------------------------------------------------------
# 3. Wrappers
# -------------------------------------------------------------
@register_test_func(['skillspotter'])
def test_timestamps_wrapper(test_loader, model, curr_epoch, args=None, **kwargs):
    # The saving logic must be handled in eval.py after DDP merge.
    return test_one_epoch_generic(
        test_loader, model, curr_epoch,
        result_keys=['video-id', 'orig_video_file', 'time', 'label', 'score'],
        unpack_func=unpack_timestamps,
        print_freq=args.print_freq
    )


def test_one_epoch_generic(
        test_loader,
        model,
        curr_epoch,
        result_keys,
        unpack_func,
        model_kwargs={},
        post_loop_callback=None,
        print_freq=20,
        rank=0,
):
    """Generic Validation/Test Loop"""
    batch_time = AverageMeter()
    model.eval()

    # 1. Init Results
    results = {k: [] for k in result_keys}
    start = time.time()

    # --- TQDM SETUP ---
    if rank == 0:
        # Show progress bar only on Main Process
        iterator = tqdm.tqdm(
            test_loader,
            total=len(test_loader),
            desc=f"Inference (Epoch {curr_epoch if curr_epoch >=0 else 'Test'})",
            mininterval=2.0 # Update every 2s to prevent PyCharm freezing
        )
    else:
        iterator = test_loader

    # 2. Inference Loop
    for iter_idx, video_list in enumerate(iterator):
        # if iter_idx == 1000: break
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(video_list, **model_kwargs)
            unpack_func(output, results)

        if (iter_idx != 0) and iter_idx % print_freq == 0:
            torch.cuda.synchronize()
            batch_time.update((time.time() - start) / print_freq)
            start = time.time()

    # 3. Concatenate (Lists -> Arrays)
    for k in result_keys:
        if len(results[k]) > 0:
            # Skip text lists
            if k in ['pred_text', 'gt_text', 'video-id', 'orig_video_file']:
                continue

            # Concat tensors/arrays
            if isinstance(results[k][0], torch.Tensor):
                results[k] = torch.cat(results[k]).float().numpy()
            elif isinstance(results[k][0], np.ndarray):
                results[k] = np.concatenate(results[k])
            else:
                results[k] = np.array(results[k])
        else:
            results[k] = np.array([])

    # 4. Optional Callback (Only used if safe for single process)
    if post_loop_callback:
        post_loop_callback(results)

    return results

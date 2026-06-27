# os imports
import os
import builtins
import time
import glob
import pickle
import argparse
import random
import gc
from joblib.externals.loky import get_reusable_executor
from pprint import pprint

# torch imports
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# our code
from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.evaluating import make_evaluator
from libs.modeling import make_meta_arch
from libs.utils import fix_random_seed
from libs.utils.test_registry import get_test_func
from libs.utils.test_utils import clip_saver_callback
from libs.utils.video_processing import get_clips_folder_name


def setup_ddp(rank, world_size):
    """Initialize distributed environment"""
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    dist.destroy_process_group()


def merge_results_dist(results, tmp_folder, rank, world_size):
    """
    Each rank saves its results to a pickle file.
    Rank 0 loads and merges them.
    """
    # 1. Save local results
    file_path = os.path.join(tmp_folder, f"results_part_{rank}.pkl")
    with open(file_path, "wb") as f:
        pickle.dump(results, f)
        f.flush()  # Force write to disk
        os.fsync(f.fileno())  # Ensure OS flushes buffer

    # 2. Sync to ensure all files are written
    dist.barrier(device_ids=[rank])

    # 3. Rank 0 merges
    combined_results = None
    if rank == 0:
        combined_results = {}
        print("Gathering results from all ranks...")

        for r in range(world_size):
            part_file = os.path.join(tmp_folder, f"results_part_{r}.pkl")

            # Robust Load
            part_data = None
            try:
                with open(part_file, "rb") as f:
                    part_data = pickle.load(f)
            except Exception as e:
                print(f"[Rank 0] Error loading {part_file}: {e}")
                continue

            if part_data is None or len(part_data) == 0:
                print(f"[Rank 0] Warning: Empty results from Rank {r}")
                continue

            # Initialize dict structure based on the first valid chunk we find
            if combined_results == {}:
                combined_results = {k: [] for k in part_data.keys()}

            # Extend lists
            for k, v in part_data.items():
                if k not in combined_results:
                    # If this rank has a key that previous ranks didn't (unlikely but possible), init it
                    combined_results[k] = []

                if isinstance(v, list):
                    combined_results[k].extend(v)
                elif hasattr(v, 'shape'):  # numpy array
                    if len(combined_results[k]) == 0:
                        combined_results[k] = v
                    else:
                        import numpy as np
                        combined_results[k] = np.concatenate((combined_results[k], v), axis=0)

    # Wait for Rank 0 to finish merging before anyone leaves
    dist.barrier(device_ids=[rank])
    return combined_results


def main_worker(rank, world_size, args, cfg, target_gpu=None):
    """
    Inference Logic.
    target_gpu: Explicit GPU ID for single mode. If None, uses 'rank' (DDP).
    """
    is_distributed = world_size > 1

    if is_distributed:
        setup_ddp(rank, world_size)
        if rank != 0:
            def print_pass(*args, **kwargs): pass

            builtins.print = print_pass

    # Determine which physical GPU to use
    gpu_id = target_gpu if target_gpu is not None else rank

    """1. Config & Paths"""
    prefix = ""

    # Add model_id if any
    if cfg['model'].get('model_id', None) is not None:
        prefix = prefix + cfg['model']['model_id'].split('/')[-1]

    # Add extension name if any
    if cfg['extension'] != "":
        if prefix == "":
            prefix = cfg['extension']
        else:
            prefix = prefix + '_' + cfg['extension']

    if prefix == "":
        prefix = "noext"

    args.ckpt = args.ckpt + '/' + prefix

    # Add ego/exo/both variant for EgoExo4D dataset
    if 'egoexo_type' in cfg['dataset']:
        args.ckpt = os.path.join(
            args.ckpt, cfg['dataset']['egoexo_type'])

    # Either use the best epoch, specific epoch, or the last epoch
    if args.use_best_epoch:
        ckpt_file = os.path.join(
            args.ckpt, 'model_best.pth.tar'
        )
        assert os.path.isfile(ckpt_file), "CKPT file for best model does not exist!"
        used_epoch = 'best'
    elif ".pth.tar" in args.ckpt:
        assert os.path.isfile(args.ckpt), "CKPT file does not exist!"
        ckpt_file = args.ckpt
        used_epoch = args.ckpt.split('_')[-1][:3]
    else:
        assert os.path.isdir(args.ckpt), "CKPT file folder does not exist!"
        if args.epoch > 0:
            ckpt_file = os.path.join(
                args.ckpt, 'epoch_{:03d}.pth.tar'.format(args.epoch)
            )
            used_epoch = 'epoch_{:03d}'.format(args.epoch)
        else:
            ckpt_file_list = sorted(glob.glob(os.path.join(args.ckpt, '*.pth.tar')))
            # check if model_best.pth.tar exists. If yes, remove it from the list
            if os.path.join(args.ckpt, 'model_best.pth.tar') in ckpt_file_list:
                ckpt_file_list.remove(os.path.join(args.ckpt, 'model_best.pth.tar'))
            assert len(ckpt_file_list) > 0, "No CKPT files found!"
            ckpt_file = ckpt_file_list[-1]
            used_epoch = ckpt_file.split('_')[-1][:3]
        assert os.path.exists(ckpt_file)

    if rank == 0:
        pprint(cfg)
        print("=> loading checkpoint '{}'".format(ckpt_file))

    """2. Model (Create first to get processor)"""
    # Fix seed
    seed = 0 + rank
    _ = fix_random_seed(seed, include_cuda=True)

    model = make_meta_arch(cfg['model_name'], **cfg['model'])

    # Extract processor before wrapping
    processor = getattr(model, 'processor', None)

    # Move to specific GPU
    device = torch.device(f"cuda:{gpu_id}")
    model = model.to(device)

    # Load Weights and map to the specific GPU to avoid OOM on GPU 0
    checkpoint = torch.load(ckpt_file, map_location=f"cuda:{gpu_id}")
    state_dict = checkpoint['state_dict']

    # EMA Logic
    if cfg['test_cfg'].get('use_ema', False) and 'state_dict_ema' in checkpoint:
        if rank == 0: print("Loading from EMA model ...")
        state_dict = checkpoint['state_dict_ema']

    # Fix DDP key mismatch: strip 'module.' prefix if needed
    """ckpt_keys = list(state_dict.keys())
    model_keys = list(model.state_dict().keys())

    if ckpt_keys[0].startswith('module.') and not model_keys[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}"""

    # Load State Dict
    load_result = model.load_state_dict(state_dict, strict=False)

    # Run Safety Check (On the result of the load)
    critical_layers = ['cls_head', 'lora']
    missing_critical_keys = []
    for missing in load_result.missing_keys:
        for crit in critical_layers:
            if crit in missing:
                missing_critical_keys.append(missing)

    if len(missing_critical_keys) > 0:
        print("\n" + "!" * 60)
        print("FATAL ERROR: Critical weights were NOT loaded!")
        print(f"Example missing keys: {missing_critical_keys[:5]}")

        # Check if the *current* state_dict (Main or EMA) has the prefix
        if any('module.' in k for k in state_dict.keys()):
            raise RuntimeError("Checkpoint (or EMA) contains 'module.' prefix but model does not.")

        raise RuntimeError(f"Model is missing critical weights: {missing_critical_keys[0]}...")

    print(f"-> Model loaded successfully.")

    # Wrap DDP
    if is_distributed:
        model = DDP(model, device_ids=[gpu_id], output_device=gpu_id)

    """3. Dataset / Loader"""
    test_dataset = make_dataset(
        cfg['dataset_name'], False, False, cfg['test_split'], **cfg['dataset']
    )

    # DDP Sampler
    sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank,
                                 shuffle=False) if is_distributed else None

    # Persistent workers must be false for a DDP setup
    loader_cfg = cfg['loader'].copy()
    if 'persistent_workers' in loader_cfg:
        del loader_cfg['persistent_workers']

    test_loader = make_data_loader(
        test_dataset,
        is_training=False,
        is_validation=False,
        generator=None,
        processor=processor,
        sampler=sampler,
        persistent_workers=False,
        **loader_cfg
    )

    """4. Run Inference"""
    if rank == 0: print("\nStart testing model {:s} ...".format(cfg['model_name']))
    start = time.time()

    test_func = get_test_func(cfg['model_name'])

    # Define output file path (Used for pickle AND text metrics)
    output_file = os.path.join(os.path.dirname(ckpt_file), 'eval_results.pkl')

    # Run loop
    local_results = test_func(
        test_loader=test_loader,
        model=model,
        curr_epoch=-1,
        args=args,
        rank=rank,
        generate_text=args.generate_text,
    )

    """5. Merge & Evaluate"""
    final_results = local_results

    if is_distributed:
        # Create temp folder
        tmp_folder = os.path.join(os.path.dirname(ckpt_file), "tmp_ddp_results")
        if rank == 0: os.makedirs(tmp_folder, exist_ok=True)
        dist.barrier(device_ids=[rank])

        final_results = merge_results_dist(local_results, tmp_folder, rank, world_size)

        if rank == 0:
            import shutil
            if os.path.exists(tmp_folder):
                shutil.rmtree(tmp_folder)

    # Only Rank 0 runs final evaluation
    if rank == 0:
        end = time.time()
        print("Inference finished. Total time: {:0.2f} sec".format(end - start))

        # Save merged results
        with open(output_file, "wb") as f:
            pickle.dump(final_results, f)
        print(f"Results saved to {output_file}")

        # Run Evaluator
        if not args.save_only:
            print("Running Evaluator on merged results...")
            print("Extension name: ", prefix)
            det_eval = make_evaluator(
                cfg['model_name'], cfg['dataset']['json_file'], test_dataset.split[0], args.generate_text,
                **cfg['evaluator']
            )

            # Save metrics to .txt file alongside the results .pkl
            metrics_file = output_file.replace('.pkl', '.txt')
            det_eval.evaluate(final_results, verbose=True, metrics_file=metrics_file)

            # Delete the object to release references
            del det_eval

            # Explicitly shut down the background worker processes
            print("Shutting down joblib workers...")
            get_reusable_executor().shutdown(wait=True)

        # Clip saving logic (for ActionFormer) ---
        if args.save_clips:
            clips_cfg = cfg['clips_out']
            if clips_cfg:
                # Get save folder
                folder_name = get_clips_folder_name(clips_cfg['clip_len'], clips_cfg['fps_out'],
                                                    clips_cfg['resize_hw'], clips_cfg['bg_ratio'],
                                                    clips_cfg['bg_min_dist'], curr_epoch=used_epoch)
                save_folder = os.path.join(clips_cfg['clips_pred_root'], folder_name)
                os.makedirs(save_folder, exist_ok=True)

                # Call the Saver
                clip_saver_callback(final_results, save_folder, cfg['dataset']['egoexo_type'], clips_cfg)

    # --- CLEANUP ---
    # 1. Kill Dataloader
    del test_loader

    # 2. Force Garbage Collection (Cleans up deleted objects)
    gc.collect()

    if is_distributed:
        print(f"[Rank {rank}] Waiting for exit sync...")
        dist.barrier(device_ids=[gpu_id])
        cleanup_ddp()

    print(f"[Rank {rank}] Exiting clean.")
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, metavar='DIR', help='path to a config file')
    parser.add_argument('ckpt', type=str, metavar='DIR', help='path to a checkpoint or folder')
    parser.add_argument('-epoch', type=int, default=-1, help='checkpoint epoch')
    parser.add_argument('-t', '--topk', default=-1, type=int, help='max number of output actions')
    parser.add_argument('-ube', '--use_best_epoch', default=False, type=lambda x: x.lower() == "true")
    parser.add_argument('-sc', '--save_clips', default=False, type=lambda x: x.lower() == "true")
    parser.add_argument('-gt', '--generate_text', default=False, type=lambda x: x.lower() == "true",
                        help='If VLM should generate text outputs during inference')
    parser.add_argument('--save_only', action='store_true', help='Only save outputs without evaluation')
    parser.add_argument('-p', '--print-freq', default=10, type=int, help='print frequency')
    args = parser.parse_args()

    # Load Config
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")

    # Detect Devices
    devices = cfg['devices']
    world_size = len(devices)

    if world_size == 1:
        # --- SINGLE GPU MODE ---
        target_gpu = int(devices[0].split(':')[-1])  # Extract '4' from 'cuda:4'
        print(f"-> Single GPU Mode: Physical GPU {target_gpu}")

        # Set default CUDA device early to prevent context creation on GPU:0
        torch.cuda.set_device(target_gpu)

        # Pass physical ID directly
        main_worker(0, 1, args, cfg, target_gpu=target_gpu)
    else:
        # --- MULTI GPU DDP MODE ---
        print(f"-> Multi-GPU Mode: {devices}")
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(random.randint(29500, 29999))

        # Map logical 0...N to physical IDs via CUDA_VISIBLE_DEVICES
        gpu_indices = ",".join([d.split(':')[-1] for d in devices])
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_indices

        mp.spawn(
            main_worker,
            args=(world_size, args, cfg, None),
            nprocs=world_size,
            join=True
        )
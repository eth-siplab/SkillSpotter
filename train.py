# os imports
import os
os.environ["NCCL_P2P_DISABLE"] = "1"  # Disable NCCL P2P for better stability. No big impact when only fine-tuning.
os.environ["NCCL_IB_DISABLE"] = "1" # Optional: Disables InfiniBand if you have it, forces local PCIe
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

# python imports
import argparse
from pprint import pprint
import builtins
import random
import time

# torch imports
import torch
import torch.nn as nn
import torch.utils.data
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

# our code
from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import (train_one_epoch, valid_one_epoch,
                        save_checkpoint, make_optimizer, make_scheduler,
                        fix_random_seed, ModelEma)


def setup_ddp(rank, world_size):
    """Initialize the distributed environment."""
    # MASTER_ADDR/PORT are set in the main block before spawning
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    dist.destroy_process_group()


def main_worker(rank, world_size, args, cfg, target_gpu=None):
    """
    The actual training logic.
    rank: 0 for single GPU. 0-N for DDP.
    target_gpu: Explicit GPU ID for single mode. If None, uses 'rank' (for DDP).
    """
    # Fix for AttributeError if resume is not used
    args.start_epoch = 0

    is_distributed = world_size > 1

    # 1. Setup Distributed
    if is_distributed:
        setup_ddp(rank, world_size)
        if rank != 0:
            def print_pass(*args, **kwargs): pass

            builtins.print = print_pass

    # 2. Determine Device
    # Single GPU: Use the explicit physical ID (e.g., 4)
    # DDP: Use 'rank' (e.g. 0) which maps to Physical ID via CUDA_VISIBLE_DEVICES
    gpu_id = target_gpu if target_gpu is not None else rank

    # 3. Setup Folders (Only Rank 0)
    ckpt_folder = ""
    if rank == 0:
        if not os.path.exists(cfg['output_folder']):
            os.makedirs(cfg['output_folder'], exist_ok=True)

        cfg_filename = os.path.basename(args.config).replace('.yaml', '')
        ckpt_folder = os.path.join(cfg['output_folder'], cfg_filename)

        prefix = ""
        if cfg['model'].get('model_id', None) is not None:
            prefix = prefix + cfg['model']['model_id'].split('/')[-1]
        if cfg['extension'] != "":
            prefix = prefix + '_' + cfg['extension'] if prefix else cfg['extension']
        if prefix == "": prefix = "noext"

        ckpt_folder = os.path.join(ckpt_folder, prefix)
        os.makedirs(ckpt_folder, exist_ok=True)

        if 'egoexo_type' in cfg['dataset']:
            ckpt_folder = os.path.join(ckpt_folder, cfg['dataset']['egoexo_type'])
        os.makedirs(ckpt_folder, exist_ok=True)

        print("Config:")
        pprint(cfg)

        with open(os.path.join(ckpt_folder, 'config.txt'), 'w') as fid:
            pprint(cfg, stream=fid)
            fid.flush()

    # Tensorboard (Only Rank 0)
    # tb_writer = SummaryWriter(os.path.join(ckpt_folder, 'logs')) if rank == 0 else None
    tb_writer = None
    if rank == 0:
        # Create a unique timestamp: e.g., "Dec08_14-30-00"
        run_name = time.strftime("%b%d_%H-%M-%S")

        # Log to: output/model_name/logs/Dec08_14-30-00/
        log_dir = os.path.join(ckpt_folder, 'logs', run_name)

        tb_writer = SummaryWriter(log_dir)
        print(f"TensorBoard logging to: {log_dir}")

    # 4. Fix Seeds
    seed = cfg['init_rand_seed'] + rank
    rng_generator = fix_random_seed(seed, include_cuda=True)

    # Scale learning rate by world size
    # cfg['opt']["learning_rate"] *= world_size

    """5. Create Datasets"""
    train_dataset = make_dataset(
        cfg['dataset_name'], True, False, cfg['train_split'], **cfg['dataset']
    )
    val_dataset = make_dataset(
        cfg['dataset_name'], False, True, cfg['val_split'], **cfg['dataset']
    )

    if rank == 0:
        train_db_vars = train_dataset.get_attributes()
        cfg['model']['train_cfg']['head_empty_cls'] = train_db_vars['empty_label_ids']

    """6. Create Model"""
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters of model: {total_params}")

    # Extract processor BEFORE wrapping
    processor = getattr(model, 'processor', None)

    # Move to GPU using the calculated gpu_id
    device = torch.device(f"cuda:{gpu_id}")
    model = model.to(device)

    # DDP Wrapping
    if is_distributed:
        if cfg.get('sync_bn', False):
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # Using gpu_id here ensures local_rank matches
        model = DDP(model, device_ids=[gpu_id], output_device=gpu_id, find_unused_parameters=True)

    optimizer = make_optimizer(model, cfg['opt'])

    """7. Create Dataloaders"""
    # DDP Samplers
    if is_distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = make_data_loader(
        train_dataset,
        is_training=True,
        is_validation=False,
        generator=rng_generator,
        processor=processor,
        sampler=train_sampler,
        **cfg['loader']
    )

    val_loader = make_data_loader(
        val_dataset,
        is_training=False,
        is_validation=True,
        generator=None,
        processor=processor,
        sampler=val_sampler,
        **cfg['loader']
    )

    scheduler = make_scheduler(optimizer, cfg['opt'], len(train_loader))

    # EMA
    if cfg['train_cfg'].get('use_ema', False):
        if rank == 0: print("Using model EMA ...")
        model_to_track = model.module if is_distributed else model
        model_ema = ModelEma(model_to_track)
    else:
        model_ema = None

    """8. Resume"""
    if args.resume:
        if os.path.isfile(args.resume):
            # Map to the correct device to avoid OOM
            checkpoint = torch.load(args.resume, map_location=f'cuda:{gpu_id}')
            args.start_epoch = checkpoint['epoch']
            model_to_load = model.module if hasattr(model, 'module') else model
            model_to_load.load_state_dict(checkpoint['state_dict'], strict=False)

            if model_ema is not None and 'state_dict_ema' in checkpoint:
                model_ema.module.load_state_dict(checkpoint['state_dict_ema'])

            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            if rank == 0:
                print(f"=> Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
            del checkpoint
        else:
            if rank == 0: print(f"=> Checkpoint '{args.resume}' not found.")
            return

    """9. Training Loop"""
    best_val_loss = float('inf')

    # Early stopping setup
    patience_counter = 0
    patience_limit = cfg['opt']['patience'] # Default 10 if not in config
    use_early_stopping = cfg['opt']['enable_early_stopping']

    if rank == 0:
        print(f"\nStart training model {cfg['model_name']} ...")

    max_epochs = cfg['opt'].get('early_stop_epochs', cfg['opt']['epochs'] + cfg['opt']['warmup_epochs'])

    start = time.time()
    for epoch in range(args.start_epoch, max_epochs):
        if is_distributed:
            train_sampler.set_epoch(epoch)

        train_one_epoch(
            train_loader, model, optimizer, scheduler, epoch,
            model_ema=model_ema,
            clip_grad_l2norm=cfg['train_cfg']['clip_grad_l2norm'],
            tb_writer=tb_writer,
            print_freq=args.print_freq,
            rank=rank,
            accumulation_steps=cfg['opt'].get('accumulation_steps', 1),
        )

        val_loss = valid_one_epoch(
            val_loader, model, epoch, cfg['model_name'],
            tb_writer=tb_writer,
            rank=rank,
            world_size=world_size
        )

        # This ensures patience_counter is synced across all GPUs
        is_best = val_loss < best_val_loss

        if is_best:
            best_val_loss = val_loss
            patience_counter = 0  # Reset
        else:
            patience_counter += 1  # Increment

        if rank == 0:
            if is_best:
                print(f"★ New Best Model! Val Loss: {best_val_loss:.4f}")

            if (((epoch + 1) == max_epochs) or
                    ((args.ckpt_freq > 0) and ((epoch + 1) % args.ckpt_freq == 0)) or
                    is_best):

                # Standardize: Always save the raw model, not the DDP wrapper
                model_to_save = model.module if hasattr(model, 'module') else model
                save_states = {
                    'epoch': epoch + 1,
                    'state_dict': model_to_save.state_dict(),  # <--- CLEAN SAVE
                    'scheduler': scheduler.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                }

                if model_ema is not None:
                    save_states['state_dict_ema'] = model_ema.module.state_dict()

                save_checkpoint(
                    save_states, is_best,
                    file_folder=ckpt_folder,
                    file_name=f'epoch_{epoch + 1:03d}.pth.tar'
                )

        # Early stopping check
        if use_early_stopping and patience_counter >= patience_limit:
            if rank == 0:
                print(f"\n[Early Stopping] Validation loss hasn't improved for {patience_limit} epochs.")
                print(f"Stopping training at Epoch {epoch + 1}.")
            break

    if rank == 0:
        if tb_writer: tb_writer.close()
        end = time.time()
        print("Training finished. Total time: {:0.2f} sec".format(end - start))
        print("All done!")

    # --- CLEANUP ---
    # 1. Kill Dataloaders (Fixes "leaked semaphore" warnings)
    if 'train_loader' in locals(): del train_loader
    if 'val_loader' in locals(): del val_loader

    # 2. Sync before exit (Safety)
    if is_distributed:
        dist.barrier(device_ids=[rank])
        cleanup_ddp()

    if rank == 0:
        if tb_writer: tb_writer.close()
        print("All done!")

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', metavar='DIR', help='path to a config file')
    parser.add_argument('-p', '--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('-c', '--ckpt-freq', default=5, type=int, help='checkpoint frequency')
    # parser.add_argument('--numa', type=int, default=-1, help='Force execution on NUMA node (0 or 1)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to a checkpoint (not best)')
    args = parser.parse_args()

    # 1. Load Config
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")

    # 2. Detect Devices
    devices = cfg['devices']
    world_size = len(devices)

    if world_size == 1:
        # --- SINGLE GPU MODE ---
        target_gpu = int(devices[0].split(':')[-1])  # Extract '4' from 'cuda:4'
        print(f"-> Single GPU Mode: Physical GPU {target_gpu}")

        # Set default CUDA device early to prevent context creation on GPU:0
        torch.cuda.set_device(target_gpu)

        # We do NOT set CUDA_VISIBLE_DEVICES here.
        # We pass the physical ID directly to the worker.
        main_worker(0, 1, args, cfg, target_gpu=target_gpu)

    else:
        # --- MULTI GPU DDP MODE ---
        print(f"-> Multi-GPU Mode: {devices}")
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(random.randint(29500, 29999))

        # For DDP, we DO use masking to map logical 0..N to physical X..Y
        gpu_indices = ",".join([d.split(':')[-1] for d in devices])
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_indices

        # spawn passes 'rank' as the first argument automatically
        # 'None' for target_gpu means "Use Rank"
        mp.spawn(
            main_worker,
            args=(world_size, args, cfg, None),
            nprocs=world_size,
            join=True
        )
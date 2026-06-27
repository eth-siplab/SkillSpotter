import os
import time
import numpy as np
import random
from copy import deepcopy
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist  # Needed for syncing

from .lr_schedulers import LinearWarmupMultiStepLR, LinearWarmupCosineAnnealingLR


################################################################################
def fix_random_seed(seed, include_cuda=True):
    rng_generator = torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if include_cuda:
        cudnn.enabled = True
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        cudnn.enabled = True
        cudnn.benchmark = True
    return rng_generator


def save_checkpoint(state, is_best, file_folder, file_name='checkpoint.pth.tar'):
    """save checkpoint to file"""
    if not os.path.exists(file_folder):
        os.mkdir(file_folder)
    torch.save(state, os.path.join(file_folder, file_name))
    if is_best:
        state.pop('optimizer', None)
        state.pop('scheduler', None)
        torch.save(state, os.path.join(file_folder, 'model_best.pth.tar'))


def make_optimizer(model, optimizer_config):
    decay = set()
    no_decay = set()
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}

    for pn, p in param_dict.items():
        if pn.endswith('bias') or p.ndim < 2:
            no_decay.add(pn)
        elif pn.endswith('rel_pe') or pn.endswith('pos_embed') or pn.endswith('scale'):
            no_decay.add(pn)
        elif pn.endswith('weight') and p.ndim >= 2:
            decay.add(pn)
        else:
            no_decay.add(pn)

    # validation
    union_params = decay | no_decay
    missing = param_dict.keys() - union_params
    assert len(missing) == 0, f"Parameters {missing} were missed!"

    # Print stats to verify LoRA is being treated correctly
    print(f"[Optimizer] Decay params: {len(decay)} (e.g., LoRA A/B, Proj Weights)")
    print(f"[Optimizer] No Decay params: {len(no_decay)} (e.g., Biases, Logit Scale)")

    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": optimizer_config['weight_decay']},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]

    if optimizer_config["type"] == "SGD":
        optimizer = optim.SGD(optim_groups, lr=optimizer_config["learning_rate"],
                              momentum=optimizer_config.get("momentum", 0.9))
    elif optimizer_config["type"] == "AdamW":
        optimizer = optim.AdamW(optim_groups, lr=optimizer_config["learning_rate"])
    else:
        raise TypeError("Unsupported optimizer!")
    return optimizer


"""def make_optimizer(model, optimizer_config):
    # 1. Separate Head vs Backbone
    head_params = []
    backbone_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # We target the classification head specifically
        if 'cls_head' in name or 'contrastive_proj' in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    print(f"[Optimizer] Head Params: {len(head_params)} (LR: {optimizer_config['learning_rate_heads']})")
    print(f"[Optimizer] Backbone Params: {len(backbone_params)} (LR: {optimizer_config['learning_rate']})")

    # 2. Assign Different Learning Rates
    optim_groups = [
        {
            "params": backbone_params,
            "lr": optimizer_config["learning_rate"],  # Keep small (e.g. 1e-5) to protect LoRA
            "weight_decay": optimizer_config["weight_decay"]
        },
        {
            "params": head_params,
            "lr": optimizer_config['learning_rate_heads'],  # BOOST 100x: Force the head to learn quickly
            "weight_decay": optimizer_config["weight_decay"]
        },
    ]

    if optimizer_config["type"] == "AdamW":
        optimizer = optim.AdamW(optim_groups)
    else:
        raise TypeError("Unsupported optimizer!")

    return optimizer"""


def make_scheduler(optimizer, optimizer_config, num_iters_per_epoch, last_epoch=-1):
    if optimizer_config["warmup"]:
        max_epochs = optimizer_config["epochs"] + optimizer_config["warmup_epochs"]
        max_steps = max_epochs * num_iters_per_epoch
        warmup_steps = optimizer_config["warmup_epochs"] * num_iters_per_epoch

        if optimizer_config["schedule_type"] == "cosine":
            scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_steps, max_steps, last_epoch=last_epoch)
        elif optimizer_config["schedule_type"] == "multistep":
            steps = [num_iters_per_epoch * step for step in optimizer_config["schedule_steps"]]
            scheduler = LinearWarmupMultiStepLR(optimizer, warmup_steps, steps,
                                                gamma=optimizer_config["schedule_gamma"], last_epoch=last_epoch)
        else:
            raise TypeError("Unsupported scheduler!")
    else:
        max_epochs = optimizer_config["epochs"]
        max_steps = max_epochs * num_iters_per_epoch
        if optimizer_config["schedule_type"] == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, max_steps, last_epoch=last_epoch)
        elif optimizer_config["schedule_type"] == "multistep":
            steps = [num_iters_per_epoch * step for step in optimizer_config["schedule_steps"]]
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, steps, gamma=optimizer_config["schedule_gamma"],
                                                       last_epoch=last_epoch)
        else:
            raise TypeError("Unsupported scheduler!")
    return scheduler


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = 0.0

    def initialize(self, val, n):
        self.val = val
        self.avg = val
        self.sum = val * n
        self.count = n
        self.initialized = True

    def update(self, val, n=1):
        if not self.initialized:
            self.initialize(val, n)
        else:
            self.add(val, n)

    def add(self, val, n):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.999, device=None):
        super().__init__()
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


# --- DDP HELPER ---
def reduce_tensor(tensor, world_size):
    """Reduces a tensor across all GPUs (Sum / World_Size)"""
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


################################################################################
def train_one_epoch(
        train_loader,
        model,
        optimizer,
        scheduler,
        curr_epoch,
        model_ema=None,
        clip_grad_l2norm=-1,
        tb_writer=None,
        print_freq=20,
        rank=0,
        accumulation_steps=1
):
    """Training the model for one epoch with Gradient Accumulation and Memory Optimization"""
    batch_time = AverageMeter()
    losses_tracker = {}
    model.train()

    if rank == 0:
        print(f"\n\n[Train] Epoch {curr_epoch} started")
        iterator = tqdm(enumerate(train_loader),
                        total=len(train_loader),
                        desc=f"[Train] Epoch {curr_epoch}",
                        leave=True,
                        mininterval=2.0,
                        maxinterval=10.0
        )
    else:
        iterator = enumerate(train_loader)

    start = time.time()

    # Ensure gradients are zero before starting the epoch loop
    optimizer.zero_grad(set_to_none=True)

    for iter_idx, video_list in iterator:
        # 1. Forward Pass
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            losses = model(video_list)

            # 2. Scale Loss for Accumulation
            loss = losses['final_loss'] / accumulation_steps

            # 3. Backward Pass (Accumulates Gradients in buffers)
            loss.backward()

        # Track "True" losses for logging (multiply back or just use raw dict values)
        # We track the raw values directly from the model output
        for key, value in losses.items():
            if key not in losses_tracker:
                losses_tracker[key] = AverageMeter()
            losses_tracker[key].update(value.item())

        # 4. Step Optimizer (Only every N steps)
        if (iter_idx + 1) % accumulation_steps == 0:

            # Gradient Clipping
            if clip_grad_l2norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_l2norm)

            # Update Weights
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            # EMA Update
            if model_ema is not None:
                model_ema.update(model)

        # Update timing
        batch_time.update(time.time() - start)
        start = time.time()

        # Logging (Rank 0 only)
        if rank == 0 and iter_idx % print_freq == 0:
            current_lr = scheduler.get_last_lr()[0]

            # Update tqdm string
            if isinstance(iterator, tqdm):
                iterator.set_postfix({
                    'loss': f"{losses_tracker['final_loss'].avg:.4f}",
                    'cls': f"{losses_tracker['cls_loss'].avg:.4f}" if 'cls_loss' in losses_tracker else "-",
                    'lr': f"{current_lr:.6f}"
                })

            # TensorBoard
            if tb_writer is not None:
                global_step = curr_epoch * len(train_loader) + iter_idx
                tb_writer.add_scalar('train/learning_rate', current_lr, global_step)
                tag_dict = {k: v.val for k, v in losses_tracker.items() if k != "final_loss"}
                tb_writer.add_scalars('train/all_losses', tag_dict, global_step)
                tb_writer.add_scalar('train/final_loss', losses_tracker['final_loss'].val, global_step)

    if rank == 0:
        lr = scheduler.get_last_lr()[0]
        print("[Train]: Epoch {:d} finished with lr={:.8f}".format(curr_epoch, lr))


def valid_one_epoch(
        val_loader,
        model,
        curr_epoch,
        model_name,
        tb_writer=None,
        print_freq=20,
        rank=0,
        world_size=1
):
    """
    Validate the model.
    Aggregates loss across all GPUs if world_size > 1.
    """
    batch_time = AverageMeter()
    losses_tracker = {}
    model.eval()

    start = time.time()
    if rank == 0:
        iterator = tqdm(
            val_loader,
            desc=f"Val Ep {curr_epoch}",
            mininterval=3.0
        )
    else:
        iterator = val_loader

    for iter_idx, video_list in enumerate(iterator):
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # Handle model wrappers
                if model_name in ['qwen3_hybrid', 'qwen3_hybrid_omnivore', 'qwen3_hybrid_contrastive']:
                    output = model(video_list, return_loss=True, generate_text=False)
                else:
                    output = model(video_list, return_loss=True)

                loss_dict = output['losses'] if (isinstance(output, dict) and 'losses' in output) else output

        # Track Local Losses
        for key, value in loss_dict.items():
            if key not in losses_tracker:
                losses_tracker[key] = AverageMeter()
            losses_tracker[key].update(value.item())

        if rank == 0:
            torch.cuda.synchronize()
            batch_time.update((time.time() - start) / print_freq)
            start = time.time()  # Reset timer

            # We grab the current average of the final loss for the status bar
            f_loss = losses_tracker['final_loss'].avg if 'final_loss' in losses_tracker else 0.0
            iterator.set_postfix(loss=f"{f_loss:.4f}", time=f"{batch_time.val:.2f}")

    # --- SYNCHRONIZATION (The Critical Part for DDP) ---
    final_loss_val = 0.0

    # We must iterate keys to reduce them
    # Note: If a key exists on one GPU but not another (rare), this might hang.
    # Assuming consistent loss dict structure.
    for key in sorted(list(losses_tracker.keys())):
        meter = losses_tracker[key]
        local_avg = torch.tensor(meter.avg, device='cuda')

        if world_size > 1:
            dist.all_reduce(local_avg, op=dist.ReduceOp.SUM)
            global_avg = local_avg / world_size
        else:
            global_avg = local_avg

        # Update tracker with global average for logging
        meter.avg = global_avg.item()

        if key == 'final_loss':
            final_loss_val = meter.avg

    # Log global averages
    if rank == 0:
        tag_dict = {k: v.avg for k, v in losses_tracker.items()}
        if tb_writer is not None:
            tb_writer.add_scalars('validation/all_losses', tag_dict, curr_epoch)
            tb_writer.add_scalar('validation/final_loss', final_loss_val, curr_epoch)

        print(f'All average validation losses (Global):')
        for k, v in tag_dict.items():
            print(f'  - {k}: {v:.4f}')

    return final_loss_val
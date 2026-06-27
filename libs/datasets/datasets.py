import torch

from .data_utils import trivial_batch_collator, worker_init_reset_seed
from .clips_collate import clips_collate_fn

datasets = {}
def register_dataset(name):
   def decorator(cls):
       datasets[name] = cls
       return cls
   return decorator

def make_dataset(name, is_training, is_validation, split, **kwargs):
   """
       A simple dataset builder
   """
   dataset = datasets[name](is_training, is_validation, split, **kwargs)
   return dataset

def make_data_loader(dataset,
                     is_training,
                     generator,
                     processor,
                     batch_size,
                     num_workers,
                     pin_memory=False,
                     prefetch_factor=2,
                     persistent_workers=False,
                     collate_fn_name=None,
                     fps_out=None,
                     is_validation=False,
                     sampler=None
                     ):
    """
         A simple dataloader builder
    """

    # 1. Create the Collator Object using the Factory
    # If collate_mode_training: Answer is appended to the text so that we can calculate the LM loss during valid
    # However: classification head does not see answer (see forward implementation) to avoid data leakage
    collate_mode_training = is_training or is_validation  # Ensure text is also loaded for validation
    actual_collate_fn = make_collate_fn(
        collate_fn_name=collate_fn_name,
        is_training=collate_mode_training,
        fps_out=fps_out,
        processor=processor
    )

    # batch_size 1 for test set
    if not is_training and not is_validation:
        batch_size = 1

    # batch_size 1 for validation set
    if is_validation:
        batch_size = 1

    # 2. HANDLE SHUFFLE LOGIC
    # If a sampler is provided (e.g. DistributedSampler), shuffle must be False.
    if sampler is not None:
        loader_shuffle = False
    else:
        loader_shuffle = is_training

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=actual_collate_fn,
        worker_init_fn=(worker_init_reset_seed if is_training else None),
        shuffle=loader_shuffle,
        sampler=sampler,
        drop_last=is_training,
        generator=generator,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )
    return loader


def make_collate_fn(collate_fn_name, is_training, fps_out, processor):
    """
    Factory function to create the specific collator object.
    """

    if collate_fn_name == 'clips_collate':
        return clips_collate_fn
    elif collate_fn_name is None:
        return trivial_batch_collator
    else:
        raise ValueError(f"Collate function {collate_fn_name} not recognized!")
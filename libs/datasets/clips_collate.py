import torch

def clips_collate_fn(batch):
    """Custom collate to handle mixed tensor/string/bool data."""
    return {
        'video': torch.stack([item['video'] for item in batch]),
        'label': torch.tensor([item['label'] for item in batch]),
        'good_text': [item['good_text'] for item in batch],  # List of strings
        'bad_text': [item['bad_text'] for item in batch],    # List of strings
        'is_background': torch.tensor([item['is_background'] for item in batch])
    }
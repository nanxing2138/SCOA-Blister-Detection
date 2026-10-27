"""Trusted local checkpoint loading across the pinned PyTorch versions."""
import inspect
import torch


def load_checkpoint(path, map_location='cpu'):
    # Training checkpoints contain optimizer and Python/NumPy RNG state as well as tensors.
    # The path must be trusted; do not use this helper for arbitrary downloaded pickles.
    kwargs = {'map_location': map_location}
    if 'weights_only' in inspect.signature(torch.load).parameters:
        kwargs['weights_only'] = False
    return torch.load(path, **kwargs)

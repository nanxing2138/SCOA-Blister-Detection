import hashlib
import numpy as np


def stream(seed, epoch, image_id, purpose):
    """Process-independent streams; policy is deliberately absent from the pool key."""
    key = f'{seed}|{epoch}|{image_id}|{purpose}'.encode('utf-8')
    value = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), 'little')
    return np.random.default_rng(value)

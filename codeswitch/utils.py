"""Device selection, seeding, memory monitoring, class weights."""
import gc
import os
import random

import numpy as np
import torch

from . import config as C

try:
    import psutil
except ImportError:
    psutil = None  # pip install psutil


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()
# Abort training when MPS allocation passes this, instead of freezing the machine
MEM_LIMIT_GB = (torch.mps.recommended_max_memory() / 1e9 * 0.9) if DEVICE.type == "mps" else 22


def set_seed(seed=C.SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def _gb(x):
    return x / 1e9


def mem(tag=""):
    s = f"[mem {tag}]"
    if DEVICE.type == "mps":
        s += f" MPS alloc {_gb(torch.mps.current_allocated_memory()):.1f} GB | driver {_gb(torch.mps.driver_allocated_memory()):.1f} GB"
    elif DEVICE.type == "cuda":
        s += f" CUDA alloc {_gb(torch.cuda.memory_allocated()):.1f} GB"
    if psutil:
        s += f" | RSS {_gb(psutil.Process(os.getpid()).memory_info().rss):.1f} GB"
    print(s)


def check_mem():
    if DEVICE.type == "mps" and _gb(torch.mps.current_allocated_memory()) > MEM_LIMIT_GB:
        raise RuntimeError(f"MPS allocated > {MEM_LIMIT_GB:.1f} GB: call free() or lower the batch size")


def free():
    gc.collect()
    if DEVICE.type == "mps":
        torch.mps.empty_cache()
    elif DEVICE.type == "cuda":
        torch.cuda.empty_cache()


def sqrt_balanced_weights(labels):
    """Square-root-damped balanced class weights for a binary label array."""
    labels = np.asarray(labels)
    counts = np.bincount(labels, minlength=2).clip(min=1)
    return (torch.tensor(len(labels) / (2 * counts), dtype=torch.float)) ** 0.5

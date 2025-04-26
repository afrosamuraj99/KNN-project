"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf
import torch as th


def setup_dist():
    """
    Setup a distributed process group.
    """
    pass


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device("cuda")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    return th.load(str(path), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    pass

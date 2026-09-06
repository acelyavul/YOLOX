#!/usr/bin/env python3
"""Deterministic execution helpers for YOLOX training."""

import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn


_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
_SUPPORTED_CUBLAS_WORKSPACE_CONFIGS = {":4096:8", ":16:8"}


def configure_determinism(seed: int) -> None:
    """Seed all training RNGs and reject known nondeterministic algorithms."""
    workspace_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if (
        workspace_config is not None
        and workspace_config not in _SUPPORTED_CUBLAS_WORKSPACE_CONFIGS
    ):
        raise RuntimeError(
            "CUBLAS_WORKSPACE_CONFIG must be ':4096:8' or ':16:8' "
            "for deterministic CUDA execution."
        )
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", _CUBLAS_WORKSPACE_CONFIG)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

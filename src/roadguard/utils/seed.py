"""Reproducibility and CUDA performance settings."""

import random


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, PyTorch and all visible CUDA devices."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_cuda(*, deterministic: bool, benchmark: bool, tf32: bool) -> None:
    """Apply feature-gated precision and fixed-shape cuDNN settings."""
    import torch

    torch.backends.cudnn.benchmark = benchmark and not deterministic
    if torch.cuda.get_device_capability(0)[0] < 8:
        return
    precision = "tf32" if tf32 else "ieee"
    if hasattr(torch.backends, "fp32_precision"):
        torch.backends.cuda.matmul.fp32_precision = precision
        torch.backends.cudnn.conv.fp32_precision = precision
    else:
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32

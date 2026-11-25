"""CUDA inspection and fail-fast checks for detection runs."""

import os
import platform
from importlib.metadata import PackageNotFoundError, version
from typing import Any


def environment_info() -> dict[str, Any]:
    """Return library versions, CUDA runtime, and visible device properties."""
    details: dict[str, Any] = {"python": platform.python_version()}
    for package in ("torch", "torchvision", "ultralytics"):
        try:
            details[package] = version(package)
        except PackageNotFoundError:
            details[package] = None
    details.update(cuda_available=False, cuda_runtime=None, gpu=None, device_properties=None)
    try:
        import torch
    except (ImportError, OSError) as error:
        details["torch_error"] = str(error)
        return details

    details["torch"] = torch.__version__
    details["cuda_runtime"] = torch.version.cuda
    details["cuda_available"] = torch.cuda.is_available()
    if details["cuda_available"]:
        properties = torch.cuda.get_device_properties(0)
        details["gpu"] = torch.cuda.get_device_name(0)
        details["device_properties"] = {
            "compute_capability": [properties.major, properties.minor],
            "total_memory_bytes": properties.total_memory,
            "multiprocessors": properties.multi_processor_count,
        }
    return details


def require_cuda(device: str = "cuda", *, deterministic: bool = True) -> Any:
    """Return CUDA device 0 after a real tensor operation and a CUDA NMS check."""
    if device != "cuda":
        raise ValueError("RoadGuard detection runs require device=cuda.")
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    details = environment_info()
    if not details["cuda_available"]:
        raise RuntimeError(
            f"CUDA is required (PyTorch={details['torch']}, CUDA={details['cuda_runtime']}). "
            "Activate a CUDA-enabled PyTorch and TorchVision environment and run "
            "scripts/check_environment.py."
        )

    import torch

    device = torch.device("cuda")
    try:
        from torchvision.ops import nms

        boxes = torch.tensor([[0, 0, 1, 1]], dtype=torch.float32, device=device)
        scores = torch.ones(1, device=device)
        nms(boxes, scores, 0.5)
        torch.cuda.synchronize(device)
    except (ImportError, AttributeError, OSError, RuntimeError, NotImplementedError) as error:
        raise RuntimeError(
            f"CUDA execution or TorchVision NMS failed: {error}. "
            "Install matching CUDA builds of PyTorch and TorchVision."
        ) from error
    return device

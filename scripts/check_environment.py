"""Check package versions and verify that CUDA execution works."""

import json

from roadguard.cli import run
from roadguard.utils.device import environment_info, require_cuda


def main() -> None:
    """Print GPU and runtime details, then require working CUDA operations."""
    print(json.dumps(environment_info(), indent=2), flush=True)
    require_cuda()
    print("CUDA execution and TorchVision CUDA NMS passed.")


if __name__ == "__main__":
    run(main)

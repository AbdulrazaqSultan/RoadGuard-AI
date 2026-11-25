"""Check configuration, CUDA fail-fast behavior and independent run outputs."""

import pytest

from roadguard.config import load_config
from roadguard.training.experiment import reserve_run
from roadguard.utils import device


def test_profile_and_cli_override_order(tmp_path):
    profile = tmp_path / "profile.yaml"
    profile.write_text("training:\n  batch_size: 4\n", encoding="utf-8")
    config = load_config(profile=profile, overrides=["training.batch_size=2"])
    assert config["training"]["batch_size"] == 2
    assert config["training"]["seed"] == 42
    with pytest.raises(ValueError, match="Unknown setting"):
        load_config(overrides=["training.worker=2"])
    with pytest.raises(ValueError, match="Invalid value"):
        load_config(overrides=["training.amp=automatic"])


def test_cpu_fallback_is_rejected(monkeypatch):
    monkeypatch.setattr(
        device,
        "environment_info",
        lambda: {
            "cuda_available": False,
            "torch": "2.8.0+cpu",
            "cuda_runtime": None,
        },
    )
    with pytest.raises(RuntimeError, match="CUDA is required"):
        device.require_cuda(deterministic=False)
    with pytest.raises(ValueError, match="require device=cuda"):
        device.require_cuda("cpu")


def test_attempts_do_not_overwrite_prior_results(tmp_path):
    identity = {"seed": 42, "model": "yolo11n"}
    first_id, first = reserve_run(tmp_path, "baseline", identity)
    (first / "results.csv").write_text("test fixture", encoding="utf-8")
    second_id, second = reserve_run(tmp_path, "baseline", identity)
    assert first_id == second_id
    assert first != second
    assert (first / "results.csv").read_text(encoding="utf-8") == "test fixture"

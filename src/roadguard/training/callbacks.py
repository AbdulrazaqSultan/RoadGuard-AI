"""Local CUDA configuration and checkpoint progress logging."""

import logging
from functools import partial
from typing import Any

from roadguard.utils.paths import write_json
from roadguard.utils.seed import configure_cuda

LOGGER = logging.getLogger(__name__)


def _prepare_cuda(trainer: Any, *, benchmark: bool, tf32: bool) -> None:
    if trainer.device.type != "cuda":
        raise RuntimeError("The detection trainer did not select CUDA.")
    configure_cuda(deterministic=trainer.args.deterministic, benchmark=benchmark, tf32=tf32)


def _checkpoint(trainer: Any) -> None:
    LOGGER.info(
        "Epoch %d checkpoint saved. Validation mAP50-95: %.6f",
        trainer.epoch + 1,
        trainer.fitness,
    )
    if trainer.fitness == trainer.best_fitness and trainer.best.is_file():
        write_json(
            trainer.save_dir / "best_checkpoint.json",
            {
                "epoch": trainer.epoch + 1,
                "checkpoint": trainer.best.relative_to(trainer.save_dir).as_posix(),
                "selection_metric": "validation mAP50-95",
                "score_at_selection": float(trainer.fitness),
            },
        )


def attach_callbacks(model: Any, settings: dict) -> None:
    """Configure CUDA after backend seed setup and log checkpoint events."""
    model.add_callback(
        "on_pretrain_routine_start",
        partial(_prepare_cuda, benchmark=settings["cudnn_benchmark"], tf32=settings["tf32"]),
    )
    model.add_callback("on_model_save", _checkpoint)

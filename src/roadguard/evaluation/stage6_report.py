"""Build the final report from the single archived test capture."""

import csv
from collections import Counter
from pathlib import Path

from roadguard.data.dataset import SPLITS
from roadguard.data.validation import sha256
from roadguard.evaluation.stage5 import CODE, verify_protected_inputs
from roadguard.evaluation.stage5_errors import (
    BACKGROUND,
    confusion_counts,
    size_breakdown,
    size_bucket,
)
from roadguard.evaluation.stage5_figures import (
    draw_gallery,
    plot_diagnostics,
    select_representative,
)
from roadguard.evaluation.stage5_review import (
    class_metrics,
    confusion_tables,
    read_capture,
    write_csv,
)
from roadguard.evaluation.stage6 import CAPTURE, EXPERIMENT, METRICS, read_json, split_checksums
from roadguard.training.screening import verify_descriptor, verify_inventory
from roadguard.utils.paths import write_json, write_text
from roadguard.utils.quality import run_quality_gates

NOTEBOOK = Path("notebooks/06_final_test_evaluation.ipynb")
DOCUMENT = Path("docs/stage6_final_test_evaluation_ar.md")
SCORES = {"map50_95": "mAP50–95", "map50": "mAP50", "precision": "Precision", "recall": "Recall"}


def _comparison(native: dict, classes: list[dict], sizes: list[dict], freeze: dict) -> dict:
    reference = freeze["stage4_reference"]
    rows = []
    for key, label in SCORES.items():
        gap = native[key] - reference[key]
        rows.append(
            {
                "metric": label,
                "validation": reference[key],
                "test": native[key],
                "difference": gap,
                "absolute_magnitude": abs(gap),
                "percentage_point_difference": 100 * gap,
            }
        )
    classes_reference = freeze["stage4_full_validation_metrics"]["per_class"]
    per_class = [
        {
            "class": row["class"],
            "validation_ap50_95": classes_reference[row["name"]]["ap50_95"],
            "test_ap50_95": row["ap50_95"],
            "difference": row["ap50_95"] - classes_reference[row["name"]]["ap50_95"],
        }
        for row in classes
    ]
    validation_sizes = {
        row["size"]: row["false_negative_rate"]
        for row in freeze["stage5_validation_size_performance"]
        if row["class"] == "All"
    }
    test_sizes = {row["size"]: row["false_negative_rate"] for row in sizes if row["class"] == "All"}
    gap = native["map50_95"] - reference["map50_95"]
    return {
        "global_metrics": rows,
        "per_class_ap50_95": per_class,
        "global_reference": "Stage 4 published four-decimal validation metrics",
        "per_class_reference": "Stage 4 archived full-precision validation AP",
        "generalization_gap": gap,
        "generalization_gap_percentage_points": 100 * gap,
        "validation_miss_rate_by_size": validation_sizes,
        "test_miss_rate_by_size": test_sizes,
        "generalization_assessment": "broadly stable with a small aggregate decline",
    }


def _galleries(
    snapshot: Path,
    run: Path,
    images: list,
    tp: list[dict],
    fp: list[dict],
    fn: list[dict],
    sizes: dict,
) -> dict:
    def area(box) -> float:
        return box[2] * box[3]

    def quality(row: dict) -> float:
        return row["iou"] * row["confidence"]

    missed = [row for row in fn if row["status"] == "missed"]
    candidates = {
        "correct_detections": select_representative(tp, quality, "predicted_class", limit=2),
        "false_positives": select_representative(
            [row for row in fp if row["status"] == "unmatched"],
            lambda row: row["confidence"],
            "predicted_class",
            limit=2,
        ),
        "false_negatives": select_representative(
            missed, lambda row: row["relative_area"], "ground_truth_class", limit=2
        ),
        "small_missed_defects": select_representative(
            [row for row in missed if size_bucket(row["relative_area"], sizes) == "small"],
            lambda row: row["relative_area"],
            "ground_truth_class",
            limit=2,
        ),
        "large_correct_detections": select_representative(
            [
                row
                for row in tp
                if size_bucket(area(row["ground_truth_box_xywh"]), sizes) == "large"
            ],
            quality,
            "predicted_class",
            limit=2,
        ),
        "class_confusions": select_representative(
            [row for row in fp if row["status"] == "class_confusion"],
            quality,
            "ground_truth_class",
            limit=2,
        ),
    }
    return draw_gallery(snapshot, run, images, candidates)


def _diagnostics(run: Path, snapshot: Path, manifest: dict, metadata: dict, freeze: dict) -> dict:
    native = read_json(run / METRICS)
    identity = metadata["identity"]
    images, raw = read_capture(
        snapshot,
        run / CAPTURE,
        manifest,
        split="test",
        threshold=identity["operating_confidence"],
        expected_images=metadata["test_images"],
    )
    rows = [row for image in images for row in image.rows]
    fn = [row for image in images for row in image.misses]
    active = [row for row in rows if row["above_operating_confidence"]]
    fp = [row for row in active if row["status"] != "true_positive"]
    tp = [row for row in active if row["status"] == "true_positive"]
    classes = class_metrics(run, native, images, tp, fp, fn, raw)
    counts = confusion_counts(images)
    pairs, matrix_rows = confusion_tables(counts)
    size_definition, sizes = size_breakdown(images, definition=freeze["size_definition"])
    comparison = _comparison(native, classes, sizes, freeze)
    error_stats = {
        "confidence": identity["operating_confidence"],
        "match_iou": identity["match_iou"],
        "true_positives": len(tp),
        "false_positives": len(fp),
        "false_negatives": len(fn),
        "fp_status": dict(Counter(row["status"] for row in fp)),
        "fn_status": dict(Counter(row["status"] for row in fn)),
        "per_class": [
            {
                key: row[key]
                for key in ("class", "true_positives", "false_positives", "false_negatives")
            }
            for row in classes
        ],
        "annotations_to_background": sum(counts[code][BACKGROUND] for code in CODE.values()),
        "class_confusions": pairs,
        "predictions_at_evaluation_floor": len(rows),
        "frozen_annotations": raw.total(),
        "evaluated_annotations": sum(row["evaluated_annotations"] for row in classes),
    }
    for name, data in (
        ("per_class_metrics", classes),
        ("size_analysis", {"definition": size_definition, "counts": sizes}),
        ("confusion_matrix", {"axis_order": list(counts), "counts": counts, "pairs": pairs}),
        ("error_statistics", error_stats),
        ("validation_vs_test_comparison", comparison),
    ):
        write_json(run / f"{name}.json", data)
    for name, records in (
        (
            "final_test_metrics",
            [{"metric": label, "value": native[key]} for key, label in SCORES.items()],
        ),
        ("per_class_metrics", classes),
        ("size_analysis", sizes),
        ("confusion_matrix", matrix_rows),
        ("predictions", rows),
        ("false_positives", fp),
        ("false_negatives", fn),
    ):
        write_csv(run / f"{name}.csv", records)
    plot_diagnostics(run, classes, counts, sizes, partition="test")
    gallery = _galleries(snapshot, run, images, tp, fp, fn, size_definition)
    for category, examples in gallery.items():
        if (category != "class_confusions" or pairs) and not examples:
            raise ValueError(f"Missing representative test examples for {category}.")
    write_json(run / "gallery_index.json", gallery)
    for name, expected in (
        ("predictions", len(rows)),
        ("false_positives", len(fp)),
        ("false_negatives", len(fn)),
    ):
        with (run / f"{name}.csv").open(encoding="utf-8", newline="") as stream:
            if sum(1 for _ in csv.DictReader(stream)) != expected:
                raise ValueError(f"Incomplete exported test records: {name}.")
    for name in (
        "BoxPR_curve.png",
        "inference_stdout.log",
        "inference_stderr.log",
        "figures/confusion_matrix.png",
        "figures/object_size.png",
    ):
        if not (run / name).stat().st_size:
            raise ValueError(f"Missing final test artifact: {name}.")
    return {
        "metrics": native,
        "size_analysis": sizes,
        "size_definition": size_definition,
        "comparison": comparison,
        "error_statistics": error_stats,
        "per_class": classes,
        "strongest_class": max(classes, key=lambda row: row["ap50_95"])["class"],
        "weakest_class": min(classes, key=lambda row: row["ap50_95"])["class"],
        "gallery_counts": {category: len(examples) for category, examples in gallery.items()},
        "detector_status": "accepted_with_limitations_for_research",
    }


def finalize_test(root: Path, run: Path) -> Path:
    """Verify unchanged inputs, then report from the archived pass without inference."""
    from roadguard.evaluation.stage6_document import arabic_report

    experiments = root / "artifacts/experiments"
    if run.parent != experiments or not run.name.startswith(EXPERIMENT + "-"):
        raise ValueError("Use the recorded Stage 6 attempt under artifacts/experiments.")
    metadata_file = run / "metadata.json"
    metadata = read_json(metadata_file)
    freeze = read_json(run / "frozen_protocol.json")
    preflight = read_json(run / "preflight.json")
    identity = metadata["identity"]
    if (
        metadata["status"] not in {"captured", "passed", "failed_quality_gates"}
        or identity != freeze["test_identity"]
        or not freeze["configuration_unchanged_since_stage5"]
        or identity["split"] != "test"
        or metadata["training_occurred"]
        or not metadata["test_evaluation_started"]
        or metadata["test_evaluation_passes"] != 1
        or sha256(run / CAPTURE) != metadata["capture_sha256"]
        or sha256(root / freeze["stage5_run"] / "stage5_result.json")
        != freeze["stage5_result_sha256"]
    ):
        raise ValueError("Stage 6 requires evidence of the single completed frozen test pass.")
    snapshot = Path(metadata["snapshot_path"])
    verify_descriptor(run, snapshot, include_test=True)
    manifest, integrity = verify_protected_inputs(root, snapshot, preflight, splits=SPLITS)
    checksums = split_checksums(manifest)
    if checksums != preflight["splits"] or metadata["test_images"] != checksums["test"]["images"]:
        raise ValueError("The test dataset or partition digests differ from the frozen preflight.")
    verified = integrity | {
        "splits": checksums,
        "training_occurred": False,
        "test_evaluated": True,
        "test_evaluation_passes": 1,
        "checkpoint_unchanged": True,
    }
    write_json(run / "integrity.json", verified)
    analysis = _diagnostics(run, snapshot, manifest, metadata, freeze)
    verify_inventory(root, preflight | {"snapshot": str(snapshot)})
    result = analysis | {
        "status": "quality_gates_pending",
        "run": run.relative_to(root).as_posix(),
        "checkpoint": metadata["checkpoint"],
        "test_images": metadata["test_images"],
        "integrity": verified,
    }
    report = run / "stage6_result.json"
    write_json(report, result)
    gates = run_quality_gates(root, run, NOTEBOOK, notebook_env={"ROADGUARD_STAGE6_RUN": str(run)})
    status = "passed" if all(gates.values()) else "failed_quality_gates"
    metadata.update(status=status, quality_gates=gates)
    result.update(status=status, quality_gates=gates)
    write_text(root / DOCUMENT, arabic_report(result, metadata, freeze))
    write_json(metadata_file, metadata)
    write_json(report, result)
    write_json(experiments / "stage6_result.json", result)
    if status != "passed":
        raise RuntimeError(f"Stage 6 quality gates failed: {gates}")
    return report

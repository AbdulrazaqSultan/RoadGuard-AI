"""Archive Stage 5 diagnostics and verify their provenance and reviewability."""

import csv
import json
from collections import Counter
from pathlib import Path

from roadguard.data.annotations import read_yolo
from roadguard.data.dataset import CLASS_NAMES
from roadguard.data.validation import sha256
from roadguard.evaluation.stage5 import CAPTURE, CODE, EXPERIMENT, verify_protected_inputs
from roadguard.evaluation.stage5_errors import (
    BACKGROUND,
    LOCALIZATION_MIN_IOU,
    ReviewedImage,
    confidence_breakdown,
    confusion_counts,
    review_image,
    size_breakdown,
)
from roadguard.evaluation.stage5_figures import (
    make_galleries,
    make_visual_sample,
    plot_diagnostics,
    rank_hard_examples,
)
from roadguard.training.screening import verify_descriptor
from roadguard.utils.paths import write_json, write_text
from roadguard.utils.quality import run_quality_gates

NOTEBOOK = Path("notebooks/05_validation_error_analysis.ipynb")
DOCUMENT = Path("docs/stage5_validation_error_analysis_ar.md")
VISUAL_SEED = 42


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict]) -> None:
    """Publish diagnostic records with structured values serialized as compact JSON."""
    temporary = path.with_suffix(".csv.partial")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, separators=(",", ":"))
                    if isinstance(value, list | tuple | dict)
                    else value
                    for key, value in row.items()
                }
            )
    temporary.replace(path)


def _open_run(root: Path, run: Path) -> dict:
    if run.parent != root / "artifacts/experiments" or not run.name.startswith(EXPERIMENT + "-"):
        raise ValueError("Specify a Stage 5 attempt in artifacts/experiments.")
    metadata = _read(run / "metadata.json")
    if metadata["identity"]["split"] != "val" or metadata["identity"]["device"] != 0:
        raise ValueError("The selected Stage 5 analysis requires the completed CUDA capture.")
    verify_descriptor(run, Path(metadata["snapshot_path"]))
    return metadata


def class_metrics(
    run: Path,
    native: dict,
    images: list[ReviewedImage],
    true_positives: list[dict],
    fp: list[dict],
    fn: list[dict],
    raw: Counter,
) -> list[dict]:
    """Join native class scores to fixed-operating-point counts."""
    captured = Counter(CODE[label.class_id] for image in images for label in image.labels)
    tps = Counter(row["predicted_class"] for row in true_positives)
    fps = Counter(row["predicted_class"] for row in fp)
    fns = Counter(row["ground_truth_class"] for row in fn)
    classes = []
    for index, code in CODE.items():
        if captured[code] != tps[code] + fns[code] or captured[code] > raw[code]:
            raise ValueError(f"Annotation counts do not reconcile for {code} in {run}.")
        key = CLASS_NAMES[index]
        count = captured[code]
        detections = tps[code] + fps[code]
        classes.append(
            {
                "class": code,
                "name": key,
                "frozen_annotations": raw[code],
                "evaluated_annotations": count,
                **native["per_class"][key],
                "true_positives": tps[code],
                "false_positives": fps[code],
                "false_negatives": fns[code],
                "precision_at_operating_point": tps[code] / detections if detections else None,
                "recall_at_operating_point": tps[code] / count if count else None,
                "false_negative_rate": fns[code] / count if count else None,
            }
        )
    return classes


def read_capture(
    snapshot: Path,
    capture: Path,
    manifest: dict,
    *,
    split: str,
    threshold: float,
    expected_images: int,
) -> tuple[list[ReviewedImage], Counter]:
    """Review complete captured records and count verified partition annotations."""
    entries = {entry["image"]: entry for entry in manifest["files"] if entry["split"] == split}
    split_labels = (snapshot / "labels" / split).resolve()
    raw: Counter = Counter()
    by_image = {}
    with capture.open(encoding="utf-8") as stream:
        for line in stream:
            captured = json.loads(line)
            image = captured["image"]
            if image in by_image or image not in entries:
                raise ValueError(f"Unexpected capture path: {image}")
            entry = entries[image]
            label_path = (snapshot / entry["label"]).resolve()
            if not label_path.is_relative_to(split_labels):
                raise ValueError(f"Capture label falls outside {split}: {image}")
            labels, errors = read_yolo(label_path)
            if errors:
                raise ValueError(f"Invalid {split} label {entry['label']}: {errors[0]}")
            raw.update(CODE[label.class_id] for label in labels)
            by_image[image] = review_image(captured, entry["source"], threshold)
    if by_image.keys() != entries.keys() or len(entries) != expected_images:
        raise ValueError(f"Prediction records do not cover the full {split} split.")
    return [by_image[key] for key in sorted(by_image)], raw


def confusion_tables(matrix: dict) -> tuple[list[dict], list[dict]]:
    """Return ranked cross-class confusions and a long-form matrix for CSV."""
    pairs = sorted(
        (
            {
                "actual": actual,
                "predicted": predicted,
                "count": matrix[actual][predicted],
                "fraction_of_actual_annotations": matrix[actual][predicted]
                / sum(matrix[actual].values()),
            }
            for actual in CODE.values()
            for predicted in CODE.values()
            if actual != predicted and matrix[actual][predicted]
        ),
        key=lambda row: (-row["count"], row["actual"], row["predicted"]),
    )
    matrix_rows = []
    for actual, values in matrix.items():
        total = sum(values.values())
        matrix_rows.extend(
            {
                "actual": actual,
                "predicted": predicted,
                "count": count,
                "row_fraction": count / total if total else None,
            }
            for predicted, count in values.items()
        )
    return pairs, matrix_rows


def analyze_capture(root: Path, run: Path) -> Path:
    """Classify the captured boxes, write diagnostics and prepare visual evidence."""
    metadata = _open_run(root, run)
    if metadata["status"] not in {"captured", "analyzed"}:
        raise ValueError("Stage 5 needs a completed validation capture.")
    snapshot = Path(metadata["snapshot_path"])
    manifest = _read(snapshot / "manifest.json")
    threshold = metadata["identity"]["operating_confidence"]
    images, raw = read_capture(
        snapshot,
        run / CAPTURE,
        manifest,
        split="val",
        threshold=threshold,
        expected_images=metadata["val_images"],
    )
    rows = [row for image in images for row in image.rows]
    fn = [row for image in images for row in image.misses]
    active = [row for row in rows if row["above_operating_confidence"]]
    fp = [row for row in active if row["status"] != "true_positive"]
    tp = [row for row in active if row["status"] == "true_positive"]
    native = _read(run / "validation_metrics.json")
    classes = class_metrics(run, native, images, tp, fp, fn, raw)
    matrix = confusion_counts(images)
    pairs, matrix_rows = confusion_tables(matrix)
    size_method, sizes = size_breakdown(images)
    confidence, histogram = confidence_breakdown(rows, threshold)
    hard = rank_hard_examples(images)
    for name, records in (
        ("predictions", rows),
        ("false_positives", fp),
        ("false_negatives", fn),
        ("per_class_metrics", classes),
        ("confusion_matrix", matrix_rows),
        ("size_analysis", sizes),
        ("confidence_histogram", histogram),
        ("hard_examples", hard),
    ):
        write_csv(run / f"{name}.csv", records)
    stats = {
        "status": "analyzed",
        "validation_images": len(images),
        "frozen_annotations": raw.total(),
        "evaluated_annotations": sum(row["evaluated_annotations"] for row in classes),
        "predictions_at_evaluation_floor": len(rows),
        "operating_confidence": threshold,
        "true_positives": len(tp),
        "false_positives": len(fp),
        "false_negatives": len(fn),
        "fp_status": dict(Counter(row["status"] for row in fp)),
        "fn_status": dict(Counter(row["status"] for row in fn)),
        "misses_with_same_class_match_below_threshold": sum(
            row["below_threshold_prediction_index"] is not None for row in fn
        ),
        "localization_min_iou": LOCALIZATION_MIN_IOU,
        "matched_iou_below_0_60": sum(row["iou"] < 0.60 for row in tp),
        "per_class": classes,
        "strongest_ap_class": max(classes, key=lambda row: row["ap50_95"])["class"],
        "weakest_ap_class": min(classes, key=lambda row: row["ap50_95"])["class"],
        "most_false_negatives_class": max(classes, key=lambda row: row["false_negatives"])["class"],
        "highest_miss_rate_class": max(classes, key=lambda row: row["false_negative_rate"])[
            "class"
        ],
        "confusion_pairs": pairs,
        "matched_at_wrong_class": sum(row["count"] for row in pairs),
        "annotations_to_background": sum(matrix[code][BACKGROUND] for code in CODE.values()),
        "confusion_matrix": matrix,
        "size_definition": size_method,
        "size_analysis": sizes,
        "confidence_analysis": confidence,
        "validation_metrics": native,
        "backend_delta_from_stage4": {
            key: native[key] - metadata["stage4_validation_metrics"][key]
            for key in ("map50_95", "map50", "precision", "recall")
        },
    }
    write_json(run / "error_statistics.json", stats)
    write_json(run / "size_analysis.json", {"definition": size_method, "counts": sizes})
    write_json(run / "confidence_analysis.json", {"summary": confidence, "histogram": histogram})
    write_json(run / "confusion_matrix.json", {"counts": matrix, "pairs": pairs})
    plot_diagnostics(run, classes, matrix, sizes, histogram)
    gallery = make_galleries(snapshot, run, images, hard)
    audit = make_visual_sample(snapshot, run, fp, fn, seed=VISUAL_SEED)
    write_json(run / "gallery_index.json", gallery)
    write_json(run / "visual_audit/sample.json", {"seed": VISUAL_SEED, "samples": audit})
    metadata.update(status="analyzed", capture_sha256=sha256(run / CAPTURE))
    write_json(run / "metadata.json", metadata)
    return run


def _audit(run: Path) -> dict:
    sampled = _read(run / "visual_audit/sample.json")
    evidence = _read(run / "visual_audit/observations.json")
    samples = sampled["samples"]
    if evidence.keys() != {row["sample_id"] for row in samples}:
        raise ValueError("Every sampled validation box requires a visual observation.")
    flattened = []
    counts: dict[str, Counter] = {"FP": Counter(), "FN": Counter()}
    for sample in samples:
        sample_id = sample["sample_id"]
        observation = evidence[sample_id]
        if not observation["cue"].strip() or not observation["evidence"].strip():
            raise ValueError(f"No visual evidence for the reviewed sample {sample_id}.")
        counts[sample_id[:2]][observation["cue"]] += 1
        flattened.append(sample | observation)
    write_csv(run / "visual_audit/observations.csv", flattened)
    return {
        "method": "seeded box sample with manual visual review; cues are not causal labels",
        "seed": sampled["seed"],
        "counts": {key: dict(counter) for key, counter in counts.items()},
        "n": {key: counter.total() for key, counter in counts.items()},
        "contact_sheets": sorted({f"visual_audit/{row['contact_sheet']}" for row in samples}),
    }


def finalize_analysis(root: Path, run: Path) -> Path:
    """Verify the archived analysis, execute the notebook and write the Arabic report."""
    from roadguard.evaluation.stage5_document import arabic_report

    metadata = _open_run(root, run)
    if metadata["status"] not in {"analyzed", "passed", "failed_quality_gates"}:
        raise ValueError("Complete analysis before finalizing Stage 5.")
    if metadata["capture_sha256"] != sha256(run / CAPTURE):
        raise ValueError("The Stage 5 capture changed after the analysis.")
    _, integrity = verify_protected_inputs(
        root,
        Path(metadata["snapshot_path"]),
        _read(run / "preflight.json"),
        splits=("train", "val"),
    )
    write_json(run / "integrity.json", integrity)
    stats = _read(run / "error_statistics.json")
    gallery = _read(run / "gallery_index.json")
    if len(gallery) != 6 or any(not items for items in gallery.values()):
        raise ValueError("The Stage 5 visual galleries are incomplete.")
    for items in gallery.values():
        for item in items:
            if not (run / item["figure"]).is_file():
                raise ValueError(f"Missing visual gallery artifact: {item['figure']}")
    for name, expected in (
        ("predictions", stats["predictions_at_evaluation_floor"]),
        ("false_positives", stats["false_positives"]),
        ("false_negatives", stats["false_negatives"]),
    ):
        with (run / f"{name}.csv").open(encoding="utf-8", newline="") as stream:
            if sum(1 for _ in csv.DictReader(stream)) != expected:
                raise ValueError(f"Archive row count mismatch for {name}.")
    audit = _audit(run)
    result = stats | {
        "status": "quality_gates_pending",
        "run": run.relative_to(root).as_posix(),
        "checkpoint": metadata["checkpoint"],
        "integrity": integrity,
        "visual_audit": audit,
    }
    report = run / "stage5_result.json"
    write_json(report, result)
    gates = run_quality_gates(root, run, NOTEBOOK, notebook_env={"ROADGUARD_STAGE5_RUN": str(run)})
    status = "passed" if all(gates.values()) else "failed_quality_gates"
    metadata.update(status=status, quality_gates=gates)
    result.update(
        status=status,
        quality_gates=gates,
        stage6_recommended=status == "passed",
        causal_attribution="Measured errors and sampled visual cues support diagnostics only.",
    )
    write_text(root / DOCUMENT, arabic_report(result, metadata))
    write_json(run / "metadata.json", metadata)
    write_json(report, result)
    write_json(root / "artifacts/experiments/stage5_result.json", result)
    if status != "passed":
        raise RuntimeError(f"Stage 5 quality gates failed: {gates}")
    return report

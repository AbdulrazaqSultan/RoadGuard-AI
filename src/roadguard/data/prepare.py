"""Publish validated, versioned YOLO snapshots and split manifests."""

import csv
import hashlib
import io
import json
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from roadguard.data.dataset import CLASS_NAMES, SPLITS, ImageRecord
from roadguard.data.validation import sha256
from roadguard.utils.paths import write_json, write_text

MANIFEST = "manifest.json"


def _digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _identity(manifest: dict) -> str:
    return _digest(json.dumps(manifest, sort_keys=True).encode())


def _label_text(record: ImageRecord) -> str:
    return "".join(f"{box.yolo_line()}\n" for box in record.boxes)


def _entry(record: ImageRecord, split: str, *, enriched: bool) -> dict:
    # Source-relative hashes keep names unique across platforms and import layouts.
    source = record.relative_image.as_posix()
    name = f"{_digest(source.encode())[:16]}{record.image.suffix.lower()}"
    label = Path(name).with_suffix(".txt").name
    entry = {
        "source": source,
        "split": split,
        "image": f"images/{split}/{name}",
        "label": f"labels/{split}/{label}",
        "image_sha256": record.image_hash,
        "label_sha256": _digest(_label_text(record).encode()),
    }
    if enriched:
        metadata = record.provenance
        entry.update(
            source_path=source, source=metadata.source, publisher_split=record.split,
            country=metadata.country, view=metadata.view, capture_group=metadata.capture_group,
            width=record.width, height=record.height, pixel_sha256=record.pixel_hash,
            objects=len(record.boxes),
            class_ids=sorted({box.class_id for box in record.boxes}),
            class_counts=dict(sorted(Counter(str(box.class_id) for box in record.boxes).items())),
        )
    return entry


def _manifest_csv(manifest: dict) -> str:
    stream = io.StringIO(newline="")
    fields = list(manifest["files"][0])
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for entry in manifest["files"]:
        row = dict(entry)
        for field in ("class_ids", "class_counts"):
            row[field] = json.dumps(row[field])
        writer.writerow(row)
    return stream.getvalue()


def snapshot_summary(manifest: dict) -> dict:
    """Derive small split summaries without loading images or model weights."""
    splits = {}
    for name in SPLITS:
        entries = [entry for entry in manifest["files"] if entry["split"] == name]
        summary: dict = {"images": len(entries)}
        if manifest["schema"] == 2:
            instances: Counter[str] = Counter()
            for entry in entries:
                instances.update(entry["class_counts"])
            summary.update(
                objects=sum(entry["objects"] for entry in entries),
                images_per_source=dict(Counter(entry["source"] for entry in entries)),
                annotations_per_class={
                    label: instances[str(class_id)] for class_id, label in CLASS_NAMES.items()
                },
            )
        splits[name] = summary
    return splits


def _metadata(manifest: dict) -> dict:
    return {
        "fingerprint": _identity(manifest),
        "dataset_version": manifest["dataset_version"],
        "source_release": manifest["source_release"],
        "seed": manifest["seed"],
        "target_ratios": manifest["ratios"],
        "splits": snapshot_summary(manifest),
    }


def write_dataset_yaml(destination: Path, snapshot: Path, *, include_test: bool) -> None:
    """Write an Ultralytics descriptor pointing to a verified local snapshot."""
    data = {"path": snapshot.resolve().as_posix(), "train": "images/train", "val": "images/val"}
    if include_test:
        data["test"] = "images/test"
    data["names"] = CLASS_NAMES
    write_text(destination, yaml.safe_dump(data, sort_keys=False))


def publish_snapshot(
    destination: Path,
    partitions: dict[str, list[ImageRecord]],
    *,
    version: str,
    seed: int,
    ratios: dict[str, float],
    groups: dict[str, str] | None = None,
    source_release: dict | None = None,
) -> Path:
    """Publish atomically or verify and reuse a matching immutable snapshot."""
    rows = [
        (_entry(record, split, enriched=source_release is not None), record)
        for split in SPLITS
        for record in sorted(partitions[split], key=lambda row: row.relative_image)
    ]
    manifest = {
        "schema": 1 if source_release is None else 2,
        "dataset_version": version,
        "seed": seed,
        "ratios": ratios,
        "groups": groups,
        "files": [entry for entry, _ in rows],
        "class_names": list(CLASS_NAMES.values()),
    }
    if source_release is not None:
        manifest["source_release"] = source_release
    fingerprint = _identity(manifest)
    output = destination / fingerprint
    if output.is_dir():
        verify_snapshot(output)
        return output
    destination.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".roadguard-", dir=destination) as staging:
        snapshot = Path(staging) / fingerprint
        for entry, record in rows:
            image = snapshot / entry["image"]
            image.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(record.image, image)
            label = snapshot / entry["label"]
            label.parent.mkdir(parents=True, exist_ok=True)
            label.write_text(_label_text(record), encoding="utf-8", newline="\n")
        write_json(snapshot / MANIFEST, manifest)
        if source_release is not None:
            write_text(snapshot / "manifest.csv", _manifest_csv(manifest))
            write_json(snapshot / "metadata.json", _metadata(manifest))
        write_dataset_yaml(snapshot / "dataset.yaml", output, include_test=True)
        verify_snapshot(snapshot)
        try:
            snapshot.rename(output)
        except OSError:
            if not output.is_dir():
                raise
            verify_snapshot(output)
    return output


def read_snapshot(root: Path) -> dict:
    """Read and check the manifest identity without hashing all image files."""
    with (root / MANIFEST).open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if (
        manifest.get("schema") not in {1, 2}
        or manifest.get("class_names") != list(CLASS_NAMES.values())
        or _identity(manifest) != root.name
    ):
        raise ValueError(f"Unexpected snapshot identity: {root}. Run prepare_dataset.py.")
    return manifest


def verify_snapshot(root: Path, *, splits: tuple[str, ...] = SPLITS) -> dict:
    """Check snapshot identity and verify files only in the requested partitions."""
    if not splits or not set(splits).issubset(SPLITS):
        raise ValueError("Snapshot verification requires known dataset partitions.")
    manifest = read_snapshot(root)
    if manifest["schema"] == 2 and (
        (root / "manifest.csv").read_text(encoding="utf-8") != _manifest_csv(manifest)
        or json.loads((root / "metadata.json").read_text(encoding="utf-8")) != _metadata(manifest)
    ):
        raise ValueError(f"Modified manifest sidecars: {root}")
    resolved_root = root.resolve()
    expected: dict[str, str] = {}
    for entry in manifest["files"]:
        if entry["split"] not in splits:
            continue
        for kind in ("image", "label"):
            relative = entry[kind]
            prefix = ("images" if kind == "image" else "labels", entry["split"])
            if Path(relative).parts[:2] != prefix or relative in expected:
                raise ValueError(f"Repeated or misplaced snapshot file: {relative}")
            expected[relative] = entry[f"{kind}_sha256"]
    folders = (
        [root / kind for kind in ("images", "labels")]
        if set(splits) == set(SPLITS)
        else [root / kind / split for kind in ("images", "labels") for split in splits]
    )
    cache_paths = {f"labels/{split}.cache" for split in splits}
    actual = {
        path.relative_to(root).as_posix()
        for folder in folders
        for path in folder.rglob("*")
        if path.is_file()
    } - cache_paths
    if actual != expected.keys():
        extra = sorted(actual - expected.keys())[:5]
        missing = sorted(expected.keys() - actual)[:5]
        raise ValueError(f"Snapshot {root} has missing files {missing} or extra files {extra}.")

    def check(item: tuple[str, str]) -> None:
        relative, digest = item
        path = root / relative
        if not path.resolve().is_relative_to(resolved_root):
            raise ValueError(f"Snapshot path escapes its root: {relative}")
        if sha256(path) != digest:
            raise ValueError(f"Modified snapshot file: {path}")

    with ThreadPoolExecutor(max_workers=8) as executor:
        for _ in executor.map(check, expected.items()):
            pass
    return manifest

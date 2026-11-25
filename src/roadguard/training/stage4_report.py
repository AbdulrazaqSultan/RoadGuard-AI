"""Verify the Stage 4 archive and publish the validation-only decision."""

import math
from pathlib import Path
from statistics import mean

import yaml

from roadguard.data.dataset import CLASS_NAMES, RDD_CLASSES
from roadguard.data.prepare import verify_snapshot
from roadguard.data.validation import sha256
from roadguard.training.archive import ExperimentArchive, open_experiment
from roadguard.training.screening import verify_descriptor, verify_inventory, verify_raw_archives
from roadguard.training.stage4 import EXPERIMENT, FINGERPRINT, PROFILE
from roadguard.utils.paths import write_json, write_text
from roadguard.utils.quality import run_quality_gates

NOTEBOOK = Path("notebooks/04_yolo26s_long_training_review.ipynb")
DOCUMENT = Path("docs/stage4_controlled_long_training_ar.md")
MAP = "metrics/mAP50-95(B)"
PRACTICAL_LATE_GAIN = 0.001
PLOTS = (
    "results.png",
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
    "BoxPR_curve.png",
    "BoxF1_curve.png",
    "BoxP_curve.png",
    "BoxR_curve.png",
)


def _verify_archive(archive: ExperimentArchive, preflight: dict, experiments: Path) -> dict:
    run = archive.directory
    metadata, best, review = archive.metadata, archive.best, archive.validation_review
    if (
        metadata["status"] != "completed"
        or metadata["experiment_name"] != EXPERIMENT
        or metadata["config"] != preflight["config"]
        or metadata["identity"]["versions"] != preflight["environment"]
        or metadata["identity"]["weights_sha256"] != preflight["initialization"]["sha256"]
        or metadata["dataset_fingerprint"] != FINGERPRINT
        or metadata["model_name"] != "YOLO26s"
        or metadata["fitted_split"] != "train"
        or metadata["selection_split"] != "val"
        or metadata["effective_amp"] is not True
        or metadata["effective_batch_size"] != 8
    ):
        raise ValueError("Stage 4 run metadata differs from the approved preflight.")
    prior = experiments / preflight["initialization"]["stage3b_run"]
    args = yaml.safe_load((run / "args.yaml").read_text(encoding="utf-8"))
    prior_args = yaml.safe_load((prior / "args.yaml").read_text(encoding="utf-8"))
    changed = {
        key for key in args.keys() | prior_args.keys() if args.get(key) != prior_args.get(key)
    }
    if (
        changed - {"data", "save_dir", "name", "model", "epochs", "patience"}
        or (args["epochs"], args["patience"]) != (50, 10)
        or Path(args["data"]) != run / "dataset.yaml"
        or Path(args["model"]) != Path(metadata["pretrained_weights"]["path"])
    ):
        raise ValueError(
            f"Effective backend arguments differ from the approved protocol: {changed}"
        )
    verify_descriptor(run, Path(preflight["snapshot"]))
    stopping = metadata["stopping"]
    history = archive.epoch_metrics
    scores = [float(row[MAP]) for row in history]
    if (
        not history
        or [int(row["epoch"]) for row in history] != list(range(1, stopping["final_epoch"] + 1))
        or not all(math.isfinite(value) and 0 <= value <= 1 for value in scores)
        or stopping["reason"] not in {"validation_patience_exhausted", "epoch_budget_completed"}
        or stopping["epoch_budget"] != 50
        or stopping["patience"] != 10
    ):
        raise ValueError("Stage 4 epoch history or stopping evidence is incomplete.")
    if (
        best is None
        or best["checkpoint"] != "weights/best.pt"
        or best["selection_metric"] != "validation mAP50-95"
        or not 1 <= best["epoch"] <= len(history)
        or not math.isclose(scores[best["epoch"] - 1], max(scores), abs_tol=0.00001)
        or sha256(run / best["checkpoint"]) != best["sha256"]
        or best["validation_map50_95"] != metadata["validation_metrics"]["map50_95"]
    ):
        raise ValueError("Stage 4 best checkpoint does not match the archived validation history.")
    if (
        review is None
        or metadata["validation_review_status"] != "saved"
        or review["scope"] != "validation_sample"
        or not review["samples"]
        or review["sample_count"] != len(review["samples"])
    ):
        raise ValueError("Stage 4 validation prediction review is missing.")
    for sample in review["samples"]:
        figure = Path(sample["figure"])
        if (
            not Path(sample["image"]).is_relative_to("images/val")
            or not figure.is_relative_to("validation_review")
            or not (run / figure).resolve().is_relative_to(run)
            or not (run / figure).stat().st_size
        ):
            raise ValueError("An archived prediction falls outside the validation review.")
    for name in (*PLOTS, "stdout.log", "gpu_telemetry.csv", "weights/last.pt", PROFILE.name):
        if not (run / name).stat().st_size:
            raise ValueError(f"Empty Stage 4 artifact: {name}")
    if not (run / "stderr.log").is_file():
        raise FileNotFoundError("The Stage 4 stderr log was not archived.")
    return {"best": best, "last_sha256": sha256(run / "weights/last.pt")}


def validation_trend(history: list[dict], stopped_early: bool) -> dict:
    """Compare adjacent five-epoch windows of held-out AP and training/validation losses."""
    window = min(5, len(history) // 2)
    early, late = history[-2 * window : -window], history[-window:]
    if not window:
        raise ValueError("At least two epochs are needed to compare the validation trend.")
    scores = [float(row[MAP]) for row in history]
    columns = [key for key in history[0] if key.endswith("_loss")]
    changes = {
        key: mean(float(row[key]) for row in late) - mean(float(row[key]) for row in early)
        for key in (MAP, *columns)
    }
    late_gain = max(scores[-window:]) - max(scores[:-window])
    train_changes = [value for key, value in changes.items() if key.startswith("train/")]
    val_changes = [value for key, value in changes.items() if key.startswith("val/")]
    overfitting_signal = (
        train_changes
        and val_changes
        and all(value < 0 for value in train_changes)
        and any(value > 0 for value in val_changes)
        and changes[MAP] < 0
    )
    return {
        "window_epochs": window,
        "last_epoch_map50_95": scores[-1],
        "new_best_in_final_window": late_gain > 0,
        "final_window_best_minus_earlier_best": late_gain,
        "practical_late_gain": PRACTICAL_LATE_GAIN,
        "mean_change": changes,
        "overfitting_signal": bool(overfitting_signal),
        "extension_worth_considering": (
            not stopped_early
            and late_gain >= PRACTICAL_LATE_GAIN
            and changes[MAP] > 0
            and not overfitting_signal
        ),
    }


def finalize_stage4(root: Path, run: Path, preflight: dict, telemetry: dict) -> dict:
    """Seal the archive and Stage 4 report after post-training validation and quality gates."""
    experiments = root / "artifacts/experiments"
    archive = open_experiment(run)
    protected = verify_inventory(root, preflight)
    hashes = verify_raw_archives(root / "data/raw/unified")
    if hashes != preflight["raw_archive_sha256"]:
        raise ValueError("Raw archive checksums changed after training.")
    verify_snapshot(Path(preflight["snapshot"]), splits=("train", "val"))
    integrity = {
        "status": "passed",
        "snapshot_fingerprint": FINGERPRINT,
        "train_val_checksums_match": True,
        "train_val_checksum_counts": preflight["train_val_checksum_counts"],
        "protected_inventory": protected,
        "raw_archive_sha256": hashes,
        "test_contents_opened": False,
        "test_evaluated": False,
    }
    write_json(experiments / "stage4_integrity.json", integrity)
    write_json(run / "integrity.json", integrity)
    write_json(run / "preflight.json", preflight)
    checkpoints = _verify_archive(archive, preflight, experiments)
    metadata = archive.metadata
    metadata.update(initialization=preflight["initialization"], gpu_telemetry=telemetry)
    write_json(run / "metadata.json", metadata)
    metrics = metadata["validation_metrics"]
    previous = preflight["initialization"]["stage3b_metrics"]
    result = {
        "status": "quality_gates_pending",
        "run": run.relative_to(root).as_posix(),
        "dataset_fingerprint": FINGERPRINT,
        "selection_split": "val",
        "test_evaluated": False,
        "selected_checkpoint": (run / "weights/best.pt").relative_to(root).as_posix(),
        "checkpoints": checkpoints,
        "validation_metrics": metrics,
        "stage3b_metrics": previous,
        "absolute_map50_95_gain": metrics["map50_95"] - previous["map50_95"],
        "stopping": metadata["stopping"],
        "trend": validation_trend(archive.epoch_metrics, metadata["stopping"]["stopped_early"]),
        "elapsed_seconds": metadata["elapsed_seconds"],
        "peak_torch_cuda_reserved_bytes": metadata["peak_torch_cuda_reserved_bytes"],
        "peak_torch_cuda_allocated_bytes": metadata["peak_torch_cuda_allocated_bytes"],
        "gpu_telemetry": telemetry,
        "integrity": integrity,
    }
    result_path = experiments / "stage4_result.json"
    write_json(result_path, result)
    result["quality_gates"] = run_quality_gates(root, run, NOTEBOOK)
    result["status"] = "passed" if all(result["quality_gates"].values()) else "failed_quality_gates"
    write_text(root / DOCUMENT, _arabic_report(result, metadata))
    write_json(run / "stage4_result.json", result)
    write_json(result_path, result)
    return result


def _arabic_report(result: dict, metadata: dict) -> str:
    metrics, previous = result["validation_metrics"], result["stage3b_metrics"]
    stopping, trend = result["stopping"], result["trend"]
    settings = metadata["config"]["training"]
    run = result["run"]
    best_epoch = result["checkpoints"]["best"]["epoch"]
    gain = result["absolute_map50_95_gain"]
    early = (
        "توقف التدريب مبكرًا بعد استنفاد مهلة التحسن في التحقق."
        if stopping["stopped_early"]
        else "أكمل التدريب ميزانية الحقب المحددة."
    )
    convergence = (
        "يشير التوقف المبكر إلى استقرار أفضل نتيجة تحقق ضمن مهلة الصبر."
        if stopping["stopped_early"]
        else (
            "أصبح mAP50–95 قريبًا من الاستقرار في النافذة الأخيرة."
            if abs(trend["mean_change"][MAP]) < trend["practical_late_gain"]
            else "واصل متوسط mAP50–95 تغيّره قرب نهاية التدريب."
        )
    )
    overfit = (
        "ظهر مؤشر محتمل على فرط التعلّم، إذ هبطت خسائر التدريب وارتفعت إحدى خسائر التحقق "
        "مع انخفاض متوسط mAP50–95."
        if trend["overfitting_signal"]
        else "لم تظهر في النافذة الأخيرة إشارة متزامنة واضحة لفرط التعلّم وفق الخسائر وmAP50–95."
    )
    recommendation = (
        "يدعم الاتجاه النظر في تمديد التدريب بعد موافقة صريحة، مع اعتماد best.pt مرشحًا حاليًا."
        if trend["extension_worth_considering"]
        else "اعتماد best.pt بوصفه مرشح المرحلة التالية. لا تدعم النتائج تمديد التدريب حاليًا."
    )
    status = (
        "اجتازت Stage 4 فحوص الأرشيف وسلامة البيانات والجودة."
        if result["status"] == "passed"
        else "اكتمل التدريب وفحص سلامة البيانات، وتعثّر فحص جودة واحد على الأقل."
    )
    gates = ", ".join(
        f"`{name}` ({'ناجح' if passed else 'متعثر'})"
        for name, passed in result["quality_gates"].items()
    )
    rows = [
        f"| {label} | {previous[key]:.4f} | {metrics[key]:.4f} |"
        for label, key in (
            ("mAP50–95", "map50_95"),
            ("mAP50", "map50"),
            ("Precision", "precision"),
            ("Recall", "recall"),
        )
    ]
    for code, index in RDD_CLASSES.items():
        name = CLASS_NAMES[index]
        old_ap = previous["per_class"][name]["ap50_95"]
        new_ap = metrics["per_class"][name]["ap50_95"]
        rows.append(f"| {code} AP50–95 | {old_ap:.4f} | {new_ap:.4f} |")
    rows.append(
        f"| D40 Recall | {previous['per_class']['pothole']['recall']:.4f} | "
        f"{metrics['per_class']['pothole']['recall']:.4f} |"
    )
    gpu = result["gpu_telemetry"]
    measures = gpu["measures"]
    observations = "، ".join(
        f"{label} {measures[key][stat]:.1f} {unit}"
        for key, stat, label, unit in (
            ("utilization.gpu", "mean", "متوسط الاستخدام", "%"),
            ("temperature.gpu", "max", "أعلى حرارة مرصودة", "°C"),
            ("power.draw", "max", "أعلى قدرة مرصودة", "W"),
            ("memory.used", "max", "أعلى استخدام مرصود للذاكرة", "MiB"),
        )
        if key in measures
    )
    telemetry = (
        f"سُجّلت {gpu['sample_count']} عينة من الجهاز كل {gpu['interval_seconds']} ثانية "
        f"تقريبًا، وكانت القياسات المتاحة {observations}."
        if observations
        else "لم تتوفر قراءات لحظية قابلة للاستخدام من nvidia-smi."
    )
    versions = metadata["identity"]["versions"]
    late_score = trend["last_epoch_map50_95"]
    late_delta = trend["mean_change"][MAP]
    late_best_gain = trend["final_window_best_minus_earlier_best"]
    practical_gain = trend["practical_late_gain"]
    box_delta = trend["mean_change"].get("val/box_loss")
    cls_delta = trend["mean_change"].get("val/cls_loss")
    loss_summary = (
        f"في النافذة نفسها، تغيّر متوسط `val/box_loss` بمقدار {box_delta:+.4f} "
        f"و`val/cls_loss` بمقدار {cls_delta:+.4f}."
        if box_delta is not None and cls_delta is not None
        else ""
    )
    table = "\n".join(rows)
    return f"""# Stage 4 — التدريب الطويل المضبوط لنموذج YOLO26s

## الهدف من المرحلة

متابعة تدريب النموذج الذي تصدّر Stage 3B لقياس أثر زيادة عدد الحقب على دقة كشف عيوب الطريق
ضمن بروتوكول ثابت، مع اختيار نقطة الحفظ بناءً على mAP50–95 في مجموعة التحقق.

## ما تم تنفيذه

- تحقق الفحص الأولي من بصمة اللقطة وبصمات ملفات التدريب والتحقق والأرشيفات الخام،
  ومن تحميل نقطة الحفظ وتشغيل CUDA وTorchVision NMS على RTX 4070 Laptop GPU.
- بدأ التدريب من `best.pt` الخاص بـ Stage 3B مع تهيئة مُحسِّن جديد وجدول من 50 حقبة.
  استُخدمت اللقطة المجمدة نفسها وسياسة المعالجة والزيادة الاصطناعية وإعدادات AdamW السابقة.
- سُجّلت المقاييس والخسائر لكل حقبة. حُفظت `best.pt` و`last.pt`، وأُعيد تقييم
  `best.pt` على مجموعة التحقق وحُفظت نتائج ذلك التقييم.
- بعد التدريب، طابقت بصمات ملفات التدريب والتحقق وملفات الأرشيف الخام السجلات السابقة،
  وتطابق جرد المسارات والأحجام وأوقات التعديل. اقتصر واصف التشغيل على train وval.
  ظلّت مجموعة الاختبار معزولة عن التحميل والتقييم والاختيار.
- فحوص الجودة هي {gates}. حُفظت سجلاتها داخل مجلد التجربة.

## إعدادات التدريب

| الإعداد | القيمة |
|---|---|
| Model | YOLO26s |
| Dataset fingerprint | `{result["dataset_fingerprint"]}` |
| epochs | {settings["epochs"]} |
| patience | {settings["early_stopping"]} |
| imgsz | {settings["image_size"]} |
| batch | {metadata["effective_batch_size"]} |
| seed | {settings["seed"]} |
| device | CUDA 0، RTX 4070 Laptop GPU |
| AMP | مفعّل |
| workers | {settings["workers"]} |
| Optimizer / scheduler | AdamW / cosine |
| lr0 / weight_decay | {settings["learning_rate"]} / {settings["weight_decay"]} |
| Reproducibility | deterministic، cuDNN benchmark وTF32 معطّلان |

إصدارات البيئة هي Python {versions["python"]} وPyTorch {versions["torch"]}
وTorchVision {versions["torchvision"]} وUltralytics {versions["ultralytics"]}
وCUDA {versions["cuda_runtime"]}.

## النتائج

| مقياس التحقق | Stage 3B | Stage 4 |
|---|---:|---:|
{table}

حققت نقطة الحفظ المختارة في الحقبة **{best_epoch}** فرقًا مطلقًا قدره **{gain:+.4f}**
في mAP50–95، أي **{gain * 100:+.2f}** نقطة مئوية. كانت الحقبة الأخيرة
**{stopping["final_epoch"]}** وسجّلت mAP50–95 يساوي **{late_score:.4f}**.
{early} بلغ عدد الحقب منذ آخر تحسّن **{stopping["epochs_without_improvement"]}**
من مهلة قدرها **{stopping["patience"]}** حقب.

بلغت مدة التدريب والتحقق النهائي {result["elapsed_seconds"] / 3600:.2f} ساعة.
بلغت ذروة الذاكرة التي حجزها PyTorch
{result["peak_torch_cuda_reserved_bytes"] / 2**30:.2f} GiB.
{telemetry}

## الملفات المستخدمة في Stage 4

| الملف / المسار | الوظيفة الأساسية | أهم الاستدعاءات أو الاستخدامات |
|---|---|---|
| `scripts/train_stage4.py` | تشغيل المرحلة وأخذ عينات الجهاز | `main()` و`_gpu_samples()` |
| `scripts/train_detector.py` | نقطة الدخول المشتركة للتدريب | `main()` يستدعي `train()` |
| `{PROFILE.as_posix()}` | ملف إعدادات المرحلة | `load_config()` |
| `src/roadguard/training/stage4.py` | الفحص الأولي ومصدر الأوزان | `stage4_preflight()` |
| `src/roadguard/training/trainer.py` | التدريب والأرشفة | `train()` و`model.train()` |
| `src/roadguard/training/label_cache.py` | عزل ذاكرة التسميات | `isolated_label_cache()` |
| `src/roadguard/data/prepare.py` | اللقطة والواصف | `verify_snapshot()` و`write_dataset_yaml()` |
| `src/roadguard/training/screening.py` | فحص السلامة | `verify_raw_archives()` |
| `src/roadguard/evaluation/metrics.py` | مقاييس التحقق | `detection_metrics()` |
| `src/roadguard/training/review.py` | عينات التحقق | `archive_review()` |
| `src/roadguard/training/stage4_report.py` | الأرشيف والقرار | `finalize_stage4()` |
| `{NOTEBOOK.as_posix()}` | مراجعة للعرض | `open_experiment()` والرسومات المحفوظة |
| `{run}/dataset.yaml` | واصف بيانات التشغيل | مسارا train وval وخريطة الفئات |
| `{run}/metadata.json` و`{run}/results.csv` | سجل التشغيل | الإعدادات والمقاييس والخسائر |
| `{run}/weights/best.pt` و`{run}/weights/last.pt` | نقطتا الحفظ | أفضل تحقق والحالة الأخيرة |

## مخرجات المرحلة

حُفظت المخرجات داخل `{run}/`، وتشمل ما يأتي.

- `args.yaml` و`metadata.json` و`preflight.json` و`integrity.json`.
- `weights/best.pt` و`weights/last.pt` و`best_checkpoint.json`.
- `results.csv` و`results.png` ومصفوفة الالتباس ومنحنيات PR وF1 وPrecision وRecall.
- `validation_review/` لصور التنبؤ وتحليل الأخطاء، و`gpu_telemetry.csv` لقراءات الجهاز.
- `stdout.log` و`stderr.log` وسجلات فحوص الجودة و`stage4_result.json`.
- نسخة منفّذة من دفتر المراجعة `{NOTEBOOK.name}`.

## قرار Stage 4

{status} {convergence} تغيّر متوسط mAP50–95 في آخر
{trend["window_epochs"]} حقب مقابل النافذة السابقة بمقدار {late_delta:+.4f}.
كان فرق أفضل نتيجة في تلك النافذة عن أفضل نتيجة قبلها {late_best_gain:+.4f}.
استُخدم فرق {practical_gain * 100:.2f} نقطة مئوية كمرجع عملي لجدوى التمديد.
{loss_summary}
{overfit} {recommendation} يتطلب بدء Stage 5 موافقة صريحة.
"""

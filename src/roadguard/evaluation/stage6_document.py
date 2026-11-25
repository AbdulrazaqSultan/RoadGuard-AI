"""Render the Stage 6 course report from verified final evaluation evidence."""


def arabic_report(result: dict, metadata: dict, freeze: dict) -> str:
    protocol = metadata["identity"]
    conf_floor, conf_diagnostic = (
        protocol["evaluation_confidence_floor"],
        protocol["operating_confidence"],
    )
    metrics = result["metrics"]
    comparison = result["comparison"]
    classes = {row["class"]: row for row in result["per_class"]}
    errors = result["error_statistics"]
    counts = errors["evaluated_annotations"]
    sizes = [row for row in result["size_analysis"] if row["class"] == "All"]
    val_miss, test_miss = (
        comparison["validation_miss_rate_by_size"],
        comparison["test_miss_rate_by_size"],
    )
    gap = comparison["generalization_gap"]
    strongest, weakest = result["strongest_class"], result["weakest_class"]
    top_confusions = "، ".join(
        f"`{row['actual']} → {row['predicted']}` ({row['count']})"
        for row in errors["class_confusions"][:3]
    )
    most_missed = max(classes.values(), key=lambda row: row["false_negatives"])
    localization = errors["fp_status"].get("localization_error", 0)
    background = errors["annotations_to_background"]
    pothole_val = freeze["stage4_full_validation_metrics"]["per_class"]["pothole"]
    pothole = classes["D40"]
    coords = result["size_definition"]
    run = result["run"]
    ms = metrics["latency_ms_per_image"]
    pipeline_ms = sum(ms[key] for key in ("preprocess", "inference", "postprocess"))
    passed = result["status"] == "passed"
    gate_label = (
        "اجتازت Stage 6 فحوص سلامة اللقطة والأوزان والأرشيف والاختبارات وRuff ودفتر المراجعة."
        if passed
        else "اكتمل التقييم وتعثّر فحص جودة واحد على الأقل. التفاصيل في سجلات التجربة."
    )
    status = (
        "**مقبول مع قيود للأغراض البحثية.**"
        if passed
        else "**تقييم المرشح معلّق حتى معالجة فحوص الجودة.**"
    )
    global_table = "\n".join(
        f"| {row['metric']} | {row['test']:.4f} |" for row in comparison["global_metrics"]
    )
    per_class_table = "\n".join(
        f"| {code} | {row['precision']:.4f} | {row['recall']:.4f} | "
        f"{row['ap50']:.4f} | {row['ap50_95']:.4f} |"
        for code, row in classes.items()
    )
    diagnostics_table = "\n".join(
        f"| {code} | {row['evaluated_annotations']:,} | "
        f"{row['false_positives']:,} | {row['false_negatives']:,} |"
        for code, row in classes.items()
    )
    sizes_table = "\n".join(
        f"| {row['size'].title()} | {row['annotations']:,} | "
        f"{row['detection_rate']:.1%} | {row['false_negative_rate']:.1%} |"
        for row in sizes
    )
    comparison_table = "\n".join(
        f"| {row['metric']} | {row['validation']:.4f} | {row['test']:.4f} | "
        f"{row['difference']:+.4f} ({row['percentage_point_difference']:+.2f} نقطة مئوية) |"
        for row in comparison["global_metrics"]
    )
    ap_table = "\n".join(
        f"| {row['class']} | {row['validation_ap50_95']:.4f} | {row['test_ap50_95']:.4f} | "
        f"{row['difference']:+.4f} |"
        for row in comparison["per_class_ap50_95"]
    )
    return f"""# Stage 6 — التقييم النهائي على مجموعة الاختبار

## الهدف من المرحلة

تقدير قدرة YOLO26s على التعميم على صور لم تُستخدم في التطوير، بتمرير استدلال نهائي واحد
على مجموعة الاختبار المجمدة بعد تثبيت النموذج والإعدادات.

## النموذج النهائي المستخدم

استُخدمت أوزان YOLO26s من `best.pt` في Stage 4 عند الحقبة **{metadata["checkpoint_epoch"]}**.
طابقت الإعدادات سجل Stage 5، وتطابقت بصمات التدريب والتحقق والاختبار والأرشيفات الخام
وبصمة الأوزان قبل التقييم وبعده. اقتصر التنفيذ على الاستدلال والتحليل والتوثيق.

## إعدادات التقييم

| الإعداد | القيمة |
|---|---|
| Model | YOLO26s |
| Checkpoint | `{result["checkpoint"]}` |
| imgsz | {protocol["imgsz"]} |
| Confidence | حد التقييم {conf_floor}، وعدّ FP وFN عند {conf_diagnostic:.2f} |
| IoU | NMS ‏{protocol["nms_iou"]:.2f}، والمطابقة التشخيصية ‏{protocol["match_iou"]:.2f} |
| device | CUDA {protocol["device"]}، {protocol["environment"]["gpu"]} |
| Dataset fingerprint | `{protocol["snapshot"]}` |
| Batch / precision | {protocol["batch_size"]} / FP16 |

استُخدم NMS لكل فئة على حدة، بحد أقصى
{freeze["nms"]["max_detections"]} تنبؤ لكل صورة. استُخدم Ultralytics
{protocol["environment"]["ultralytics"]} وPyTorch {protocol["environment"]["torch"]}
وCUDA {protocol["environment"]["cuda_runtime"]}.

## النتائج النهائية

شمل الاختبار **{result["test_images"]:,}** صورةً و**{counts:,}** تعليقًا مقيّمًا.

| Metric | Test |
|---|---:|
{global_table}

## النتائج حسب الفئة

| Class | Precision | Recall | AP50 | AP50–95 |
|---|---:|---:|---:|---:|
{per_class_table}

تُعرض مقاييس Ultralytics عند نقطة التقييم التي يعتمدها. أعداد الأخطاء أدناه
تستخدم المطابقة الأحادية نفسها في Stage 5 عند ثقة ≥ 0.25 وIoU ≥ 0.50.

| Class | Annotations | FP | FN |
|---|---:|---:|---:|
{diagnostics_table}

## الأداء حسب حجم العيب

استُخدمت حدود Stage 5 نفسها. تُحسب المساحة بضرب عرض الصندوق المطبّع في ارتفاعه،
وتضم الفئة الصغيرة مساحة لا تتجاوز {coords["small_max_inclusive"]:.3%} من مساحة الصورة.
تضم الفئة المتوسطة ما فوق الحد الأول حتى {coords["medium_max_inclusive"]:.3%}،
وتضم الفئة الكبيرة ما يزيد على الحد الثاني.

| Size | Annotations | Detection Rate | Miss Rate |
|---|---:|---:|---:|
{sizes_table}

## مقارنة Validation و Test

استُخدمت قيم Stage 4 المنشورة بأربع منازل عشرية للمقارنة العامة. الفرق هو Test ناقص Validation.

| Metric | Validation | Test | Difference |
|---|---:|---:|---:|
{comparison_table}

| Class | Validation AP50–95 | Test AP50–95 | Difference |
|---|---:|---:|---:|
{ap_table}

## أهم الملاحظات

- **التعميم.** بقي الأداء الإجمالي مستقرًا تقريبًا مع انخفاض طفيف. بلغ فرق mAP50–95
  {gap:+.4f}، أي {gap * 100:+.2f} نقطة مئوية. مقدار الانخفاض المطلق {abs(gap):.4f}.
  حققت {strongest} أعلى AP50–95 وجاءت {weakest} في المرتبة الأخيرة.
- **العيوب الصغيرة.** تكرر نمط الفقد في الاختبار. فات النموذج {test_miss["small"]:.1%}
  من العيوب الصغيرة مقابل {test_miss["large"]:.1%} من الكبيرة، مقارنةً بـ
  {val_miss["small"]:.1%} و{val_miss["large"]:.1%} في تحقق Stage 5.
- **الحفر.** سجّلت D40 في الاختبار Recall ‏{pothole["recall"]:.4f}
  وAP50–95 ‏{pothole["ap50_95"]:.4f} مقابل {pothole_val["recall"]:.4f}
  و{pothole_val["ap50_95"]:.4f} في التحقق. عند عتبة التشخيص، سُجّل
  {pothole["false_negatives"]:,} FN من {pothole["evaluated_annotations"]:,} تعليقًا للحفر.
- **أخطاء الصناديق.** سُجّل {errors["false_positives"]:,} FP و{errors["false_negatives"]:,} FN.
  ظهرت {localization:,} حالة اقتران من الفئة الصحيحة دون حد IoU المطلوب.
  كانت {most_missed["class"]} الأعلى في عدد الحالات المفقودة ({most_missed["false_negatives"]:,}).
  ظهر {background:,} تعليقًا في عمود الخلفية. أبرز الالتباسات كانت {top_confusions}.
- **النشر في المركبة.** بلغ زمن الكشف الدفعي نحو {pipeline_ms:.1f} ms لكل صورة
  على بطاقة الاختبار. تتطلب جاهزية المركبة قياسًا مستقلًا لجودة التقاط الكاميرا وضبابية
  الحركة وسرعة السيارة ودقة GPS وإزالة تكرار الأحداث ومحرك الشدة والشبكة وزمن API
  والتخزين في قاعدة البيانات وسلوك لوحة المعلومات.

## الملفات المستخدمة في Stage 6

| الملف / المسار | الوظيفة الأساسية | أهم الاستدعاءات أو الاستخدامات |
|---|---|---|
| `scripts/evaluate_stage6.py` | نقطة الدخول | `main()` و`--finalize` |
| `src/roadguard/evaluation/stage6.py` | التثبيت والاختبار | `evaluate_test_once()` |
| `src/roadguard/evaluation/stage5.py` | التقاط نتائج الاستدلال | `capture_predictions()` |
| `src/roadguard/evaluation/evaluator.py` | تشغيل المقيم | `validate_checkpoint()` |
| `src/roadguard/evaluation/stage5_errors.py` | المطابقة والحجم | `review_image()` |
| `src/roadguard/evaluation/stage6_report.py` | التقرير والجودة | `finalize_test()` |
| `src/roadguard/data/prepare.py` | سلامة اللقطة | `verify_snapshot()` |
| `{run}/dataset.yaml` | واصف اللقطة | `split=test` و`verify_descriptor()` |
| `notebooks/06_final_test_evaluation.ipynb` | مراجعة العرض | قراءة النتائج والرسومات المحفوظة |

## مخرجات المرحلة

حُفظت المخرجات داخل [مجلد الاختبار النهائي](../{run}/).

- `metadata.json` و`frozen_protocol.json` و`preflight.json` و`integrity.json` وسجلات التنفيذ.
- `final_test_metrics.json` و`final_test_metrics.csv` و`per_class_metrics.json`.
- `validation_vs_test_comparison.json` و`size_analysis.json` و`error_statistics.json`.
- `test_capture.jsonl` و`predictions.csv` و`false_positives.csv` و`false_negatives.csv`.
- `figures/confusion_matrix.png` و`BoxPR_curve.png` و`galleries/` ونسخة دفتر المراجعة المنفّذة.

## القرار النهائي لمرحلة الكشف

{status} {gate_label}
تظل العيوب الصغيرة والاستدعاء الإجمالي أهم القيود. المرحلة التالية الموصى بها هي
دمج النظام والتحقق الميداني بموافقة مستقلة، مع الاحتفاظ بالنموذج والإعدادات المثبتة.
"""

"""Render the university report from the measured Stage 5 evidence."""


def arabic_report(result: dict, metadata: dict) -> str:
    """Write Stage 5 conclusions using counts, metrics and reviewed visual cues."""
    run = result["run"]
    old = metadata["stage4_validation_metrics"]
    metrics = result["validation_metrics"]
    protocol = metadata["identity"]
    threshold = protocol["operating_confidence"]
    match_iou = protocol["match_iou"]
    rows = result["per_class"]
    table = "\n".join(
        f"| {r['class']} | {r['precision']:.4f} | {r['recall']:.4f} | {r['ap50']:.4f} | "
        f"{r['ap50_95']:.4f} | {r['false_positives']:,} | {r['false_negatives']:,} |"
        for r in rows
    )
    annotation_counts = "، ".join(f"{r['class']} ({r['evaluated_annotations']:,})" for r in rows)
    by_code = {row["class"]: row for row in rows}
    strongest, weakest = result["strongest_ap_class"], result["weakest_ap_class"]
    most_missed = by_code[result["highest_miss_rate_class"]]
    false_positives, false_negatives = result["false_positives"], result["false_negatives"]
    tp = result["true_positives"]
    missed_in_matrix = result["annotations_to_background"]
    total_annotations = result["evaluated_annotations"]
    confusions = result["confusion_matrix"]
    confusion_count = result["matched_at_wrong_class"]
    pair_00_20 = confusions["D00"]["D20"] + confusions["D20"]["D00"]
    pair_00_10 = confusions["D00"]["D10"] + confusions["D10"]["D00"]
    pair_20_40 = confusions["D20"]["D40"] + confusions["D40"]["D20"]
    locations = result["fp_status"]["localization_error"]
    duplicates = result["fp_status"]["duplicate_detection"]
    unmatched = result["fp_status"]["unmatched"]
    weak_matches = result["misses_with_same_class_match_below_threshold"]
    fp_low = result["confidence_analysis"]["false_positive"]["operating_below_0_50"]
    sizes = {row["size"]: row for row in result["size_analysis"] if row["class"] == "All"}
    small, medium, large = (sizes[k]["detection_rate"] for k in ("small", "medium", "large"))
    small_rate = sizes["small"]["false_negative_rate"]
    large_rate = sizes["large"]["false_negative_rate"]
    bounds = result["size_definition"]
    audit = result["visual_audit"]
    fp_reviewed, fn_reviewed = audit["n"]["FP"], audit["n"]["FN"]
    fp_cracks = audit["counts"]["FP"].get("visible_crack_like_line", 0)
    fn_thin = audit["counts"]["FN"].get("thin_or_small", 0)
    fn_low_contrast = audit["counts"]["FN"].get("low_contrast", 0)
    quality = (
        "اجتازت Stage 5 فحوص سلامة البيانات والأرشيف وpytest وفحصَي Ruff وتنفيذ دفتر المراجعة."
        if result["status"] == "passed"
        else "تعثّر فحص جودة واحد على الأقل. تتضمن سجلات التجربة تفاصيل الفحوص."
    )
    recommendation = (
        "يوصى بإجراء التقييم النهائي في Stage 6 بعد موافقة صريحة، "
        "مع تثبيت الأوزان والإعدادات والإبلاغ عن الأداء حسب الفئة والحجم."
        if result["stage6_recommended"]
        else "يُؤجّل قرار Stage 6 حتى معالجة فحوص الجودة المتعثرة."
    )
    return f"""# Stage 5 — تحليل أخطاء التحقق Validation Error Analysis

## الهدف من المرحلة

تحديد مواضع نجاح وإخفاق YOLO26s على مجموعة التحقق المجمدة، وتوثيق حدود أدائه قبل التقييم النهائي.

## النموذج المستخدم

استُخدمت نقطة الحفظ `best.pt` من Stage 4 عند الحقبة **{metadata["checkpoint_epoch"]}**.
كانت نتائجها المسجّلة mAP50–95 ‏**{old["map50_95"]:.4f}** وmAP50 ‏**{old["map50"]:.4f}**
وPrecision ‏**{old["precision"]:.4f}** وRecall ‏**{old["recall"]:.4f}**.
أعاد تمرير Stage 5 على RTX 4070 Laptop GPU قياس mAP50–95
عند **{metrics["map50_95"]:.4f}** وmAP50 عند **{metrics["map50"]:.4f}**.

## ما تم تنفيذه

- فُحصت بصمة اللقطة المجمدة وبصمات ملفات التحقق والأرشيفات الخام و`best.pt`.
  طابقت خريطة الفئات D00 ‏Longitudinal Crack وD10 ‏Transverse Crack
  وD20 ‏Alligator Crack وD40 ‏Pothole. بقيت مجموعة الاختبار معزولة عن التحميل والتقييم.
- استُخدم `imgsz={protocol["imgsz"]}` و`batch={protocol["batch_size"]}` وFP16،
  مع `NMS IoU={protocol["nms_iou"]}` وحد تقييم أدنى `{protocol["evaluation_confidence_floor"]}`.
  أُوقفت محاولة CPU المؤقتة بعد عودة CUDA، واستُخدم تمرير GPU الكامل في النتائج.
- حُللت **{result["validation_images"]:,}** صورة و**{total_annotations:,}** تعليقًا،
  وحُفظت الصناديق والثقة والمطابقة لكل تنبؤ. عند ثقة ≥ {threshold:.2f} وIoU ≥ {match_iou:.2f}
  من الفئة نفسها، سُجّل **{tp:,} TP** و**{false_positives:,} FP** و**{false_negatives:,} FN**.
- حُلّل الالتباس وأخطاء تحديد الموقع والحجم والثقة. فُحصت قصاصات مختارة ببذرة ثابتة
  وصُنفت الإشارات المرئية مع إبقاء الحالات غير الواضحة. رُتبت المشاهد الصعبة تشخيصيًا
  بدرجة `2 × FN + FP`، مع ترتيب الحالات المتعادلة بأعلى ثقة FP ثم أكبر مساحة مفقودة.
- بعد التحليل، طابقت بصمات التدريب والتحقق والأرشيفات والأوزان
  وجرد المسارات والأحجام وأوقات التعديل.

## النتائج حسب الفئة

| Class | Precision | Recall | AP50 | AP50–95 | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
{table}

أعداد التعليقات المقيمة هي {annotation_counts}. جاءت Precision وRecall وAP من Ultralytics.
استُخرجت أعداد FP وFN عند عتبة التشغيل الثابتة {threshold:.2f} بالمطابقة الأحادية.
تصدّرت **{strongest}** مقياس AP50–95، وجاءت **{weakest}** أخيرًا.
كانت **{most_missed["class"]}** الأعلى في معدل الفقد ({most_missed["false_negative_rate"]:.1%})
وفي عدد الحالات المفقودة ({most_missed["false_negatives"]:,}).

## أهم الأخطاء المكتشفة

- **الحجم.** حُسبت مساحة كل صندوق من حاصل ضرب عرض الصندوق المطبّع في ارتفاعه المطبّع.
  حدّا الثلثين التجريبيين في التحقق هما {bounds["small_max_inclusive"]:.3%}
  و{bounds["medium_max_inclusive"]:.3%} من مساحة الصورة. بلغت نسب الكشف
  {small:.1%} للصغير و{medium:.1%} للمتوسط و{large:.1%} للكبير. بلغ الفقد
  {small_rate:.1%} للصغير مقابل {large_rate:.1%} للكبير.
- **موضع الصندوق.** رُصدت {locations:,} حالة من الفئة الصحيحة بقي فيها IoU بين
  {result["localization_min_iou"]:.2f} و{match_iou:.2f} دون بلوغ حد المطابقة.
  وسُجّل {duplicates:,} تنبؤًا مكررًا و{unmatched:,} تنبؤًا غير مقترن من إجمالي FP.
- **الالتباس.** وُضع {missed_in_matrix:,} من التعليقات ({missed_in_matrix / total_annotations:.1%})
  في عمود الخلفية، وسُجّل {confusion_count} التباس بين الفئات. كان الزوج الأبرز D00 وD20
  ({pair_00_20} حالة، منها {confusions["D00"]["D20"]} من D00 إلى D20).
  وسُجّل {pair_00_10} التباسًا بين D00 وD10 و{pair_20_40} بين D20 وD40.
- **الثقة.** كان {fp_low:,} من أصل {false_positives:,} FP ({fp_low / false_positives:.1%})
  دون ثقة 0.50. وظهر لـ{weak_matches:,} من أصل {false_negatives:,} FN
  تطابق من الفئة نفسها دون عتبة التشغيل. قد ترتفع FP عند خفض العتبة، لذا بقيت عند {threshold:.2f}.
- **المراجعة البصرية.** ظهرت خطوط تشبه الشقوق في {fp_cracks} من {fp_reviewed} عينات FP.
  تضمنت عينات أخرى علامة طريق أو فاصل رصف أو تباينًا ضوئيًا.
  في عينات FN كانت {fn_thin} من {fn_reviewed} رفيعة أو صغيرة بصريًا،
  وكانت {fn_low_contrast} منخفضة التباين. هذه أوصاف للعينة.
  تُظهر المطابقة حجم أخطاء النموذج في تحديد الموقع، وتستدعي بعض الخطوط غير المقترنة
  مراجعة تغطية الوسوم. تحديد سبب كل خطأ يتطلب أدلة إضافية.

## أمثلة بصرية

يتضمن [مجلد المعارض](../{run}/galleries/) أمثلة منفصلة للكشف الصحيح وFP وFN وتحديد الموقع
والالتباس والمشاهد الصعبة. تتضمن
[عينات المراجعة](../{run}/visual_audit/observations.csv) أوصاف الإشارات المرئية لكل قصاصة.
توضح [مصفوفة الالتباس](../{run}/figures/confusion_matrix.png)
و[تحليل الحجم](../{run}/figures/object_size.png) توزيع حالات الفقد.

## الملفات المستخدمة في Stage 5

| الملف / المسار | الوظيفة الأساسية | أهم الاستدعاءات أو الاستخدامات |
|---|---|---|
| `scripts/analyze_stage5.py` | نقطة الدخول | `main()` و`--analyze` و`--finalize` |
| `src/roadguard/evaluation/stage5.py` | الفحص والتقاط التنبؤات | `capture_validation()` |
| `src/roadguard/evaluation/evaluator.py` | التحقق | `validate_checkpoint()` و`detector.val()` |
| `src/roadguard/evaluation/metrics.py` | مقاييس Ultralytics | `detection_metrics()` |
| `src/roadguard/evaluation/box_matching.py` | مطابقة الصناديق | `box_iou()` و`match_indices()` |
| `src/roadguard/evaluation/stage5_errors.py` | الأخطاء والحجم | `review_image()` |
| `src/roadguard/evaluation/stage5_figures.py` | الرسومات | `make_galleries()` |
| `src/roadguard/evaluation/stage5_review.py` | الأرشيف | `finalize_analysis()` |
| `src/roadguard/data/prepare.py` | سلامة التحقق | `verify_snapshot()` و`write_dataset_yaml()` |
| `{result["checkpoint"]}` | أوزان الحقبة 49 | `load_detector()` |
| `{run}/dataset.yaml` | واصف التحقق | `verify_descriptor()` و`split=val` |
| `notebooks/05_validation_error_analysis.ipynb` | دفتر العرض | قراءة ملفات Stage 5 والرسومات |

## مخرجات المرحلة

حُفظت المخرجات داخل `{run}/`.

- `metadata.json` و`preflight.json` و`integrity.json` وسجلات التنفيذ والجودة.
- `validation_capture.jsonl` و`predictions.csv` و`false_positives.csv` و`false_negatives.csv`.
- `validation_metrics.json` و`per_class_metrics.csv` و`error_statistics.json`.
- `confusion_matrix.csv` و`size_analysis.csv` و`confidence_analysis.json` و`hard_examples.csv`.
- `figures/` و`galleries/` و`visual_audit/` ونسخة منفّذة من دفتر المراجعة.

## قرار Stage 5

{quality} أبرز ضعف هو فقد العيوب الصغيرة، وكانت D00 الأكثر فقدًا بين الفئات.
{recommendation}
"""

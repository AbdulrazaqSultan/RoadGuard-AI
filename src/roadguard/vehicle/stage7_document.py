"""Render the Arabic Stage 7 report from the measured archive."""


def arabic_report(result: dict, metadata: dict, stage6: dict) -> str:
    run = result["run"]
    runtime = result["runtime"]
    scores = stage6["metrics"]
    settings = metadata["identity"]["config"]["vehicle"]
    track = settings["association"]
    dedup = settings["duplicates"]
    device = metadata["identity"]["device"]
    precision = metadata["predict_precision"]
    passed = result["status"] == "partial_pass"
    status = "**PARTIAL PASS**" if passed else "**FAIL**"
    quality = (
        "نجحت الفحوص البرمجية والتحقق من المخرجات ودفتر المراجعة."
        if passed
        else "تعثّر فحص جودة واحد على الأقل. تُبيّن سجلات التشغيل تفاصيله."
    )
    names = {
        "throughput": "YOLO26s · أقصى سرعة",
        "realtime": "YOLO26s · محاكاة الزمن الحقيقي",
        "event_fixture": "اختبار أحداث بكشوف مبرمجة",
    }
    rows = []
    for mode, row in runtime.items():
        ratio = row["duplicate_reduction_ratio"]
        reduction = f"{ratio:.1%}" if ratio is not None else "غير معرّف"
        gpu_bytes = row["gpu_peak_allocated_bytes"]
        vram = f"{gpu_bytes / 2**30:.2f} GiB" if gpu_bytes is not None else "غير مقاس"
        fps = f"{row['source_fps']:.2f}" if row["source_fps"] is not None else "غير متاح"
        latency = row["mean_total_frame_ms_excluding_intentional_wait"]
        rows.append(
            f"| {names[mode]} | {fps} | {row['processed_frames']} | {latency:.2f} | "
            f"{row['effective_fps']:.2f} | {row['raw_detections']} | {row['final_events']} | "
            f"{reduction} | {vram} |"
        )
    table = "\n".join(rows)
    if result["source_kind"] == "synthetic_software":
        fixture = runtime["event_fixture"]
        late = runtime["realtime"]["realtime_late_frames"]
        time_mode = (
            f"تجاوز {late} إطارًا موعده بأكثر من مدة إطار واحدة."
            if late
            else "لم يتجاوز أي إطار موعده بأكثر من مدة إطار واحدة."
        )
        observed = (
            "عولج فيديو MP4 اصطناعي مولّد من أشكال هندسية في وضعي التشغيل. "
            f"انتهى التشغيلان عند EOF. في محاكاة الزمن الحقيقي {time_mode}\n"
            f"أنتج اختبار الأحداث المبرمج {fixture['raw_detections']} كشفًا، "
            f"ودمج {fixture['merged_detections']} مشاهدة متكررة واستبعد "
            f"{fixture['discarded_short_track_detections']} كشفًا عابرًا. "
            f"حُفظ {fixture['final_events']} حدث للحفر مع صور الدليل.\n"
            "انتهى تشغيل YOLO26s على المشهد الاصطناعي بصفر كشوف عند العتبة المثبّتة. "
            "لذلك جاءت أمثلة الدمج وصور الأحداث من الاختبار المبرمج."
        )
        location = (
            "لم يُستخدم موقع جغرافي في تشغيل YOLO26s. استُخدم مسار برمجي تجريبي "
            "فقط في اختبار الأحداث، وحملت كل نقطة الوسم `simulated=true` "
            "والمصدر `simulated_route`."
        )
        output = (
            "- `synthetic_input.mp4` و`throughput/annotated_video.mp4` "
            "و`realtime/annotated_video.mp4`.\n"
            "- `event_fixture/annotated_video.mp4` و`event_fixture/evidence/` "
            "و`event_fixture/crops/`."
        )
        speed_note = (
            "سُجّل إحماء إطار واحد لكل وضع YOLO26s بصورة منفصلة قبل القياس. "
            "تشمل السرعة الفعلية القراءة والاستدلال والربط وكتابة الفيديو والانتظار المقصود. "
            "زمن الإطار في الجدول يستثني انتظار المحاكاة. قياس اختبار الأحداث يستعمل كاشفًا مبرمجًا."
        )
    else:
        source = metadata["identity"]["source"]
        observed = f"عُولج المصدر المقدم `{source}`. يتطلب القرار الميداني مراجعة تشغيل المركبة."
        location = (
            f"كان وضع الموقع `{settings['location']['mode']}`. يحمل المسار التجريبي وسمًا صريحًا."
        )
        output = "- ملفات الفيديو والأحداث والأدلة داخل مجلد وضع التشغيل."
        speed_note = "تشمل سرعة التشغيل القراءة والاستدلال والتسجيل ووقت الانتظار الفعلي."
    small = stage6["comparison"]["test_miss_rate_by_size"]["small"]
    return f"""# Stage 7 — خط معالجة المركبة في الزمن الحقيقي والتحقق الميداني

## الهدف من المرحلة

دمج الكاشف النهائي في خط معالجة مستمر يحوّل إطارات فحص الطريق إلى أحداث عيوب ذات دليل مصوّر.

## النموذج المستخدم

استُخدم YOLO26s من `best.pt` في Stage 4، الحقبة **49**. بلغت نتائج Stage 6
mAP50–95 ‏{scores["map50_95"]:.4f} وmAP50 ‏{scores["map50"]:.4f}
وPrecision ‏{scores["precision"]:.4f} وRecall ‏{scores["recall"]:.4f}.
بقي حجم الاستدلال 640 وحد الثقة 0.25 وNMS IoU عند 0.70.

## معمارية النظام

```text
Camera / Video → Frame source → YOLO26s → Temporal association
→ Duplicate suppression → Evidence selection → Timestamp
→ Location provider → Severity provider → JSONL / CSV event
```

يستخدم ملف الفيديو وفهرس الكاميرا الواجهة `FrameSource` ومسار الاستدلال نفسه.
يمكن إضافة مصدر RTSP عبر الواجهة نفسها. يشير الرمز `T` في الفيديو إلى المسار المؤقت،
ويُكتب معرّف الحدث النهائي مستقلًا في سجلات الأحداث.

## ما تم تنفيذه

- فُصلت قراءة الإطارات والاستدلال والربط الزمني والتسجيل. تُحفظ أبعاد كل إطار وتوقيته
  ومعرّفات المسارات والصناديق والثقة في `frame_detections.jsonl`.
- يحفظ المسجل إطارًا كاملًا واحدًا لكل حدث، مع قصاصة عند تفعيلها.
  يُختار أعلى كشف ثقةً، ويُقدّم الإطار الأسبق عند التعادل.
- أُعيد فتح الفيديو الناتج وفك ترميز إطاراته، وطوبقت سجلات JSONL وCSV
  وعدد الكشوف والإطارات وأدلة الأحداث. تطابقت بصمات الأوزان والأرشيفات
  وبصمة اللقطة المجمدة وجردها. ظل العمل في نطاق الاستدلال والتحقق.

## منع تكرار العيوب

يُربط كل كشف بكشف سابق من الفئة نفسها ضمن {track["max_gap_frames"]} إطارات، بمطابقة أحادية.
تُفضّل المطابقة عند IoU ≥ {track["min_iou"]:.2f}. البديل هو قرب المركز المطبّع
حتى {track["max_center_distance"]:.3f} ونسبة مساحة لا تتجاوز {track["max_area_ratio"]:.1f}.
يمكن ضم مسار جديد إلى حدث منتهٍ من الفئة نفسها ضمن {dedup["max_gap_frames"]} إطارات
عند IoU ≥ {dedup["min_iou"]:.2f}. يصدر الحدث بعد {dedup["minimum_observations"]} مشاهدات على الأقل.
تعني نسبة خفض التكرار المشاهدات المدمجة مقسومةً على الكشوف الخام، وتُحسب الكشوف العابرة على حدة.

## الموقع الجغرافي

{location}

## الخطورة

تُسجّل `Unknown` في جميع الأحداث. تعبّر ثقة YOLO عن تنبؤ الكاشف،
وتقدير أولوية الإصلاح مسؤولية مكوّن خطورة مستقل.

## قياسات الأداء

جرت قياسات YOLO26s على `{device.upper()}` بدقة `{precision}`.

| التشغيل | FPS المصدر | الإطارات | زمن الإطار ms | FPS فعلي | الكشوف | الأحداث | الدمج | VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---|
{table}

{speed_note}

## نتائج التحقق الميداني

{observed}

الفيديو الاصطناعي تحقق برمجي عبر ملف فيديو. ما زال التحقق بكاميرا حية في ظروف قيادة فعلية معلقًا،
بما يشمل ثبات الكشف بين الإطارات وفي السرعات المختلفة وتأثير الظلال والاهتزاز وضبابية الحركة.
لم تُستخدم صور Stage 6 بوصفها لقطات ميدانية.

## الملفات المستخدمة في Stage 7

| الملف / المسار | الوظيفة الأساسية | أهم الاستدعاءات أو الاستخدامات |
|---|---|---|
| `scripts/run_vehicle_pipeline.py` | اختيار التشغيل | `--video` و`--camera` و`--software-check` |
| `configs/vehicle/default.yaml` | إعدادات المركبة | قيم الربط والدمج والموقع وحفظ الفيديو |
| `configs/vehicle/stage7.yaml` | ملف النموذج | `load_config()` |
| `src/roadguard/vehicle/source.py` | التقاط الإطارات | `OpenCVSource.read()` |
| `src/roadguard/inference/detector.py` | استدلال YOLO26s | `Detector.predict_frame()` |
| `src/roadguard/vehicle/tracking.py` | الربط والدمج | `TemporalAssociator` و`DuplicateSuppressor` |
| `src/roadguard/vehicle/events.py` | الحدث والدليل | `RoadDefectEvent` و`EventRecorder` |
| `src/roadguard/vehicle/pipeline.py` | التشغيل والقياسات | `run_pipeline()` |
| `src/roadguard/vehicle/software_fixture.py` | اختبار الأحداث | `ScriptedDetections` |
| `src/roadguard/vehicle/review.py` | فحوص الأرشيف | `verify_outputs()` |
| `notebooks/07_real_time_vehicle_pipeline_review.ipynb` | عرض النتائج | قراءة المخرجات المحفوظة |

## المخرجات

حُفظت المخرجات في [أرشيف Stage 7](../{run}/).

{output}
- `events.jsonl` و`events.csv` و`frame_detections.jsonl` داخل كل تشغيل.
- `metadata.json` و`pipeline_config.json` و`runtime_metrics.json` و`integrity.json` والسجلات.
- نسخة منفّذة من دفتر المراجعة.

## القيود

بلغ فقد العيوب الصغيرة في اختبار Stage 6 نسبة {small:.1%}. تظل آثار ضبابية الحركة
وتغيّر منظور الكاميرا والظلال غير مقاسة ميدانيًا، وقد ينقسم العيب الواحد بعد انقطاع طويل
أو يتداخل مساران متقاربان. يتطلب GPS الحقيقي وتقدير الخطورة وعتاد المركبة تحققًا مستقلًا.
سرعة هذا التشغيل القصير تخص المعالجة التتابعية على الجهاز المقاس.

## قرار Stage 7

{status} {quality}
اكتمل خط المعالجة برمجيًا، وبقي التحقق الميداني الفعلي معلقًا.
الخطوة التالية الموصى بها هي تشغيله على فيديو طريق مستقل أو كاميرا مركبة معروفة المصدر،
وربط GPS فعلي وتقييم التكرار وثبات الكشف قبل الانتقال إلى التكامل الخارجي.
"""

# Training and Evaluation

## Environment and snapshot

Use Python 3.11 or 3.12 and a matching CUDA-enabled PyTorch and TorchVision pair for training. Install that pair for your NVIDIA driver first, then install RoadGuard AI and confirm the device and TorchVision NMS work.

```shell
python -m pip install -e ".[acquisition,eda,notebooks,detection,dev]"
python scripts/check_environment.py
python scripts/download_unified.py
python scripts/extract_unified.py --config configs/data/unified.yaml
python scripts/prepare_dataset.py --config configs/data/unified.yaml
python scripts/analyze_dataset.py --config configs/data/unified.yaml
```

The [Unified Road Defect Dataset](https://huggingface.co/datasets/TamAko783/Unified_Road_Defect_Dataset/tree/7ba2309c243f989d72a6e9a8a9aca902871232ce) combines RDD2022, UAV-PDD2023, and RoadDamageVision. Download, verify, extract, and prepare it locally. The deterministic split targets 80/10/10 and the recorded snapshot contains 24,147 training, 3,019 validation, and 3,019 test images. The preparation profile records a reviewed exact duplicate and keeps recognized related captures together. Physical-defect identity across independent views remains a limitation.

## Review and reproduce training

Open [the baseline notebook](../notebooks/02_detection_baseline_training.ipynb) to verify CUDA, review the dataset and model setup, and launch training explicitly. The default is review mode. The notebook imports the training modules from `src/roadguard/training/`. Point it at a locally archived run to load the epoch metrics, training curves, per-class AP, precision-recall curves, confusion matrix, checkpoint details, and sample predictions.

Use the baseline configuration and the same prepared snapshot for the YOLO11n and YOLO11s runs. Store trusted pretrained weights under `artifacts/models/pretrained/` as referenced in the profiles. Replace `<snapshot>` with the path printed by dataset preparation.

```shell
python scripts/train_detector.py --config configs/training/stage3_baseline.yaml --dataset "<snapshot>"
python scripts/train_detector.py --config configs/training/stage3_baseline.yaml --dataset "<snapshot>" --set model.model_variant=s --set model.weights=artifacts/models/pretrained/yolo11s.pt --set experiment.experiment_name=stage3_yolo11s_challenger
python scripts/compare_detector_runs.py --baseline "<baseline-run>" --challenger "<challenger-run>"
```

The notebooks in `notebooks/` also cover model screening, controlled long training, error analysis, final evaluation, and video processing. Each script writes the run configuration and results to `artifacts/experiments/<run>/`. Some later course-stage scripts expect the archived inputs, weights, and fingerprints from earlier stages. Reproduce them in sequence or supply the verified local archive. Keep the test partition untouched until model and settings are selected using validation.

## Held-out results

The frozen YOLO26s checkpoint was evaluated on 3,019 test images. Overall precision was **68.66%**, recall was **57.67%**, mAP50 was **63.64%**, and mAP50–95 was **31.70%**.

| Defect | Test annotations | AP50 | AP50–95 |
| --- | ---: | ---: | ---: |
| D00 · Longitudinal crack | 3,092 | 61.36% | 31.70% |
| D10 · Transverse crack | 1,921 | 59.17% | 27.92% |
| D20 · Alligator crack | 1,182 | 66.37% | 34.54% |
| D40 · Pothole | 1,249 | 67.68% | 32.62% |

AP aggregates performance across confidence thresholds. The detector is suitable for research with known misses and false positives. The evaluation does not establish physical severity or automated maintenance readiness.

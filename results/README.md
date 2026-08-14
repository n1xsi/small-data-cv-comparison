# Results

Metrics from the original runs, transcribed from their logs. Everything here was
produced on a Colab T4; nothing in this directory is generated at import time or
by the test suite.

```
classification/
  metrics.csv              four models × {val, test}
detection/
  metrics.csv              both detectors, the columns the CLI writes
  yolo_per_class.csv       YOLOv8n broken down by class
  dfine_coco_stats.csv     D-FINE's full 12-stat pycocotools block
figures/                   plots from the same runs
```

Re-running `scripts/train_classifier.py` or `scripts/train_detector.py` overwrites
`metrics.csv` in place with the same schema. The per-class and COCO-stat files are
transcriptions and are not written by the CLI.

## Two things to know before quoting these numbers

**D-FINE has no test row.** Its evaluation config was derived from the training
config by a string replacement that silently matched nothing, so `--test-only`
re-read the validation split. The bug is fixed in `write_dfine_configs` and covered
by `tests/test_dfine_configs.py`, but the recorded number is a validation number and
is labelled as one. Compare it against YOLOv8n's `val` row, not its `test` row.

**Precision and recall are blank for D-FINE.** pycocotools reports AP and AR at
various IoU thresholds and detection budgets, not the single precision/recall pair
Ultralytics gives. The closest analogue, AR@100 = 0.606, is in
`dfine_coco_stats.csv` — it is not the same quantity as YOLO's `recall` column and
is deliberately not placed there.

Sample sizes are small: 300 test images for classification, 61 for detection, 60 for
validation. Differences of a few points are noise. See
[../docs/results.md](../docs/results.md) for what the numbers do and do not support.

# Results

Full metrics for both experiments, with interpretation. For experimental design
and implementation reasoning see [methodology.md](methodology.md).

Every number here comes from the runs recorded in `results/*/metrics.csv`, with
per-class and full COCO breakdowns alongside them. See
[../results/README.md](../results/README.md) for provenance and two caveats about
what the detection columns mean.

## Contents

- [Part 1: classification](#part-1-classification)
- [Part 2: detection](#part-2-detection)
- [What the two parts say together](#what-the-two-parts-say-together)
- [Limitations](#limitations)

---

## Part 1: classification

### Dataset

2,000 images, 1,000 per class, split 1,400 / 300 / 300. The validation split holds
exactly 150 cats and 150 dogs, confirming stratification held.

Classical baselines work on 64×64 grayscale (4,096 features when flattened); the
neural models take 160×160 RGB.

### Validation split (n = 300)

| Model | Accuracy | Macro F1 | ROC-AUC |
|---|---|---|---|
| MobileNetV2 (transfer learning) | 0.9867 | 0.9867 | 0.9999 |
| Small CNN (from scratch) | 0.6667 | 0.6659 | 0.6995 |
| HOG + LinearSVC | 0.5633 | 0.5633 | 0.6505 |
| Raw pixels + LogisticRegression | 0.5233 | 0.5218 | 0.5000 |

### Test split (n = 300)

| Model | Accuracy | Macro F1 | ROC-AUC | Avg. precision |
|---|---|---|---|---|
| MobileNetV2 (transfer learning) | 0.9833 | 0.9833 | 0.9988 | 0.9988 |
| HOG + LinearSVC | 0.6067 | 0.6065 | 0.6459 | 0.6260 |
| Small CNN (from scratch) | 0.5667 | 0.5651 | 0.6256 | 0.6692 |
| Raw pixels + LogisticRegression | 0.4900 | 0.4893 | 0.5062 | 0.4970 |

HOG grid search selected `C = 0.1` with a cross-validated macro F1 of 0.6097 on
the training split — close to its eventual test score of 0.6065, so the search was
not optimistic.

### Reading the numbers

**Chance is 0.50.** Three of four approaches sit between 0.49 and 0.61 macro F1 on
test. Only one is a working model.

**Raw pixels achieved nothing.** Test macro F1 0.4893 and ROC-AUC 0.5062 — the
ROC-AUC in particular is the tell. 0.506 means the model's confidence carries
essentially no information about the correct answer; it is not a weak classifier,
it is not a classifier. This is the expected outcome and the point of including
it: a flattened image forces the model to learn a weight per absolute pixel
position, and any shift of the subject changes the whole feature vector.

**HOG's advantage was smaller than expected.** On validation, HOG scored 0.5633
against raw pixels' 0.5218 — a gain of 0.041. That is not the decisive improvement
the standard framing of HOG suggests. The gap is clearer on test (0.6065 vs
0.4893), but the honest summary is that HOG moved a useless model to a barely
usable one.

The reason is specific to this problem. HOG encodes gradient orientation and
discards absolute intensity — good for detecting pedestrians against varied
backgrounds, where silhouette is the signal. Cats and dogs share silhouette, fur
texture and pose. What separates them includes colour and fine facial structure,
and HOG throws colour away by construction. A feature descriptor is only as good
as its match to the problem, and this is a mismatch.

**The from-scratch CNN lost to HOG on test.** 0.5651 vs 0.6065 macro F1 — and it
had led on validation (0.6659 vs 0.5633). A model that leads on validation and
trails on test is overfitting the selection process, which is exactly what limited
data produces. Note also the shape of its errors: ROC-AUC 0.6256 with average
precision 0.6692, higher than HOG's on both counts, meaning its *ranking* is
somewhat better than its *decisions*. The threshold is poorly placed.

More capacity with no more data bought nothing. 1,400 images cannot supply the
gradient signal a convolutional stack needs to learn useful filters from random
initialisation.

**Transfer learning changed the problem.** 0.9833 test macro F1, ROC-AUC 0.9988.
The gap to the next-best model is 0.3768 macro F1 — not an incremental gain but a
different regime.

Worth being precise about what changed. MobileNetV2 is a convolutional network,
like the from-scratch CNN that scored 0.5651. The architecture family is the same.
What differs is that its features were already learned, on a million images this
project never touched. Freezing the backbone and training only the head means the
1,400 available images were spent on a much easier problem: not "learn to see", but
"learn to separate two classes given features that already describe images well".

The validation-to-test drop is 0.0034 macro F1 — essentially nothing, which is
what a genuinely generalising model looks like.

### Grad-CAM: implemented, not yet run

There is no heatmap to report. The run that produced these numbers went as far as
resolving the target layer — `mobilenetv2_1.00_160`, the last 4-D layer inside the
nested backbone — and stopped there. `plot_gradcam_examples` in
`cvlab.classification.explain` is complete and will render the maps, but it has not
been executed over the test split.

This is the gap worth naming, because a 98% score on a small dataset is consistent
with two very different situations: the model reads the animal, or the model found
a photographic regularity in how these particular images were collected. Metrics
cannot separate those, and the second one collapses on any new data source. Until
the heatmaps exist, 0.9833 is a number without an explanation behind it.

---

## Part 2: detection

### Dataset

401 images annotated by hand, 201 per class, carrying 465 boxes: 244 cat and 221
dog objects, about 1.16 objects per image. 30 boxes extended past the image frame
and were clipped rather than discarded. Split 280 / 60 / 61.

Both detectors trained 80 epochs at 640×640 from random initialisation, evaluated
at confidence threshold 0.40.

### YOLOv8n, from scratch

| Split | mAP50-95 | mAP50 | Precision | Recall |
|---|---|---|---|---|
| Validation | 0.3033 | 0.4805 | 0.4457 | 0.5545 |
| Test | 0.2587 | 0.4799 | 0.4532 | 0.6792 |

### D-FINE-n, from scratch

Validation split (n = 60).

| Metric | Value |
|---|---|
| mAP50-95 | 0.114 |
| mAP50 | 0.251 |
| mAP75 | 0.076 |
| AR100 | 0.606 |

**No test-split number exists for D-FINE.** The run generated its evaluation
config by string-editing the training config to swap `img_folder` and `ann_file`
onto the test split, but those keys live in the dataset config, not the model
config — the replacement matched nothing and returned the string unchanged. So
`--test-only` loaded the validation set twice. The two evaluations in the original
notebook are byte-identical, down to `'TPs': 6, 'FPs': 11, 'FNs': 66`, which is
what confirmed it.

`write_dfine_configs` now builds the `val_dataloader` block from a parameterised
template that restates `img_folder`/`ann_file` per split, and
`tests/test_dfine_configs.py` asserts the generated test config names the test
split and nothing else. Re-running would produce the missing number; the table
above reports what was measured.

Every D-FINE comparison below is therefore against YOLOv8n's **validation** row
(mAP50-95 0.3033, mAP50 0.4805), not its test row.

### Reading the numbers

**mAP50 held between splits; mAP50-95 did not.** YOLOv8n scored 0.4805 and 0.4799
mAP50 on validation and test — stable. But mAP50-95 fell from 0.3033 to 0.2587.

The two metrics differ only in localisation strictness: mAP50 accepts any box with
IoU ≥ 0.5, while mAP50-95 averages across thresholds up to 0.95. So the model finds
objects about equally well on both splits and places *looser* boxes on the test
split. With 61 test images, a handful of awkward poses or partial occlusions moves
this number, and box-tightness noise from annotation lands here too.

**Recall exceeded precision, by a lot.** 0.6792 vs 0.4532 on test. The model
proposes more boxes than it should and is over-confident about the wrong ones. That
is a calibration problem, not a blindness problem — and it means raising the
confidence threshold to clean up false positives would cost recall the model cannot
spare.

**How the detector fails was inspected, not counted.** The original run looked at
examples of each error type — a missed animal, two boxes on one animal, a cat
labelled dog — and did not tally them, so this section makes no claim about which
type dominates. `classify_errors` in `cvlab.detection.compare` produces the counts;
they have not been run over the test split.

What the metrics do support is the calibration reading above. The distinction the
taxonomy draws still matters for what a count would mean: cat-versus-dog confusion
would point at the classification head, missed objects at insufficient data, and
duplicate boxes at NMS and threshold settings rather than at the model's
understanding. Those are three different fixes behind one mAP penalty.

**D-FINE reached about a third of YOLOv8n's mAP50-95 on the same split.** 0.114 vs
0.3033 on validation, and mAP50 0.251 vs 0.4805.

The most informative number in D-FINE's table is the contrast between AR100 (0.606)
and mAP75 (0.076). Average recall at 100 detections says it *does* find objects —
comparably often to YOLOv8n, in fact. mAP75 says it cannot place tight boxes on
them. The failure is localisation precision, not detection.

This is what the architectural difference predicts. YOLO's anchor grid and
convolutional structure encode "objects are compact regions at various scales"
before training starts. DETR-style set prediction discards that and learns
object-query-to-region assignment from data, which is a good trade at COCO scale
(118,000 images) and a bad one at 280. The transformer spent its limited data
learning what YOLO was given for free.

Nothing here says transformer detectors are worse. It says they are worse *in this
regime*, for a reason that is specific and predictable.

**D-FINE inference ran at roughly 34 FPS on a T4** — real-time, which is the
practical argument for these architectures regardless of the accuracy comparison.
Measured on new images at confidence 0.40; YOLOv8n's throughput was not recorded.

---

## What the two parts say together

Both experiments varied how much prior knowledge a model brought and held the data
roughly fixed. Both produced the same ordering.

In part 1, the model with pretrained features beat the best model without them by
0.3768 macro F1. In part 2, the model with hand-built architectural priors beat the
one that learns them by 2.7× mAP50-95 on the shared validation split. Different
tasks, different frameworks, same mechanism: **inductive bias substitutes for data.**

The practical consequence is a decision rule. On a dataset of a few hundred to a
few thousand images, the question "which architecture is best?" matters far less
than "what can I start from?" — a pretrained backbone, or an architecture whose
built-in assumptions already fit the problem. Architecture search on 2,000 images
optimises the wrong variable.

The corollary is that the numbers in these tables do not transfer. A
from-scratch CNN scoring 0.565 here says nothing about from-scratch CNNs in general;
it says something about 1,400 images.

---

## Limitations

**Small samples.** 300 test images for classification, 61 for detection, 60 for
validation. One misclassification moves classification macro F1 by roughly 0.003;
on detection, single-image effects are visible in the 0.045 mAP50-95 gap between
validation and test. Differences of a few points in these tables are not reliable.
The differences carrying the conclusions — 0.3768 macro F1 and a 2.7× mAP ratio —
are large enough to survive that noise.

**The detector comparison rests on one split.** D-FINE has a validation number and
no test number, for the reason given above, so the two detectors are compared on
the split that also drove early stopping. That is weaker evidence than a held-out
comparison would be. It is not fatal here — the gap is 2.7× and YOLOv8n's own
validation-to-test movement is 0.045 — but the number to quote is a validation
number, and it is quoted as one.

**Two claims are not backed by measurement.** No Grad-CAM heatmaps were produced,
so nothing is asserted about what the classifier attends to. No error counts were
tabulated, so nothing is asserted about whether missed objects or misclassifications
dominate. Both are implemented and cheap to run; neither was run.

**Single run per model.** Seeds are fixed at 42 for reproducibility, which is not
the same as knowing the variance. No error bars are reported because none were
measured. Re-running with several seeds would be the first thing to add.

**One annotator per image.** No inter-annotator agreement was measured. Box
tightness is annotator-dependent and mAP50-95 is directly sensitive to it, so some
of the validation-to-test mAP50-95 drop is plausibly annotation noise rather than
model behaviour.

**The transfer-learning result is favourably biased.** ImageNet contains many cat
and dog breeds, so MobileNetV2's features are close to ideal for this specific
task. 0.983 should be read as an upper bound on what transfer learning offers, not
a typical figure. Classes absent from ImageNet would show a smaller gap.

**From-scratch detection numbers are low by design.** Loading COCO weights would
raise both detectors substantially — COCO includes cats and dogs. The comparison
deliberately gives up absolute performance to isolate architecture, so neither
number should be read as what these detectors can do.

**No OpenCV.** The stack is Pillow, scikit-image, scikit-learn, Keras, Ultralytics
and PyTorch. Stated so the dependency list is not read as something it is not.

# Cats vs Dogs: Classification and Detection on a Small Dataset

[![CI](https://github.com/USERNAME/cats-vs-dogs-cv/actions/workflows/ci.yml/badge.svg)](https://github.com/USERNAME/cats-vs-dogs-cv/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

*[Русская версия](README.ru.md)*

Two experiments on the same two classes, run to answer one question: **how much
does pretraining matter when you only have a small dataset?**

Part 1 compares four classifiers on 2,000 images. Part 2 compares two detectors
on 401 hand-annotated images. Every model is ranked by the same metric on the same
split, held out from training and from hyperparameter choice.

The short answer: on a dataset this size, transfer learning is not an incremental
improvement over training from scratch — it is the difference between a usable
model and an unusable one.

---

## Results

### Part 1 — classification (test split, n = 300)

| Model | Accuracy | Macro F1 | ROC-AUC | Avg. precision |
|---|---|---|---|---|
| **MobileNetV2 (transfer learning)** | **0.983** | **0.983** | **0.999** | **0.999** |
| HOG + LinearSVC | 0.607 | 0.607 | 0.646 | 0.626 |
| Small CNN (from scratch) | 0.567 | 0.565 | 0.626 | 0.669 |
| Raw pixels + LogisticRegression | 0.490 | 0.489 | 0.506 | 0.497 |

Random guessing on this balanced problem scores 0.50. Read the table that way and
three of the four approaches are close to worthless: the raw-pixel baseline is
indistinguishable from a coin flip, and neither HOG nor the from-scratch CNN gets
past 0.61. MobileNetV2 reaches 0.983 — a 0.376 macro-F1 gap over the next-best
model, from a backbone that was never trained on this dataset.

### Part 2 — detection

| Model | Split | mAP50-95 | mAP50 | Precision | Recall |
|---|---|---|---|---|---|
| **YOLOv8n (from scratch)** | test (n = 61) | **0.259** | **0.480** | 0.453 | 0.679 |
| YOLOv8n (from scratch) | validation (n = 60) | 0.303 | 0.481 | 0.446 | 0.554 |
| D-FINE-n (from scratch) | validation (n = 60) | 0.114 | 0.251 | — | 0.606 (AR100) |

D-FINE is reported on validation because that is the split it was actually
measured on — the original run generated its evaluation config by string-editing
the training one, and the edit silently did nothing, so `--test-only` re-read the
validation set. The bug is fixed in `write_dfine_configs` and covered by a test,
but the recorded numbers are what they are. Compare it against YOLOv8n's
validation row (0.303 vs 0.114) rather than the test row.

Neither detector loaded pretrained weights, so this compares architectures under
equal and severe data starvation. On validation, YOLOv8n's mAP50-95 is 2.7× and
its mAP50 1.9× D-FINE's. The transformer is not the weaker architecture in general — it is
the weaker architecture *here*, because DETR-style set prediction has to learn
object localisation that YOLO's anchor grid and convolutional priors provide for
free. 280 training images is nowhere near enough to learn it.

Both parts point the same way: **inductive bias substitutes for data.** Every
result in these tables is a measurement of how much prior knowledge the model
brought with it.

---

## What is in here

| Part | Task | Approaches |
|---|---|---|
| 1 | Binary classification | Raw pixels, HOG features, CNN from scratch, MobileNetV2 transfer learning |
| 2 | Object detection | YOLOv8n and D-FINE-n, both from random initialisation |

Beyond the models themselves:

- **Grad-CAM** on the transfer-learned classifier, to check whether a 98% model
  looks at the animal rather than a background cue — implemented and ready to
  run; see [Findings](#findings) for why it has not been run yet
- **A manual annotation pipeline** — Label Studio, COCO export, verified
  conversion to YOLO format
- **An error taxonomy** for detection: false negatives, duplicate detections and
  misclassifications counted separately, because they call for different fixes
- **A reusable package** (`src/cvlab/`) rather than notebook-only code, with
  tests over the parts that fail silently

---

## Quick start

```bash
git clone https://github.com/USERNAME/cats-vs-dogs-cv.git
cd cats-vs-dogs-cv

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -e ".[classification]"     # part 1
pip install -e ".[detection]"          # part 2
pip install -e ".[dev]"                # tests and linting
```

Place your data as described in [`data/README.md`](data/README.md), then:

```bash
# Part 1 — train everything and write the comparison table
python scripts/train_classifier.py --model all

# Or one approach at a time
python scripts/train_classifier.py --model hog
python scripts/train_classifier.py --model mobilenet

# Part 2 — convert the COCO export, then train
python scripts/prepare_detection.py
python scripts/train_detector.py --model yolov8

# Inference on new images
python scripts/predict.py --task classify \
    --weights results/classification/weights/mobilenet.keras \
    --source data/new_images

python scripts/predict.py --task detect \
    --weights runs/yolo/yolov8n_scratch/weights/best.pt \
    --source data/new_images
```

Every run writes metrics to `results/` as CSV, so tables and charts can be rebuilt
without retraining. Settings live in `configs/*.yaml`; any of them can be
overridden from the command line.

---

## Method

### Evaluation

**Macro F1 is the primary metric**, not accuracy. Accuracy on a balanced dataset
hides which class a model is failing on; macro F1 weights both classes equally,
so a model cannot look competent by favouring the majority.

The split is stratified 70/15/15. The validation split guides model selection and
early stopping; **the test split is touched exactly once per model**, at the end.
The HOG grid search cross-validates on the training split only, so hyperparameter
choice never sees held-out data either. Splitting happens before any feature
extraction or scaling, so no statistic computed on validation or test data can
leak into training.

Seeds are fixed at 42 across `random`, NumPy, TensorFlow and PyTorch. The
detection split is derived deterministically from the COCO export, so both
detectors see byte-identical data.

### Part 1 — four approaches, increasing prior knowledge

Ordered by how much they already know about images:

1. **Raw pixels + logistic regression** — 64×64 grayscale flattened to 4,096
   features. No spatial structure at all: a linear model on this input has no way
   to know two adjacent pixels are related. This is the honest floor.
2. **HOG + linear SVM** — hand-designed gradient-orientation histograms, with `C`
   chosen by 3-fold cross-validated grid search (best: `C = 0.1`, CV macro F1
   0.610). A real prior about edges and shape, designed by hand.
3. **Small CNN from scratch** — three conv blocks, global average pooling instead
   of a wide dense head, dropout 0.3, light augmentation. The convolutional prior
   is now learned rather than designed.
4. **MobileNetV2 transfer learning** — frozen ImageNet backbone, then fine-tuning
   from layer 100 at `lr = 1e-5`. Two stages matter: training the head first
   avoids large early gradients destroying the pretrained features, and the low
   fine-tuning rate keeps them intact while adapting them.

Augmentation is deliberately conservative — horizontal flip, ±5% rotation, ±10%
zoom. On ~1,400 training images, aggressive augmentation costs more than it
returns, and vertical flips are not realistic for photographs of animals.

### Part 2 — two detectors, no pretrained weights

401 images annotated by hand: 465 boxes, 244 cat and 221 dog objects, ~1.16
objects per image. 30 boxes extended past the image frame and were clipped rather
than dropped. Split 280 / 60 / 61.

COCO JSON is the single source of truth. It is audited once, split once, then
written out in both layouts — COCO for D-FINE, normalised text labels for
Ultralytics. The bbox conversion (`[x_min, y_min, w, h]` in pixels → `cx cy w h`
normalised) is verified by reading the labels back off disk, because getting it
wrong is the classic silent failure in detection work: training runs perfectly
and mAP sits near zero.

**YOLOv8n** is built from `yolov8n.yaml`, not `yolov8n.pt` — a `.pt` file would
load COCO weights and make the comparison meaningless. **D-FINE-n** uses an
HGNetv2-B0 backbone with `pretrained: false`. 80 epochs each, 640×640, confidence
threshold 0.40. D-FINE inference on new images ran at roughly 34 FPS on a T4.

---

## Findings

**HOG barely beat raw pixels on validation.** 0.563 vs 0.522 macro F1 — a gain of
0.041, which is not the decisive improvement the textbook framing of HOG would
suggest. It only separates clearly on the test split (0.607 vs 0.489). Cats and
dogs share silhouette, fur texture and pose; HOG discards colour, which is one of
the few cues that actually distinguishes them. A feature descriptor is only as
good as its fit to the specific problem.

**The from-scratch CNN lost to HOG on the test split.** 0.565 vs 0.607 macro F1.
More capacity with no more data buys nothing: 1,400 images cannot supply the
gradient signal a convolutional stack needs to learn useful filters from random
initialisation. It fit the training set and generalised worse than a hand-designed
descriptor with one tuned hyperparameter.

**Only pretraining broke the ceiling.** The same convolutional architecture family
that scored 0.565 from scratch reached 0.983 with ImageNet weights. The
architecture did not change — what changed is that the features were already
learned, from a million images this project never had access to.

**Grad-CAM is wired up but was not used to validate this result.** `cvlab.classification.explain`
resolves the target layer inside the nested MobileNetV2 submodel and renders
heatmaps, and the original run went as far as confirming the layer it would
attach to. It never produced the maps. So the 98% is unaudited: a model can hit
that on a small dataset by exploiting a photographic regularity rather than the
animal, and the metric alone cannot tell you which one you have. Running
`plot_gradcam_examples` over the test split is the first thing this project needs.

**The recall/precision gap is the clearest signal in the detection errors.**
YOLOv8n scores 0.679 recall against 0.453 precision on test: the model finds
objects but its confidence is poorly calibrated, so raising the threshold trades
away recall it cannot afford. The error taxonomy in `cvlab.detection.compare`
separates missed objects, duplicate boxes and misclassifications — they call for
different fixes — but the original run only inspected examples of each rather
than counting them, so no claim about which type dominates is made here.

**The transformer detector underperformed, as expected under these conditions.**
D-FINE reached mAP50-95 0.114 on validation against YOLOv8n's 0.303 on the same
split, and mAP75 of 0.076 — it localises loosely. Its AR100 of 0.606 shows it does
find objects; it just cannot place tight boxes on them. DETR-style architectures
trade hand-built priors for learned set prediction, which is a good trade with
COCO-scale data and a bad one with 280 images.

### Honest limitations

- **2,000 classification images and 401 detection images** is small. Differences
  of a few points here are not reliable; the differences that carry the argument
  (0.376 macro F1, 2× mAP) are large enough to survive that.
- **Single run per model, no confidence intervals.** Seeds are fixed for
  reproducibility, not to estimate variance.
- **A single annotator per image, no inter-annotator agreement.** Box tightness is
  a real source of noise in the mAP50-95 numbers.
- **The MobileNetV2 result is not a general claim about cats and dogs.** ImageNet
  contains many cat and dog breeds, so this transfer task is unusually favourable.
  Expect a harder time on classes ImageNet does not cover.
- **No OpenCV in these two experiments.** The stack is Pillow, scikit-image,
  scikit-learn, Keras, Ultralytics and PyTorch. Named here so the dependency list
  is not mistaken for something it is not.

---

## Project layout

```
├── src/cvlab/                  # the package — all logic lives here
│   ├── common/                 # config, seeding, IO, plotting
│   ├── classification/         # part 1: data, features, models, evaluation, Grad-CAM
│   └── detection/              # part 2: COCO, conversion, YOLO, D-FINE, comparison
├── scripts/                    # CLI entry points
│   ├── train_classifier.py
│   ├── prepare_detection.py
│   ├── train_detector.py
│   └── predict.py
├── notebooks/                  # narrative walkthroughs, importing from src/
├── configs/                    # classification.yaml, detection.yaml
├── tests/                      # pytest suite, no GPU or heavy frameworks needed
├── docs/                       # methodology.md, results.md
└── data/                       # not committed — see data/README.md
```

Notebooks import from `src/cvlab` rather than redefining logic, so a fix lands in
one place and the notebook narrative stays readable.

## Development

```bash
pip install -e ".[dev]"

pytest -q                                   # 70 tests
ruff check src scripts tests
ruff format src scripts tests
python tests/check_notebooks_clean.py       # fails on committed notebook outputs
```

The test suite deliberately avoids TensorFlow and PyTorch. It covers bbox
geometry, COCO auditing, dataset splitting, the error taxonomy and config
validation — the code where a bug produces plausible-looking wrong numbers instead
of an exception. CI runs the same commands on Python 3.11.

Install [nbstripout](https://github.com/kynan/nbstripout) before committing
notebooks:

```bash
pip install nbstripout && nbstripout --install
```

## Reproducing the detection experiments

D-FINE has no Python API — it is a research repository driven by YAML configs and
a CLI trainer. `scripts/train_detector.py --model dfine` generates the configs and
shells out to it, so clone it first:

```bash
git clone https://github.com/Peterande/D-FINE third_party/D-FINE
pip install -r third_party/D-FINE/requirements.txt
```

Total training time was about 1.5 GPU-hours on a T4 across both detectors.

## References

- Dalal & Triggs, *Histograms of Oriented Gradients for Human Detection*, CVPR 2005
- Sandler et al., *MobileNetV2: Inverted Residuals and Linear Bottlenecks*, CVPR 2018
- Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks via
  Gradient-based Localization*, ICCV 2017
- Carion et al., *End-to-End Object Detection with Transformers*, ECCV 2020
- Peng et al., *D-FINE: Redefine Regression Task of DETRs as Fine-grained
  Distribution Refinement*, 2024
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)

## Authorship

Coursework for the Intelligent Systems course at SUAI. This repository is a
rewrite of that work — the experiments were originally Colab notebooks, and the
package structure, tests, CI and documentation here were written for this release.

## License

[MIT](LICENSE)


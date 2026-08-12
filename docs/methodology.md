# Methodology

Detailed notes on experimental design, the reasoning behind each choice, and the
mistakes worth avoiding. For results and their interpretation see
[results.md](results.md).

## Contents

- [Experimental design](#experimental-design)
- [Part 1: classification](#part-1-classification)
- [Part 2: detection](#part-2-detection)
- [Reproducibility](#reproducibility)

---

## Experimental design

### Why macro F1

Accuracy is the default reflex and the wrong primary metric even here, on a
balanced dataset. Accuracy compresses both classes into one number, so a model
that gets every cat right and half the dogs wrong reports the same 0.75 as one
that fails evenly. Macro F1 averages the per-class F1 scores with equal weight,
which surfaces exactly that asymmetry.

Accuracy is still reported, because it is the number most readers have intuition
for. ROC-AUC and average precision are computed where a continuous score exists,
since they describe ranking quality independent of the decision threshold — useful
when a model is well-ordered but badly calibrated.

### Splitting discipline

Stratified 70/15/15, split once, with three separate roles:

- **train** — fits parameters
- **validation** — model selection, early stopping, hyperparameter choice
- **test** — touched once per model, at the very end

Two leakage paths are closed explicitly:

1. **Preprocessing leakage.** The split happens before feature extraction and
   before any scaler is fitted. `StandardScaler` inside a `Pipeline` sees training
   folds only, so no validation or test statistic reaches the training distribution.
2. **Selection leakage.** The HOG grid search cross-validates within the training
   split. If it had searched over validation scores, the reported validation number
   would be an optimistic estimate of the search itself rather than of the model.

Stratification matters at this scale. On 300 test images an unstratified split can
easily land 60/40 by chance, which shifts every metric enough to change conclusions.

### Class imbalance

The classification dataset is balanced by construction, but `class_weight="balanced"`
and the Keras `class_weight` dict are wired in anyway. The detection data *is*
mildly imbalanced (244 cat vs 221 dog objects). Handling it uniformly costs nothing
and removes one variable from the comparison.

---

## Part 1: classification

### The four approaches as a ladder

The models are not four arbitrary choices; they form an ordered sequence of
increasing prior knowledge about images. Each step adds exactly one thing:

| Step | Model | Prior added |
|---|---|---|
| 1 | Raw pixels + LogisticRegression | none |
| 2 | HOG + LinearSVC | hand-designed gradient/edge structure |
| 3 | Small CNN | learned convolutional structure |
| 4 | MobileNetV2 | convolutional structure *plus* ImageNet features |

Reading the results as a ladder is what makes them interpretable: the gap between
any two rows attributes performance to one specific ingredient.

### 1. Raw pixels

64×64 grayscale, flattened to 4,096 features, `StandardScaler(with_mean=False)`
then logistic regression.

`with_mean=False` is deliberate — centring a 4,096-column matrix offers nothing
here and would force densification. Only per-feature variance is normalised.

This baseline exists to be beaten. A flattened image has no notion that pixel
(10, 10) neighbours (10, 11); the model must learn a weight per absolute pixel
position. Any translation of the subject changes the entire feature vector. It
scores near chance, and that is the honest floor everything else is measured from.

### 2. HOG + linear SVM

HOG with 9 orientations, 8×8 cells, 2×2 blocks, L2-Hys normalisation.

HOG throws away absolute intensity and keeps the local distribution of gradient
orientations. That buys partial invariance to lighting and small shifts, and
encodes shape rather than appearance.

`C` is tuned by 3-fold cross-validated grid search over `{0.1, 0.3, 1, 3, 10}`,
scored by macro F1. The best value was `C = 0.1` — strong regularisation, which is
what you expect when features outnumber samples.

HOG parameters live in a single `HOG_PARAMS` dict shared by the extractor and the
visualiser. Keeping them in one place prevents the failure where the picture in the
report shows different features from the ones the model was trained on.

### 3. Small CNN from scratch

Three conv blocks (32/64/128 filters, 3×3, ReLU, max pooling), dropout 0.3, global
average pooling, dense softmax head.

Two deliberate choices:

- **Global average pooling instead of flatten + dense.** A flatten on a
  20×20×128 feature map into even a 128-unit dense layer costs ~6.5M parameters.
  GAP costs zero and regularises by construction. With ~1,400 training images,
  parameter count is the binding constraint.
- **Conservative augmentation** — horizontal flip, ±5% rotation, ±10% zoom. Vertical
  flips are unphysical for animal photographs, and heavy augmentation on a small
  dataset mostly adds noise to an already weak gradient signal.

Early stopping on validation loss with `patience=3` and `restore_best_weights=True`.
Without the restore flag you keep the *last* weights rather than the best ones,
which quietly reports a worse model than you trained.

### 4. MobileNetV2 transfer learning

Two stages, and the order matters.

**Stage 1 — frozen backbone, train the head.** `lr = 1e-3`, 5 epochs. The head
starts random, so its initial gradients are large. Applying them through an
unfrozen backbone would destroy the pretrained features before they were used.

**Stage 2 — fine-tune the top, `lr = 1e-5`.** Unfreeze from layer 100 up. Early
layers encode generic edges and textures that transfer as-is; only the deeper,
more task-specific layers are worth adapting. The learning rate drops by 100× for
the same reason as stage 1: the goal is to nudge good features, not to relearn them.

Two subtleties that silently degrade results if missed:

- **`backbone(x, training=False)`** keeps frozen BatchNorm layers in inference mode.
  Without it, BN updates its running statistics on the new data even while the
  weights are frozen, which shifts the very features you are trying to preserve.
- **`preprocess_input` must be applied**, and inside the model rather than in the
  loader. MobileNetV2 expects inputs scaled to [-1, 1], not [0, 1] or [0, 255].
  Baking it in means inference cannot forget it.

### Grad-CAM

A 98% model on a small dataset warrants suspicion. It could be reading the animal,
or it could have found that cat photos in this collection are indoors and dog
photos outdoors. Metrics cannot distinguish these.

Grad-CAM weights the final convolutional feature maps by the gradient of the
predicted class with respect to those maps, then keeps the positive part. The
result is a coarse map of the evidence the network actually used.

One implementation detail worth flagging: the target layer lives *inside* the
nested MobileNetV2 submodel, not in the outer classifier. `find_last_conv_layer`
recurses to find it and returns an `"outer/inner"` path. Pointing Grad-CAM at the
outer model's last 4-D layer yields a heatmap of the wrong thing.

**Not yet run.** The original work resolved the layer and stopped; no heatmaps were
generated, so [results.md](results.md) reports no finding from this. The code path
is `plot_gradcam_examples(image_paths, model, class_names, ...)` and
`scripts/train_classifier.py --model mobilenet` invokes it, so producing the maps
needs a trained checkpoint and nothing else.

---

## Part 2: detection

### Annotation

Label Studio, self-hosted via Docker, "Object Detection with Bounding Boxes"
template, exported as COCO.

Guidelines applied consistently: tight boxes around the visible extent of the
animal, occluded parts not included, one box per animal, ambiguous or unclear
images discarded rather than guessed. 401 images survived, carrying 465 boxes —
244 cat and 221 dog objects, ~1.16 objects per image.

Consistency matters more than any single rule chosen. mAP50-95 rewards tight
localisation, so systematically loose boxes depress the metric across the board;
*inconsistently* loose boxes inject noise the model cannot learn around.

### COCO as the single source of truth

The export is audited once, split once, and then written out in both formats. That
ordering is what makes the comparison valid — had each detector's dataset been
built independently, any difference in scores could be a difference in splits.

The audit catches the problems a scraped-and-annotated dataset actually has:

- **Missing images.** Label Studio rewrites `file_name` with an upload hash and URL
  escaping, so naive path joining fails. Images are matched by basename with the
  hash prefix stripped.
- **Out-of-bounds boxes.** 30 boxes extended past the frame. They are clipped to the
  image, not dropped — a box drawn slightly wide is still a valid object.
- **Degenerate boxes.** Zero width or height, or non-finite coordinates, are dropped.
  There is nothing to recover.

### Coordinate conversion

This is where detection projects break silently.

COCO stores `[x_min, y_min, width, height]` in absolute pixels. YOLO wants
`class_id cx cy w h` with the centre and size normalised to [0, 1] by the image
dimensions:

```
cx = (x_min + width / 2)  / image_width
cy = (y_min + height / 2) / image_height
w  = width  / image_width
h  = height / image_height
```

Get this wrong — corner instead of centre, unnormalised, or width/height swapped —
and nothing raises. Training runs to completion and mAP sits near zero, at which
point the natural suspicion is the model or the learning rate rather than the data.

Two guards are in place. `audit_yolo_dataset` reads every written label back off
disk and checks the ranges; a unit test round-trips a known box from COCO through
the writer and back to pixel corners.

One more trap: **every image needs a label file, even with no objects.** Ultralytics
treats a missing `.txt` as an unlabelled image, not a negative example. Empty files
are written explicitly.

### D-FINE integration

D-FINE has no Python API. It is a research repository driven by YAML configs and a
CLI trainer, so integration means generating configs and shelling out.

Two things that cost real debugging time:

- **`num_classes` must exceed the highest category id.** D-FINE reserves index 0 for
  background, so two classes with ids 1 and 2 require `num_classes: 3`. Setting it
  to 2 trains without complaint and produces garbage.
- **Config `__include__` paths resolve relative to the upstream repo's config tree.**
  The generated dataset config has to be written into `configs/dataset/` inside the
  clone, not next to the model config.

And one that cost a result. The evaluation config was originally derived from the
training config by `str.replace`, swapping the validation `img_folder`/`ann_file`
for the test ones. Those keys are not in the model config — they are in the dataset
config — so the replacement matched nothing, returned its input unchanged, and
`--test-only` evaluated validation. Nothing failed; two different runs simply
printed the same numbers.

`write_dfine_configs` now emits the `val_dataloader` block from a parameterised
template that states `img_folder`/`ann_file` explicitly for whichever split it is
building, so the two configs cannot converge. `tests/test_dfine_configs.py` asserts
the test config names the test split, that the val split appears nowhere in it, and
that train and test differ on exactly those two lines. A generated config is plain
text handed to an external process — no import, no type check, no schema — so the
only thing standing between a wrong path and a wrong published number is a test
that reads the file back.

Metrics come back by parsing pycocotools output from the trainer's log. The parser
tries the compact `Averaged stats` line first and falls back to the human-readable
`Average Precision (AP) @[...]` block, since the format varies across versions.

### Why not pretrained weights

Both detectors are trained from random initialisation: YOLOv8n from `yolov8n.yaml`
rather than `yolov8n.pt`, D-FINE with `pretrained: false` on its HGNetv2 backbone.

Loading COCO weights would be the right choice for a production model and the wrong
one for this experiment. COCO contains both cat and dog classes, so pretrained
weights would mean reporting how well someone else's training run transfers, with
the architectures' own inductive biases invisible underneath.

The cost is that both absolute numbers are low. That is the honest price of the
question being asked.

### Error taxonomy

mAP is a single number and hides *how* a detector fails. Predictions are matched
greedily by descending confidence to the highest-IoU unmatched ground-truth box,
then bucketed:

| Bucket | Meaning | What it suggests |
|---|---|---|
| `correct` | matched, right class | — |
| `misclassified` | matched, wrong class | classification head, class confusion |
| `false_positive` | no match, or duplicate of a claimed object | threshold, NMS |
| `false_negative` | ground-truth box nothing covered | recall, more data, augmentation |

The distinction between a duplicate detection and a genuine false positive matters:
duplicates point at NMS and threshold settings, while true false positives point at
the model hallucinating objects. Both land in the same mAP penalty.

The taxonomy is implemented and tested; it has not been run over the test
predictions, so [results.md](results.md) reports no breakdown.

---

## Reproducibility

**Seeds.** Fixed at 42 for `random`, NumPy, TensorFlow and PyTorch. Framework
seeding is lazy and wrapped in `try/except ImportError`, so seeding works whether
or not the heavy frameworks are installed.

**Configuration.** All settings live in `configs/*.yaml`, loaded into dataclasses.
The original notebooks used Colab form fields with absolute `/content/` paths;
config files mean a run reproduces on a laptop, a workstation or a hosted GPU
without editing code. Unknown YAML keys are ignored rather than fatal, and split
fractions are validated to sum to 1.0 at construction time.

**What is not committed.** Datasets, model weights and run directories are
gitignored. Metrics are committed as CSV, so every table and chart in the docs can
be rebuilt without a GPU.

**Remaining nondeterminism.** cuDNN kernel selection and GPU floating-point
reduction order are not fully deterministic. Expect small variation in the third
decimal place between runs on different hardware.

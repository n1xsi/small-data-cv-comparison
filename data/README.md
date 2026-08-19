# Data

Images and annotations are not committed to this repository. Both parts expect
you to place data here yourself; the layouts below are what the scripts look for.

## Part 1 — classification

One subdirectory per class:

```
data/raw/cats_dogs/
  cat/
    cat_0001.jpg
    ...
  dog/
    dog_0001.jpg
    ...
```

Class names come from the directory names, so `configs/classification.yaml`
lists `[cat, dog]` to match. Any source of cat and dog photographs works — the
original experiments used ~1000 images per class collected from open datasets.

`build_dataset_dataframe` decodes every file before training and drops anything
unreadable, so truncated or mislabelled images are reported by filename rather
than crashing an epoch in.

## Part 2 — detection

A COCO export plus the images it references:

```
data/raw/
  annotations/
    result.json        # COCO export from Label Studio
  images/
    img_0001.jpg
    ...
```

`result.json` is the default Label Studio filename; any name works, and
`--coco-json` accepts either a file or a directory to search.

Label Studio rewrites `file_name` with an upload hash and URL escaping. Images
are matched by basename and the prefix is stripped if needed, so the export works
without editing the JSON by hand.

### Annotation workflow

The original annotations were produced in a self-hosted Label Studio:

```bash
docker run -it -p 8080:8080 -v "$(pwd)/label-studio-data:/label-studio/data" \
  heartexlabs/label-studio:latest
```

Create a project with the "Object Detection with Bounding Boxes" template, define
the `cat` and `dog` labels, import the images, draw boxes, then export as
**COCO**. Any tool that emits COCO JSON works equally well.

### Generated layouts

`scripts/prepare_detection.py` reads the export and writes two derived datasets:

```
data/dataset_coco/          # for D-FINE
  images/{train,val,test}/
  annotations/{train,val,test}.json

data/dataset_yolo/          # for Ultralytics
  images/{train,val,test}/
  labels/{train,val,test}/
  dataset.yaml
```

Both are generated from the same split, so the two detectors train on identical
data. Both directories are gitignored — regenerate them rather than committing
them.

## Optional: unlabelled images

`data/new_images/` holds images for qualitative inference only. `scripts/predict.py`
reads from here by default and nothing in it is ever used for training or
evaluation.

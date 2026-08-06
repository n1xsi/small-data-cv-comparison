"""Part 1 — image classification: features, models, evaluation and Grad-CAM."""

from .data import (
    DatasetAudit,
    DataSplits,
    build_dataset_dataframe,
    load_images_as_arrays,
    split_dataset,
)
from .evaluate import (
    EvalResult,
    evaluate_keras_model,
    evaluate_sklearn_model,
    plot_confusion_matrix,
    plot_history,
    plot_model_comparison,
    plot_roc_and_pr,
    show_misclassified,
    summarize,
)
from .explain import find_last_conv_layer, make_gradcam_heatmap, plot_gradcam_examples
from .features import extract_hog_features, flatten_pixels, hog_with_visualisation
from .models import (
    build_hog_svm_grid,
    build_mobilenetv2_transfer,
    build_pixel_baseline,
    build_small_cnn,
    compile_model,
    compute_class_weight_dict,
    early_stopping,
    make_tf_dataset,
    unfreeze_top_layers,
)

__all__ = [
    "DataSplits",
    "DatasetAudit",
    "EvalResult",
    "build_dataset_dataframe",
    "build_hog_svm_grid",
    "build_mobilenetv2_transfer",
    "build_pixel_baseline",
    "build_small_cnn",
    "compile_model",
    "compute_class_weight_dict",
    "early_stopping",
    "evaluate_keras_model",
    "evaluate_sklearn_model",
    "extract_hog_features",
    "find_last_conv_layer",
    "flatten_pixels",
    "hog_with_visualisation",
    "load_images_as_arrays",
    "make_gradcam_heatmap",
    "make_tf_dataset",
    "plot_confusion_matrix",
    "plot_gradcam_examples",
    "plot_history",
    "plot_model_comparison",
    "plot_roc_and_pr",
    "show_misclassified",
    "split_dataset",
    "summarize",
    "unfreeze_top_layers",
]

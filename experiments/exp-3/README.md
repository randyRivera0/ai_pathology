# Experiment 3: group-aware ResNet50 binary colon classifier

This experiment repeats the Experiment 2 ResNet50 protocol while assigning
complete LC25000-clean image groups to train, validation, or test. It is
research and educational work only and must not be used for medical diagnosis
or clinical decision-making.

## Protocol

- Dataset: `andrewmvd/lung-and-colon-cancer-histopathological-images`
- Published groups: `GeorgeBatch/LC25000-clean`
- Classes: benign (`0`) and adenocarcinoma (`1`)
- Split: two-stage `GroupShuffleSplit` within each class, seed 42
- Requested split: 70/15/15; actual image split: 70.83/14.66/14.51
- Input: 224 x 224 RGB
- Backbone: frozen ResNet50 v1 with ImageNet weights
- Head: global average pooling, Dense(128), Dense(64), Dense(1, sigmoid)
- Augmentation: rotation, horizontal/vertical flips, zoom, and translation on
  training images only
- Optimization: Adam at `1e-3`, binary cross-entropy
- Selection: minimum validation loss, maximum 10 epochs, early-stopping
  patience 3
- Decision threshold: 0.5

The exact assignments and group IDs are recorded in
`outputs/group_split_manifest.csv`. Complete groups are disjoint across all
three partitions.

## ResNet50 result

Training stopped after epoch 5 and restored epoch 2, which had validation loss
0.000532. On the untouched 1,451-image, 77-group test partition, the selected
checkpoint produced:

| Metric | Value |
| --- | ---: |
| Accuracy | 0.998622 |
| Precision | 1.000000 |
| Recall | 0.997347 |
| ROC AUC | 0.999991 |
| Loss | 0.003940 |

The confusion matrix contains 1,449 correct and 2 incorrect predictions. Both
errors belong to one test group.

Experiment 2's random image-level split produced 0.995333 accuracy. The grouped
run is 0.003288 higher, so this run found no leakage-associated accuracy drop.
This does not establish external or patient-level generalization.

## Color-only explanatory baseline

A post-hoc baseline summarizes each image with 96 global color features: 32
normalized histogram bins for each RGB channel. A standardized logistic
regression trained on the same group-aware training partition achieved
0.932460 test accuracy, compared with 0.998622 for ResNet50.

This result shows that global color is strongly predictive in this internal
dataset. ResNet50's 0.066161 advantage indicates that richer spatial, texture,
and morphological information remains useful. Neither result is evidence of
clinical validity.

## Artifacts

- `ResNet50.ipynb`: executed experiment and explanatory-baseline record
- `outputs/group_split_manifest.csv`: exact image and group assignments
- `outputs/group_split_metadata.json`: source, method, seed, counts, and limits
- `outputs/resnet50_model_manifest.json`: model contract and checkpoint checksum
- `outputs/resnet50_training_history.csv`: per-epoch metrics
- `outputs/resnet50_test_metrics.json`: selected-checkpoint test metrics
- `outputs/resnet50_classification_report.json`: per-class metrics
- `outputs/resnet50_confusion_matrix*.csv`: raw and normalized matrices
- `outputs/resnet50_test_predictions.csv`: per-image predictions with group IDs
- `outputs/color_histogram_baseline_metrics.json`: color-only baseline metrics
- `outputs/color_histogram_test_predictions.csv`: color-only predictions
- `outputs/*.png`: training and confusion-matrix figures

The 98 MB native Keras checkpoint and the distribution ZIP are intentionally
excluded from Git. The checkpoint SHA-256 is
`cdcd06a7d5cde685bc70eeb391b93b5393806ae952379746c195ca0f66e23691`.

## Limitations

LC25000-clean groups related augmented tiles, but the evaluation is not split
by patient, slide, laboratory, scanner, or hospital. The small number of
independent test groups and strong color separability also limit how broadly
the image-level metrics can be interpreted. External-cohort validation is
required before making any claim about generalization.

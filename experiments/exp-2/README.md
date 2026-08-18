# Experiment 2: ResNet50 binary colon classifier

This experiment compares benign colon tissue with colon adenocarcinoma from
LC25000. It is research and educational work only and must not be used for
medical diagnosis or clinical decision-making.

## Protocol

- Dataset: `andrewmvd/lung-and-colon-cancer-histopathological-images`
- Classes: benign (`0`) and adenocarcinoma (`1`)
- Split: stratified 70/15/15 with seed 42
- Input: 224 x 224 RGB
- Backbone: frozen ResNet50 v1 with ImageNet weights
- Head: global average pooling, Dense(128), Dense(64), Dense(1, sigmoid)
- Optimization: Adam at `1e-3`, binary cross-entropy
- Selection: minimum validation loss, maximum 10 epochs, early-stopping
  patience 3
- Decision threshold: 0.5

The exact assignments are recorded in `outputs/split_manifest.csv`. A later
comparison model must reuse this manifest rather than create another split.

## Result

Training stopped after epoch 6 and restored epoch 3, which had validation loss
0.001990. On the untouched 1,500-image test partition, the selected checkpoint
produced:

| Metric | Value |
| --- | ---: |
| Accuracy | 0.995333 |
| Precision | 0.992053 |
| Recall | 0.998667 |
| ROC AUC | 0.999973 |
| Loss | 0.010694 |

The test confusion matrix contains 1,493 correct and 7 incorrect predictions.

## Artifacts

- `ResNet50.ipynb`: cleaned executed experiment record
- `outputs/split_manifest.csv`: exact reusable split assignments
- `outputs/split_metadata.json`: dataset, seed, class mapping, and counts
- `outputs/resnet50_model_manifest.json`: model contract and checkpoint checksum
- `outputs/resnet50_training_history.csv`: per-epoch metrics
- `outputs/resnet50_test_metrics.json`: selected-checkpoint test metrics
- `outputs/resnet50_classification_report.json`: per-class metrics
- `outputs/resnet50_confusion_matrix*.csv`: raw and normalized matrices
- `outputs/resnet50_test_predictions.csv`: per-image test predictions
- `outputs/*.png`: training and confusion-matrix figures

The 98 MB native Keras checkpoint is intentionally excluded from Git. Its
SHA-256 is recorded in the model manifest so a separately distributed copy can
be verified.

## Limitation

LC25000 expands a smaller collection through augmentation and does not provide
reliable source-group provenance for every released image. This image-level
split may therefore place related derivatives in different partitions. These
results describe internal, in-distribution performance and are not evidence of
patient-level generalization or clinical validity.

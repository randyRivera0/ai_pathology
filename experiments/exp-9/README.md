# Experiment 9: final HISTOPANTUM colorectal model

Experiment 9 corrects the finalization limitation identified in Experiment 8.
It does not compare fold checkpoints or architectures against the test set. It
uses Experiment 8's five-fold case-disjoint cross-validation to lock one model
configuration, trains that configuration once on the complete development
pool, and evaluates it on the same locked holdout used in Experiment 8.

This experiment is for research and educational use only. It is not clinical
validation and must not be used for diagnosis or clinical decision-making.

## Locked protocol

The canonical assignment is copied unchanged from Experiment 8:

| Partition | Cases | Patches |
| --- | ---: | ---: |
| Complete development pool | 32 | 21,801 |
| Reused locked holdout | 8 | 5,447 |

All patches from a TCGA case remain in one partition. The five development
fold labels are retained in `cv_split_assignment.csv` for provenance, but they
are not used to exclude any development case during final training.

The following model-selection decisions come from Experiment 8 cross-validation:

- architecture: ResNet50, based on pooled out-of-fold results;
- frozen-head duration: 2 epochs;
- fine-tuning duration: 3 epochs;
- use of partial fine-tuning rather than a frozen-only final model.

The remaining settings are inherited fixed protocol choices. They were not
optimized from either cross-validation or holdout results:

- augmentation and preprocessing: unchanged from Experiment 8;
- initialization: ImageNet weights;
- learning rates: `1e-3` for the head and `1e-5` for fine-tuning;
- unfreezing: the 22 non-BatchNorm `conv5_*` layers;
- batch size: 32;
- decision threshold: `0.5`;
- random seed: 42;
- case assignment: the existing 32/8 split from Experiment 8.

ResNet50 was selected because it led the pooled out-of-fold accuracy, balanced
accuracy, F1, and ROC-AUC, although its advantage over MobileNetV2 was small.
The two frozen-head epochs are the median best frozen epoch across all five
ResNet50 folds: 2, 2, 3, 5, and 1. Fine-tuning won validation-loss selection in
three folds; their best fine-tuning epochs were 1, 5, and 3, whose median is 3.
This is a fixed, test-independent stopping rule. Experiment 9 therefore has no
validation callbacks, best-fold selection, or post-holdout model comparison.
Although Experiment 8's holdout comparison favored MobileNetV2, Experiment 9
uses ResNet50 because the cross-validation aggregates slightly favored it. This
supports that the architecture choice followed CV rather than the holdout winner.

## Workflow

Run `HISTOPANTUM_Colon_Final_ResNet50_Colab.ipynb` from top to bottom in a GPU
Colab runtime after uploading the dataset and `cv_split_assignment.csv`.
Paths may be supplied through `HISTOPANTUM_COLON_ROOT` and
`HISTOPANTUM_CV_SPLIT_CSV`.

The notebook:

1. verifies the exact 40-case, 27,248-patch inventory;
2. separates all 32 development cases from the locked 8-case test group;
3. trains one ResNet50 on all development patches for the fixed 2+3 schedule;
4. saves the final model before loading or predicting the holdout group;
5. evaluates that locked model at threshold `0.5`;
6. exports predictions, metrics, training history, and an artifact manifest.

Expected compact evidence files are:

- `resnet50_training_history.json`;
- `resnet50_test_predictions.csv`;
- `resnet50_test_metrics.json`;
- `resnet50_model_manifest.json`.

## Executed result

The fixed 2+3 schedule completed without runtime errors. The final training
epoch reached 0.9816 training accuracy and 0.0527 training loss. These are
optimization diagnostics, not generalization estimates, because the final fit
has no validation subset.

| Evaluation | Accuracy | Balanced accuracy | Precision | Recall | Specificity | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Reused 8-case holdout | 0.9218 | 0.8996 | 0.8949 | 0.9906 | 0.8087 | 0.9403 | 0.9907 |

The patch-level confusion matrix is `[[1666, 394], [32, 3355]]`, ordered as
`[[TN, FP], [FN, TP]]`. Metrics recompute exactly from 5,447 unique prediction
rows across the assigned eight cases. The archived final model is 214,693,367
bytes with SHA-256
`6228d60d3c067d01a18c4ca118ba58af5292ecb0749efc525ab2352054fd64ae`.

The `.keras` model and generated archives must remain outside Git. Compact
evidence should be reviewed before any result is documented. Holdout results must
not be used to alter this configuration or select another model; doing so would
further turn the holdout group into validation data.

## Interpretation

The Experiment 9 artifact—not any Experiment 8 fold checkpoint—is the final
model produced by this protocol. Its metrics describe performance for the
locked CV-selected recipe on this small internal reused holdout. The eight cases
were excluded from Experiment 9 training and configuration selection, but they
are not globally untouched: Experiment 8 had already evaluated ResNet50 and
MobileNetV2 fold-0 candidates on the same cases. Therefore, this is not a
pristine one-time final-test estimate. A genuinely unbiased final estimate
requires new patients, ideally from an external institution. These results do
not establish
cross-hospital generalization, independent patch-level sample size, clinical
validity, or diagnostic safety.

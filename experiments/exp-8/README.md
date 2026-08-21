# Experiment 8: HISTOPANTUM colorectal cross-validation

The shared dataset inventory and filename contract are documented in the
[HISTOPANTUM colorectal dataset card](../../docs/datasets/histopantum-colon.md).

This experiment compares ImageNet-pretrained ResNet50 and MobileNetV2 on the
HISTOPANTUM colorectal subset using a case-disjoint 80/20 holdout and five-fold
cross-validation within the 80% development pool. It is for research and
educational use only and is not clinical validation.

## Dataset assignment

The 40 TCGA cases and 27,248 patches are partitioned as follows:

| Partition | Cases | Patches |
| --- | ---: | ---: |
| Five-fold CV pool | 32 | 21,801 |
| Held-out test | 8 | 5,447 |

`cv_split_assignment.csv` is the canonical case assignment. All patches from a
case remain in one group, and the same folds are used for both architectures.
The test cases are excluded from training, checkpoint selection, and
cross-validation summaries.

## Training protocol

Each architecture is initialized independently for every fold. One fold is
held out for validation and the other four CV folds are used for training.
Each of the five runs has two sequential phases:

1. train the classification head while the pretrained backbone is frozen;
2. restore the best frozen checkpoint and partially unfreeze the backbone;
3. fine-tune at a lower learning rate while keeping Batch Normalization frozen;
4. select the frozen or fine-tuned checkpoint using validation loss only.

ResNet50 unfreezes the 22 non-BatchNorm `conv5_*` layers. MobileNetV2 unfreezes
the 20 non-BatchNorm layers from `block_14_expand` onward. This produces ten
fold pipelines and twenty phase checkpoints across both architectures. Model
files are intentionally excluded from Git.

Validation loss selected fine-tuned checkpoints for ResNet50 folds 1-3 and
MobileNetV2 folds 0-3. Frozen checkpoints were selected for ResNet50 folds 0
and 4 and MobileNetV2 fold 4.

## Validation roles and selection bias

The split is case-disjoint: every TCGA case, slide, and associated patch stays
within one group. Thus, no fold trains on patches from a case that appears in
that fold's validation predictions. The separate eight-case holdout is also
excluded from every cross-validation training run.

However, Experiment 8 is grouped cross-validation rather than nested grouped
cross-validation. In each run, the same held-out fold serves two roles:

1. its validation loss controls early stopping and selects the best frozen or
   fine-tuned checkpoint;
2. predictions from that selected checkpoint produce the reported fold score.

This is not cross-case patch leakage, but it is checkpoint-selection reuse. The
reported fold and pooled OOF metrics can therefore be somewhat optimistic and
should not be described as fully selection-independent outer-fold estimates.

A stronger future comparison should use three distinct case-level roles:

1. **Inner training cases:** fit model parameters.
2. **Inner validation cases:** select epochs, checkpoints, training phase,
   threshold, and other hyperparameters.
3. **Outer held-out fold:** score the already selected pipeline only.

After repeating that nested procedure across outer folds, cross-validation
should choose the architecture and training recipe. A new model can then be
trained on the complete development pool using a stopping rule fixed from the
nested results and evaluated once on a separate, never-before-used final test
set. The outer folds estimate the development procedure; they do not replace
that final test set.

## Cross-validation results

Mean and sample standard deviation are calculated across the five held-out
folds. Pooled OOF metrics concatenate each patch's single out-of-fold
prediction. Because patches within a case are correlated, neither aggregation
turns patch count into independent sample count.

| Model | Accuracy | Balanced accuracy | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| ResNet50 fold mean +/- SD | 0.9206 +/- 0.0574 | 0.9086 +/- 0.0744 | 0.9392 +/- 0.0345 | 0.9539 +/- 0.0625 |
| MobileNetV2 fold mean +/- SD | 0.9113 +/- 0.0648 | 0.9024 +/- 0.0744 | 0.9321 +/- 0.0372 | 0.9548 +/- 0.0691 |
| ResNet50 pooled OOF | 0.9209 | 0.9192 | 0.9358 | 0.9747 |
| MobileNetV2 pooled OOF | 0.9156 | 0.9125 | 0.9317 | 0.9726 |

Fold 4 was substantially harder for both architectures: ResNet50 accuracy was
0.8320 with ROC-AUC 0.8482, and MobileNetV2 accuracy was 0.7961 with ROC-AUC
0.8314. The cross-validation result therefore exposes material case/domain
sensitivity that a single favorable split could hide. Neither architecture
dominates consistently across metrics and folds.

## Prespecified held-out test

Fold 0 was fixed before training as the checkpoint source for the one-time test
evaluation; it was not selected because it was the best-performing fold. Each
fold-0 frozen/fine-tuned pair was first resolved by validation loss.

| Model | Selected phase | Accuracy | Balanced accuracy | F1 | ROC-AUC |
| --- | --- | ---: | ---: | ---: | ---: |
| ResNet50 fold 0 | Frozen | 0.8979 | 0.8681 | 0.9235 | 0.9860 |
| MobileNetV2 fold 0 | Fine-tuned | 0.9189 | 0.8961 | 0.9381 | 0.9888 |

MobileNetV2 is the stronger of the two prespecified test-evaluated artifacts,
but the small eight-case test set and cross-fold variability do not establish a
universally superior architecture. A future finalization protocol should lock
the architecture and training choices from CV, retrain on the full development
pool with an internal stopping strategy, and evaluate on new external cases.

## Reproducibility and artifacts

- `HISTOPANTUM_Colon_CrossValidation_Colab.ipynb`: canonical executed Colab run.
- `cv_split_assignment.csv`: deterministic case-level assignment.
- `outputs/*_fold*_predictions.csv`: out-of-fold patch predictions.
- `outputs/*_fold*_case_metrics.csv`: case-level metrics for each held-out fold.
- `outputs/*_test_predictions.csv` and `outputs/*_test_metrics.json`: the
  prespecified fold-0 test evaluations.
- `outputs/*_model_manifest.json`: selected test-model identity, size, and hash.
- `outputs/cross_validation_summary.csv`: reproducible fold and pooled metrics.
- `summarize_results.py`: local evidence validation and summary generation.

Regenerate the summary without loading images or models:

```bash
python experiments/exp-8/summarize_results.py
```

The compact and full ZIP archives are ignored. The compact archive's contents
are committed under `outputs/`; the full archive contains approximately 1.58 GB
of `.keras` checkpoints and must remain outside Git.

## Limitations

- Only 40 cases from one colorectal release are available.
- Patch observations within a case are correlated.
- Fold-level estimates are based on only six or seven cases each.
- The same held-out fold selects checkpoints and supplies the reported fold
  score; nested grouped CV would be required for selection-independent scores.
- Fold 4 demonstrates substantial sensitivity to the held-out case mixture.
- The test evaluation uses a prespecified fold-0 checkpoint trained on only the
  complement of fold 0, rather than a final model retrained on all 32 CV cases.
- Internal discrimination does not demonstrate cross-hospital, external, or
  clinical generalization.

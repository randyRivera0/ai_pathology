# Experiment 10: fast grayscale ResNet50 toy model

Experiment 10 trains a small grayscale-specific ResNet50 for an experimental
RGB/grayscale comparison in the research GUI. It prioritizes a short runtime
and an engaging demonstration over a publication-grade model comparison.

The model accepts an RGB image but performs a fixed luminance conversion inside
the saved Keras graph before classification. The GUI can therefore display the
original image beside its grayscale version while invoking a model that was
actually trained on grayscale inputs.

This experiment is for research and educational use only. The toy model is not
clinically validated and must not be used for diagnosis or clinical
decision-making.

## Patient-clustered split

The 40 HISTOPANTUM cases are assigned deterministically using seed 42 and
tumour-fraction strata:

| Partition | Cases | Percentage | Role |
| --- | ---: | ---: | --- |
| Training | 28 | 70% | Fit model parameters |
| Validation | 6 | 15% | Select the frozen or fine-tuned checkpoint |
| Test | 6 | 15% | One internal toy evaluation |

Every patch from a patient remains in exactly one partition. The canonical
assignment is `toy_split_assignment.csv`.

## Short training protocol

- ImageNet-pretrained ResNet50 at 224 by 224 pixels.
- Fixed model-internal RGB-to-grayscale conversion.
- Three frozen-backbone head epochs at `1e-3`.
- Two partial `conv5_*` fine-tuning epochs at `1e-5`.
- Batch Normalization remains frozen.
- Frozen and fine-tuned checkpoints are compared using validation loss.
- The selected checkpoint is evaluated once on the six test cases at threshold
  `0.5`.

This is one 3+2-epoch training run, not cross-validation.

## Run in Colab

Upload:

- `HISTOPANTUM_Colon_Grayscale_Toy_Colab.ipynb`;
- `toy_split_assignment.csv`;
- the HISTOPANTUM colorectal dataset.

The default split path is `/content/toy_split_assignment.csv`. Dataset paths
can be overridden with `HISTOPANTUM_COLON_ROOT`, and the split path with
`HISTOPANTUM_TOY_SPLIT_CSV`.

Run the notebook from top to bottom in a T4 GPU runtime. Expected runtime is
approximately 15–30 minutes, depending on Colab storage and GPU conditions.

Expected outputs include:

- `grayscale_resnet50_toy.keras`;
- `grayscale_resnet50_toy_manifest.json`;
- `grayscale_resnet50_toy_test_predictions.csv`;
- `grayscale_resnet50_toy_test_metrics.json`;
- `grayscale_resnet50_toy_case_metrics.csv`;
- `grayscale_resnet50_toy_confusion_matrix.png`;
- a compact evidence ZIP without model files.

The `.keras` checkpoints and ZIP archives must remain outside Git.

## GUI presentation

The primary RGB model and this toy grayscale model may be displayed side by
side under:

> Experimental RGB/grayscale comparison

The interface should also state:

> The grayscale model is an experimental visual comparison trained with a
> short internal protocol. Its score is not directly comparable to the primary
> model and is not a diagnosis.

The models use different training protocols and evaluation partitions. Their
scores should not be presented as a fair architecture comparison or used to
claim that one representation is superior.

## Limitations

- This is a deliberately short toy-model training run.
- Only six internal patients are used for testing.
- There is no cross-validation or external validation.
- The validation set selects the checkpoint and training phase.
- Grayscale retains stain-derived intensity and texture information.
- Patch observations within a patient are correlated.
- Results do not establish interpretability, external generalization, clinical
  validity, or diagnostic safety.

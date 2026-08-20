# Experiment 7: HISTOPANTUM colorectal MobileNetV2

The shared dataset inventory and filename contract are documented in the
[HISTOPANTUM colorectal dataset card](../../docs/datasets/histopantum-colon.md).

This experiment repeats Experiment 6 on the colorectal subset of HISTOPANTUM
with an ImageNet-pretrained MobileNetV2. It classifies 224 x 224 H&E patches as
`non-tumour` or `tumour` for research and educational use only.

## Dataset and split

The dataset is not committed. Experiment 7 reuses the exact case-disjoint
Experiment 6 assignment (SHA-256
`b8bca9f2baabca3e3bf3f118a321f8e65fa146f35858f597645aaeb1df85a37a`):

| Split | Cases | Patches | Non-tumour | Tumour |
| --- | ---: | ---: | ---: | ---: |
| Train | 28 | 19,092 | 7,224 | 11,868 |
| Validation | 6 | 3,985 | 1,535 | 2,450 |
| Test | 6 | 4,171 | 1,540 | 2,631 |

The manifest contains 27,248 unique relative paths from 40 TCGA cases and 40
slides. Every case and slide occurs in exactly one partition.

In this local release, each TCGA case is treated as the patient-level grouping
unit and has one slide. Each slide contributes multiple correlated patches, so
patch count must not be interpreted as patient count.

## Training protocol

Both phases use MobileNetV2 preprocessing embedded as a serializable Keras
`Rescaling` layer, a binary sigmoid head, and training-only flip, rotation,
zoom, and contrast augmentation.

Training augmentation consists of horizontal and vertical flips, rotations up
to approximately +/-90 degrees, zoom of approximately +/-10%, and contrast
variation of approximately +/-10%. These transformations are generated in
memory only during `model.fit()`. Validation and test receive no random
augmentation; they receive only deterministic resizing and scaling to `[-1, 1]`.

1. Train only the 1,281-parameter classification head at learning rate `1e-3`.
2. Restore the best frozen checkpoint.
3. Unfreeze the 20 non-BatchNorm layers from `block_14_expand` onward and
   fine-tune at `1e-5`.
4. Select the deployment candidate using validation loss only.
5. Evaluate both prespecified checkpoints once on the untouched test cases.

Fine-tuning exposes 1,512,001 of 2,259,265 parameters (66.92%). This is 20 of
155 backbone layers (12.90% by layer count); all Batch Normalization layers
remain frozen.

## MobileNetV2 results

| Metric | Frozen | Block 14 fine-tuned |
| --- | ---: | ---: |
| Validation loss | **0.380686** | 0.386616 |
| Test accuracy | 0.951570 | **0.959242** |
| Test balanced accuracy | 0.943840 | **0.949787** |
| Tumour precision | 0.950984 | **0.951228** |
| Tumour recall | 0.973394 | **0.985937** |
| Specificity | **0.914286** | 0.913636 |
| Tumour F1 | 0.962059 | **0.968272** |
| ROC-AUC | 0.986326 | **0.990239** |

Validation loss selected the frozen checkpoint before test evaluation. The
fine-tuned test result is retained as a prespecified experimental comparison,
but it must not replace the selected checkpoint based on test performance.
The selected checkpoint's test confusion matrix is `[[1408, 132], [70, 2561]]`.

## Comparison with Experiment 6

The defensible architecture comparison uses this experiment and
[Experiment 6](../exp-6/README.md) only at this explicit comparison boundary.
Each experiment's validation-selected checkpoint is evaluated on the identical
test split:

| Metric | ResNet50 selected | MobileNetV2 selected |
| --- | ---: | ---: |
| Test accuracy | **0.967634** | 0.951570 |
| Test balanced accuracy | **0.963708** | 0.943840 |
| Tumour precision | **0.970234** | 0.950984 |
| Tumour recall | **0.978715** | 0.973394 |
| Specificity | **0.948701** | 0.914286 |
| Tumour F1 | **0.974456** | 0.962059 |
| ROC-AUC | **0.996158** | 0.986326 |

ResNet50 performed better on every reported selected-model test metric.
MobileNetV2's selected file is about 9.7 MB versus about 214.7 MB for ResNet50,
so it offers a substantially smaller artifact at the cost of lower performance
in this run. File size is not a latency or deployment benchmark.

The high patch-level scores are plausible because the classes contain visible
morphological and staining differences, the backbones start from ImageNet
weights, and all splits come from the same acquisition domain. They demonstrate
internal patch separation, not generalization across hospitals or datasets.

## Artifacts

- `HISTOPANTUM_Colon_MobileNetV2.ipynb`: canonical executed Colab record.
- `outputs/split_manifest.csv`: exact case-disjoint assignment.
- `outputs/*_history.csv`: frozen and fine-tuning histories.
- `outputs/*_test_predictions.csv`: per-patch probabilities and predictions.
- `outputs/*_case_metrics.csv`: per-case evaluation.
- `outputs/*_test_metrics.json`: pooled and case-macro metrics.
- `outputs/validation_selection.json`: validation-only model selection.
- `outputs/model_manifest.json`: selected model identity and checksum.

The selected model is intentionally excluded from Git:

```text
file: mobilenet_v2_selected_best.keras
size: 9652368 bytes
sha256: 4d10344cd5b481d936a2471178b390cc2987c353b44fdcabae6af4a152a98c55
```

The ZIP archives are also ignored because the compact evidence is committed
under `outputs/` and model artifacts must not enter the repository.

## Limitations

- The evaluation contains only six test cases from a 40-case dataset.
- Patch-level observations within a case are correlated.
- The 4,171 test patches must not be interpreted as 4,171 independent patients.
- This is internal colorectal evaluation, not external or clinical validation.
- A different 25/8/7 split would add only one test case, reduce training
  diversity, and require both architectures to be retrained. Case-level
  cross-validation and an external test source would provide stronger evidence.
- The selected frozen checkpoint and fine-tuned checkpoint disagree depending
  on whether validation loss or test metrics are inspected; selection remains
  fixed by validation loss to avoid test leakage.
- The executed notebook and exported files under `outputs/` preserve the Colab
  run. The final manifest field correction was applied after execution to
  distinguish the selected frozen model from the fine-tuning configuration.

# Experiment 6: HISTOPANTUM colorectal fine-tuning

The shared dataset inventory and filename contract are documented in the
[HISTOPANTUM colorectal dataset card](../../docs/datasets/histopantum-colon.md).

This experiment replaces the augmentation-heavy LC25000 baseline with the
colorectal subset of HISTOPANTUM. It classifies 224 x 224 H&E patches as
`non-tumour` or `tumour` for research and educational use only.

## Dataset and split

The dataset itself is not committed. The local audit found 27,248 unique JPEG
patches from 40 TCGA cases and 40 slides:

- 10,299 non-tumour patches;
- 16,949 tumour patches;
- one slide per case in this colorectal release.

In this local release, the TCGA case ID is treated as the patient-level
grouping unit. Each case has one slide, and each slide contributes multiple
correlated patches. Patch count must therefore not be interpreted as patient
count.

Filenames encode the TCGA slide and patch coordinates. The notebook extracts
the TCGA case ID and assigns every patch from a case to one partition. A
deterministic seed-42 search targets 70/15/15 case counts while balancing total,
tumour, and non-tumour patch counts.

| Split | Cases | Patches | Non-tumour | Tumour |
| --- | ---: | ---: | ---: | ---: |
| Train | 28 | 19,092 | 7,224 | 11,868 |
| Validation | 6 | 3,985 | 1,535 | 2,450 |
| Test | 6 | 4,171 | 1,540 | 2,631 |

The committed manifest contains 27,248 unique paths. Each case and slide occurs
in exactly one partition.

The canonical case assignment has SHA-256
`b8bca9f2baabca3e3bf3f118a321f8e65fa146f35858f597645aaeb1df85a37a`.

## Training protocol

Both phases use the same split, ImageNet-pretrained ResNet50, Caffe-style
preprocessing, binary sigmoid head, and training-only flip, rotation, zoom, and
contrast augmentation.

Training augmentation consists of horizontal and vertical flips, rotations up
to approximately +/-90 degrees, zoom of approximately +/-10%, and contrast
variation of approximately +/-10%. These transformations are generated in
memory only during `model.fit()`. Validation and test receive no random
augmentation; they receive only deterministic resizing and Caffe preprocessing.

1. Train only the 2,049-parameter classification head at learning rate `1e-3`.
2. Restore the best frozen checkpoint.
3. Unfreeze the 22 non-BatchNorm `conv5_*` layers and fine-tune at `1e-5`.
4. Select the deployment candidate using validation loss only.
5. Evaluate both prespecified checkpoints once on the untouched test cases.

The fine-tuning phase exposes 14,955,521 trainable parameters, approximately
63.4% of all model parameters. This corresponds to 22 of the 175 backbone
layers (12.57% by layer count). Batch Normalization layers remain frozen.

## Results

| Metric | Frozen | Conv5 fine-tuned |
| --- | ---: | ---: |
| Validation loss | 0.616192 | **0.485846** |
| Test accuracy | 0.966675 | **0.967634** |
| Test balanced accuracy | **0.966045** | 0.963708 |
| Tumour precision | **0.978495** | 0.970234 |
| Tumour recall | 0.968453 | **0.978715** |
| Specificity | **0.963636** | 0.948701 |
| Tumour F1 | 0.973448 | **0.974456** |
| ROC-AUC | 0.995331 | **0.996158** |

Validation loss selected the conv5 fine-tuned checkpoint. Its test confusion
matrix is `[[1461, 79], [56, 2575]]`, with rows representing non-tumour and
tumour ground truth. Fine-tuning increased tumour sensitivity but produced more
non-tumour false positives.

Mean case-level accuracy was 0.9692 +/- 0.0151 for the frozen model and
0.9705 +/- 0.0288 after fine-tuning. Case-level metrics are important because
patches from the same case are correlated and case patch counts are uneven.

The high patch-level scores are plausible because the classes contain visible
morphological and staining differences, the backbone starts from ImageNet
weights, and all splits come from the same acquisition domain. They demonstrate
internal patch separation, not generalization across hospitals or datasets.

## Artifacts

- `HISTOPANTUM_Colon_ResNet50.ipynb`: canonical executed Colab record.
- `outputs/split_manifest.csv`: exact case-disjoint assignment.
- `outputs/*_history.csv`: frozen and fine-tuning histories.
- `outputs/*_test_predictions.csv`: per-patch probabilities and predictions.
- `outputs/*_case_metrics.csv`: per-case evaluation.
- `outputs/*_test_metrics.json`: pooled and case-macro metrics.
- `outputs/validation_selection.json`: validation-only model selection.
- `outputs/model_manifest.json`: selected model identity and checksum.

The selected model is intentionally excluded from Git:

```text
file: resnet50_selected_best.keras
size: 214686627 bytes
sha256: 7fee95edb0df79a96ac4c0630d84ffa9e41c298292512dd850767087a8dbf00c
```

The compact ZIP is also ignored because its contents are committed under
`outputs/`.

## Limitations

- The evaluation contains only six test cases from a 40-case dataset.
- Patch-level observations within a case are not independent.
- The 4,171 test patches must not be interpreted as 4,171 independent patients.
- This is internal colorectal evaluation, not external or clinical validation.
- A different 25/8/7 split would add only one test case, reduce training
  diversity, and require both architectures to be retrained. Case-level
  cross-validation and an external test source would provide stronger evidence.
- The small improvement does not establish that fine-tuning will always
  generalize better.
- The saved model embeds Caffe preprocessing in a Keras `Lambda` and requires
  the notebook's `custom_objects` mapping when loaded. The current application
  adapter preprocesses externally, so the inference boundary must be corrected
  before deployment to avoid deserialization failure or double preprocessing.

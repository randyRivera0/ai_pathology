# Experiment 5: EBHI-SEG external pilot

This experiment checks whether the frozen final ResNet50 colon classifier retains
its LC25000 performance on images from another dataset. It is a preliminary
external robustness check, not clinical validation.

## Protocol

- Dataset: EBHI-SEG histology source images; segmentation masks are excluded.
- Pilot subset: 50 Normal and 50 Adenocarcinoma images.
- Selection: deterministic random samples using seeds 42 and 43, respectively.
- Model: unchanged final binary ResNet50 checkpoint.
- Preprocessing: unchanged 224 x 224 nearest-neighbor resize and explicit
  Caffe-style RGB-to-BGR mean subtraction.
- Decision threshold: unchanged at 0.5.
- No retraining, fine-tuning, threshold selection, or stain normalization.

`EBHI-SEG-pilot-colab.zip` contains the exact 100 images evaluated by the
notebook and uses portable archive paths suitable for Colab.

## Results

| Metric | Result |
| --- | ---: |
| Accuracy | 0.8900 |
| Balanced accuracy | 0.8900 |
| ROC-AUC | 1.0000 |
| Macro F1 | 0.8887 |
| Normal recall | 0.7800 |
| Adenocarcinoma recall | 1.0000 |

Confusion matrix, ordered as Normal and Adenocarcinoma:

```text
[[39, 11],
 [ 0, 50]]
```

The frozen 0.5 threshold produced 11 false positives and no false negatives in
this balanced pilot. Perfect ROC-AUC indicates complete ranking separation on
these 100 images, while the lower fixed-threshold accuracy demonstrates a
cross-dataset probability shift.

## Limitations

- This is a small balanced pilot, not the complete EBHI-SEG evaluation.
- EBHI-SEG and LC25000 differ in acquisition and label semantics.
- Images may not represent 100 statistically independent patients.
- The threshold must not be tuned on this pilot and then evaluated on the same
  images.
- Results are for research and education only and cannot support diagnosis or
  clinical decision-making.

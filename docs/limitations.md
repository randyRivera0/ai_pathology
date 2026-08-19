# Limitations

## Research validity

Experiment 1 used an image-level 80/20 split of LC25000. Because the dataset
contains augmented and potentially related images, closely related samples may
appear on both sides of the split. Its reported 99.3% validation accuracy must
therefore not be presented as independent patient-level generalization.

The final binary ResNet50 experiment instead assigns each published
LC25000-clean image group wholly to train, validation, or test. It reached 100%
accuracy on its 1,451-image internal test partition. This prevents identified
group overlap, but it does not create independent patient-level validation or
eliminate dataset-specific color, stain, and acquisition shortcuts. A
color-histogram logistic-regression baseline reached 91.18% on the same test
partition, showing that color alone is strongly predictive in this dataset.

A frozen-model pilot on 100 balanced EBHI-SEG images used unchanged
preprocessing and a fixed 0.5 threshold. It reached 89% balanced accuracy,
88.87% macro F1, and 1.0 ROC-AUC, with confusion matrix `[[39, 11], [0, 50]]`.
This drop is evidence of cross-dataset probability shift, not full external
validation. The pilot is small, may contain correlated images, and must not be
used both to tune and evaluate a new threshold.

The project has not established:

- full external patient-level validation;
- performance under scanner, laboratory, stain, or population shifts;
- calibrated clinical probabilities;
- demographic or clinically relevant subgroup performance;
- robustness to artifacts and out-of-distribution images.

## Product limitations

The current application handles prepared PNG and JPEG images. It does not
implement whole-slide-image ingestion, tissue detection, tiling, quality
control, suspicious-region localization, human review workflows, audit trails,
monitoring, or clinical integration.

## Clinical status

The system is a course and research prototype. It has not undergone clinical,
regulatory, privacy, security, or medical-device validation and must not be used
for diagnosis or treatment decisions.

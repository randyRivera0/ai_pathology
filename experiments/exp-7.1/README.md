# Experiment 7.1: retrospective ResNet50 and MobileNetV2 comparison

Experiment 7.1 summarizes the validation-selected models from Experiments 6
and 7 in one reproducible report. It compares the ResNet50 and MobileNetV2
training histories, selected-epoch metrics, and held-out-test confusion
matrices using only the compact CSV and JSON evidence already committed for
those experiments. It does not load image data or model checkpoints.

This report is for research and educational use only. It is not clinical
validation and must not be used for diagnosis or clinical decision-making.

## Selection and evaluation roles

Each source experiment selected its checkpoint using validation loss:

- Experiment 6 selected the partially fine-tuned ResNet50 checkpoint.
- Experiment 7 selected the frozen MobileNetV2 checkpoint.

The report preserves those choices and does not select a model from test-set
performance. The held-out cases had already been evaluated in their respective
source experiments, so this retrospective visualization is not a new or
independent test. It must not be used to tune either model or claim that one
architecture is universally superior.

Experiment 8 provides the stronger architecture comparison because it uses
five-fold case-disjoint cross-validation and exposes between-case variability.
Experiment 9 subsequently locks a ResNet50 recipe from those cross-validation
results and trains one model on the complete development pool.

## Reproduce the report

From the repository root, run:

```powershell
python experiments/exp-7.1/generate_comparison_report.py
```

The script reads from `experiments/exp-6/outputs/` and
`experiments/exp-7/outputs/`, then writes:

- `outputs/metrics_table.csv`;
- `outputs/metrics_table.png`;
- `outputs/accuracy_loss_curves.png`;
- `outputs/confusion_matrices.png`.

Use `--repo-root` or `--output-dir` to override the inferred paths when needed.

## Interpretation limits

- The train and validation rows come from the validation-selected epoch.
- Some train and validation metrics were not recorded by the source notebooks;
  unavailable values are displayed as an em dash.
- Patch observations within a case are correlated and are not independent
  patient-level samples.
- The experiments use a small internal case set from one colorectal dataset.
- These figures do not demonstrate external, cross-hospital, or clinical
  generalization.

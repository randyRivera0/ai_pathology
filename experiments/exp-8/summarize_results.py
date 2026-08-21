"""Summarize Experiment 8 out-of-fold predictions without model inference."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


ARCHITECTURES = ("resnet50", "mobilenetv2")
METRICS = ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc")


def calculate_metrics(frame: pd.DataFrame) -> dict[str, float]:
    """Calculate binary patch-level metrics from an exported prediction frame."""
    y_true = frame["label"].to_numpy(dtype=int)
    y_pred = frame["prediction"].to_numpy(dtype=int)
    probabilities = frame["probability"].to_numpy(dtype=float)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
    }


def summarize_architecture(architecture: str, output_dir: Path) -> pd.DataFrame:
    """Return fold, fold-macro, and pooled OOF metrics for one architecture."""
    fold_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []

    for fold_id in range(5):
        path = output_dir / f"{architecture}_fold{fold_id}_predictions.csv"
        frame = pd.read_csv(path)
        if len(frame) != frame["relative_path"].nunique():
            raise ValueError(f"Duplicate patch paths in {path}")
        prediction_frames.append(frame)
        fold_rows.append(
            {
                "architecture": architecture,
                "scope": f"fold{fold_id}",
                "fold_id": fold_id,
                "patches": len(frame),
                "cases": frame["case_id"].nunique(),
                **calculate_metrics(frame),
            }
        )

    fold_frame = pd.DataFrame(fold_rows)
    pooled = pd.concat(prediction_frames, ignore_index=True)
    if len(pooled) != pooled["relative_path"].nunique():
        raise ValueError(f"OOF patch paths overlap for {architecture}")

    summary_rows = [
        {
            "architecture": architecture,
            "scope": "fold_macro_mean",
            "fold_id": np.nan,
            "patches": len(pooled),
            "cases": pooled["case_id"].nunique(),
            **{metric: fold_frame[metric].mean() for metric in METRICS},
        },
        {
            "architecture": architecture,
            "scope": "fold_macro_std",
            "fold_id": np.nan,
            "patches": len(pooled),
            "cases": pooled["case_id"].nunique(),
            **{metric: fold_frame[metric].std(ddof=1) for metric in METRICS},
        },
        {
            "architecture": architecture,
            "scope": "pooled_oof",
            "fold_id": np.nan,
            "patches": len(pooled),
            "cases": pooled["case_id"].nunique(),
            **calculate_metrics(pooled),
        },
    ]
    return pd.concat([fold_frame, pd.DataFrame(summary_rows)], ignore_index=True)


def generate_summary(output_dir: Path) -> Path:
    """Validate the evidence files and write the cross-validation summary CSV."""
    frames = [summarize_architecture(name, output_dir) for name in ARCHITECTURES]
    summary = pd.concat(frames, ignore_index=True)
    destination = output_dir / "cross_validation_summary.csv"
    summary.to_csv(destination, index=False, float_format="%.9f")
    return destination


def main() -> None:
    """Run the command-line summary generator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Directory containing Experiment 8 prediction CSVs.",
    )
    args = parser.parse_args()
    destination = generate_summary(args.output_dir.resolve())
    print(destination)


if __name__ == "__main__":
    main()

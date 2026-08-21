"""Generate comparison tables and diagnostic plots for Experiments 6 and 7.

The script reads only the compact CSV/JSON artifacts committed under each
experiment's ``outputs`` directory. It does not load the datasets or models.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


EXPERIMENTS = {
    "Experiment 6 — ResNet50": {
        "directory": "exp-6",
        "selected_history": "fine_tune_history.csv",
        "test_metrics": "conv5_fine_tuned_test_metrics.json",
    },
    "Experiment 7 — MobileNetV2": {
        "directory": "exp-7",
        "selected_history": "frozen_history.csv",
        "test_metrics": "frozen_test_metrics.json",
    },
}

CLASS_NAMES = ("Non-tumour", "Tumour")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    default_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Create metric tables and plots for Experiments 6 and 7."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=default_root,
        help="Repository root (default: inferred from this script).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: experiments/exp-7.1/outputs).",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object and raise a useful error when it is absent."""
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_history(path: Path) -> dict[str, list[float]]:
    """Read a Keras history CSV into numeric columns."""
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Training history is empty: {path}")
    return {key: [float(row[key]) for row in rows] for key in rows[0]}


def selected_training_metrics(history: dict[str, list[float]]) -> dict[str, float]:
    """Return metrics from the minimum-validation-loss history row."""
    best_index = int(np.argmin(history["val_loss"]))
    return {
        "accuracy": history["accuracy"][best_index],
        "loss": history["loss"][best_index],
        "precision": history["precision"][best_index],
        "recall": history["recall"][best_index],
        "roc_auc": history["roc_auc"][best_index],
    }


def build_metric_rows(experiments_dir: Path) -> list[dict[str, Any]]:
    """Build train, validation, and held-out-test rows for selected models."""
    rows: list[dict[str, Any]] = []
    for model_name, config in EXPERIMENTS.items():
        outputs = experiments_dir / config["directory"] / "outputs"
        selection = read_json(outputs / "validation_selection.json")
        validation = selection["results"][selection["selected"]]
        history = read_history(outputs / config["selected_history"])
        training = selected_training_metrics(history)
        test_document = read_json(outputs / config["test_metrics"])
        test = test_document["pooled_patch_metrics"]

        for split, metrics in (
            ("Train (selected epoch)", training),
            ("Validation", validation),
            ("Test", test),
        ):
            rows.append(
                {
                    "model": model_name,
                    "split": split,
                    "accuracy": metrics.get("accuracy"),
                    "balanced_accuracy": metrics.get("balanced_accuracy"),
                    "loss": metrics.get("loss"),
                    "precision": metrics.get("precision"),
                    "recall": metrics.get("recall"),
                    "specificity": metrics.get("specificity"),
                    "f1": metrics.get("f1"),
                    "roc_auc": metrics.get("roc_auc"),
                }
            )
    return rows


def write_metric_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write the combined metric table as a reusable CSV file."""
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def format_metric(value: Any) -> str:
    """Format a metric for the rendered table."""
    return "—" if value is None else f"{float(value):.4f}"


def plot_metric_table(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Render the combined metrics table as a PNG image."""
    metric_keys = (
        "accuracy",
        "balanced_accuracy",
        "loss",
        "precision",
        "recall",
        "specificity",
        "f1",
        "roc_auc",
    )
    headers = ("Model", "Split", "Accuracy", "Balanced acc.", "Loss", "Precision", "Recall", "Specificity", "F1", "ROC-AUC")
    table_rows = []
    for row in rows:
        short_model = "ResNet50" if "ResNet50" in row["model"] else "MobileNetV2"
        table_rows.append(
            [short_model, row["split"]]
            + [format_metric(row[key]) for key in metric_keys]
        )

    fig, ax = plt.subplots(figsize=(15, 4.3))
    ax.axis("off")
    table = ax.table(cellText=table_rows, colLabels=headers, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.55)
    for (row_index, _), cell in table.get_celld().items():
        if row_index == 0:
            cell.set_facecolor("#264653")
            cell.set_text_props(color="white", weight="bold")
        elif row_index in (1, 2, 3):
            cell.set_facecolor("#e9f5f2")
        else:
            cell.set_facecolor("#fff3df")
    ax.set_title("Experiments 6 and 7 — Selected-model metrics", weight="bold", pad=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def combine_histories(outputs: Path) -> tuple[dict[str, np.ndarray], int]:
    """Combine frozen and fine-tuning histories on a continuous epoch axis."""
    frozen = read_history(outputs / "frozen_history.csv")
    fine_tune = read_history(outputs / "fine_tune_history.csv")
    combined = {
        key: np.asarray(frozen[key] + fine_tune[key], dtype=float)
        for key in frozen
        if key in fine_tune
    }
    return combined, len(frozen["epoch"])


def plot_learning_curves(experiments_dir: Path, output_path: Path) -> None:
    """Plot accuracy and loss across frozen and fine-tuning phases."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex="col")
    for column, (model_name, config) in enumerate(EXPERIMENTS.items()):
        outputs = experiments_dir / config["directory"] / "outputs"
        history, fine_tune_start = combine_histories(outputs)
        epochs = np.arange(1, len(history["accuracy"]) + 1)

        for row, (metric, label) in enumerate((("accuracy", "Accuracy"), ("loss", "Loss"))):
            ax = axes[row, column]
            ax.plot(epochs, history[metric], marker="o", label="Training")
            ax.plot(epochs, history[f"val_{metric}"], marker="o", label="Validation")
            ax.axvline(
                fine_tune_start + 0.5,
                color="#6c757d",
                linestyle="--",
                linewidth=1.2,
                label="Fine-tuning starts" if row == 0 else None,
            )
            ax.set_ylabel(label)
            ax.grid(alpha=0.25)
            if row == 0:
                ax.set_title(model_name, weight="bold")
                ax.legend(fontsize=8)
            else:
                ax.set_xlabel("Epoch (frozen phase followed by fine-tuning)")
    fig.suptitle("Training and validation learning curves", fontsize=15, weight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices(experiments_dir: Path, output_path: Path) -> None:
    """Plot test confusion matrices for the validation-selected checkpoints."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    for ax, (model_name, config) in zip(axes, EXPERIMENTS.items()):
        metrics_path = (
            experiments_dir
            / config["directory"]
            / "outputs"
            / config["test_metrics"]
        )
        matrix = np.asarray(read_json(metrics_path)["confusion_matrix"], dtype=int)
        image = ax.imshow(matrix, cmap="Blues")
        threshold = matrix.max() / 2
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                ax.text(
                    column,
                    row,
                    f"{matrix[row, column]:,}",
                    ha="center",
                    va="center",
                    color="white" if matrix[row, column] > threshold else "black",
                    fontsize=12,
                    weight="bold",
                )
        ax.set_xticks(range(2), CLASS_NAMES)
        ax.set_yticks(range(2), CLASS_NAMES)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title(model_name, weight="bold")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Held-out test confusion matrices — selected checkpoints", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Generate all Experiment 6/7 comparison report artifacts."""
    args = parse_args()
    repo_root = args.repo_root.resolve()
    experiments_dir = repo_root / "experiments"
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else experiments_dir / "exp-7.1" / "outputs"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = build_metric_rows(experiments_dir)
    write_metric_csv(rows, output_dir / "metrics_table.csv")
    plot_metric_table(rows, output_dir / "metrics_table.png")
    plot_learning_curves(experiments_dir, output_dir / "accuracy_loss_curves.png")
    plot_confusion_matrices(experiments_dir, output_dir / "confusion_matrices.png")

    print(f"Report generated in: {output_dir}")
    for filename in (
        "metrics_table.csv",
        "metrics_table.png",
        "accuracy_loss_curves.png",
        "confusion_matrices.png",
    ):
        print(f"- {filename}")


if __name__ == "__main__":
    main()

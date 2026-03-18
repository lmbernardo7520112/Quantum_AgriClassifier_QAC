"""QAC — Evaluation module.

Provides classification metrics, model comparison, and visualization.
ClassificationResult is the typed output for all models.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


@dataclass(frozen=True)
class ClassificationResult:
    """Typed result of a classification experiment.

    Mirrors SolverResult from vqe_knapsack_local — fully serializable.
    """

    model_type: str
    model_path: str
    model_hash: str
    accuracy: float
    f1_weighted: float
    f1_macro: float
    precision: float
    recall: float
    confusion_matrix: list[list[int]]
    training_time_s: float
    inference_time_s: float
    y_pred: np.ndarray
    hyperparameters: dict[str, Any]
    dataset_metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "model_path": self.model_path,
            "model_hash": self.model_hash,
            "accuracy": self.accuracy,
            "f1_weighted": self.f1_weighted,
            "f1_macro": self.f1_macro,
            "precision": self.precision,
            "recall": self.recall,
            "confusion_matrix": self.confusion_matrix,
            "training_time_s": self.training_time_s,
            "inference_time_s": self.inference_time_s,
            "hyperparameters": self.hyperparameters,
        }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Compute standard classification metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        Dict with accuracy, F1, precision, recall, and confusion matrix.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


@dataclass
class ComparisonResult:
    """Comparison between classical and quantum models."""

    results: list[ClassificationResult]
    best_model: ClassificationResult
    quantum_advantage: bool
    accuracy_delta: float

    def to_dict(self) -> dict[str, Any]:
        table = []
        for r in self.results:
            table.append({
                "model_type": r.model_type,
                "accuracy": round(r.accuracy, 4),
                "f1_weighted": round(r.f1_weighted, 4),
                "training_time_s": r.training_time_s,
                "inference_time_s": r.inference_time_s,
            })
        return {
            "comparison_table": table,
            "best_model": self.best_model.model_type,
            "quantum_advantage": self.quantum_advantage,
            "accuracy_delta": round(self.accuracy_delta, 4),
        }


def compare_results(results: list[ClassificationResult]) -> ComparisonResult:
    """Generate comparison between all model results.

    Args:
        results: List of ClassificationResult from different models.

    Returns:
        ComparisonResult with best model and quantum advantage analysis.
    """
    best = max(results, key=lambda r: r.accuracy)

    quantum_models = [r for r in results if r.model_type in ("vqc", "qsvm")]
    classical_models = [r for r in results if r.model_type in ("svm",)]

    quantum_advantage = False
    accuracy_delta = 0.0

    if quantum_models and classical_models:
        best_quantum = max(quantum_models, key=lambda r: r.accuracy)
        best_classical = max(classical_models, key=lambda r: r.accuracy)
        quantum_advantage = best_quantum.accuracy > best_classical.accuracy
        accuracy_delta = best_quantum.accuracy - best_classical.accuracy

    return ComparisonResult(
        results=results,
        best_model=best,
        quantum_advantage=quantum_advantage,
        accuracy_delta=accuracy_delta,
    )


def plot_comparison(
    results: list[ClassificationResult],
    output_path: str = "outputs/comparison.png",
    title: str = "Classical vs Quantum Classifier — EuroSAT",
) -> str:
    """Generate comparison bar chart and save to file.

    Args:
        results: List of ClassificationResult.
        output_path: Path to save the figure.
        title: Chart title.

    Returns:
        Path to saved figure.
    """
    import matplotlib.pyplot as plt
    from pathlib import Path

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    models = [r.model_type.upper() for r in results]
    accuracies = [r.accuracy for r in results]
    f1_scores = [r.f1_weighted for r in results]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, accuracies, width, label="Accuracy", color="#2196F3")
    bars2 = ax.bar(x + width / 2, f1_scores, width, label="F1 (weighted)", color="#FF9800")

    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim(0, 1.15)

    for bar in bars1:
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, yval + 0.02,
            f"{yval:.3f}", ha="center", va="bottom", fontsize=11,
        )
    for bar in bars2:
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, yval + 0.02,
            f"{yval:.3f}", ha="center", va="bottom", fontsize=11,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path


def plot_confusion_matrix(
    result: ClassificationResult,
    class_names: tuple[str, ...],
    output_path: str = "outputs/confusion_matrix.png",
) -> str:
    """Plot confusion matrix for a single model result.

    Args:
        result: ClassificationResult with confusion matrix.
        class_names: Tuple of class name strings.
        output_path: Path to save the figure.

    Returns:
        Path to saved figure.
    """
    import matplotlib.pyplot as plt
    from pathlib import Path

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    cm = np.array(result.confusion_matrix)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=list(class_names),
        yticklabels=list(class_names),
        title=f"Confusion Matrix — {result.model_type.upper()}",
        ylabel="True Label",
        xlabel="Predicted Label",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path

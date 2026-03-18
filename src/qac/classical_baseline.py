"""QAC — Classical Baseline (SVM).

Provides SVM baseline classifier for comparison with quantum models.
Simplified from original classical/baseline.py — only SVM for Bloco 3 scope.
"""
from __future__ import annotations

import hashlib
import logging
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.svm import SVC

from qac.config import ModelConfig
from qac.data_loader import DatasetResult
from qac.evaluation import ClassificationResult, compute_metrics

logger = logging.getLogger(__name__)


def train_svm(
    dataset: DatasetResult,
    config: ModelConfig,
    models_dir: str | Path = "models",
    experiment_id: str = "",
) -> ClassificationResult:
    """Train SVM baseline classifier.

    Args:
        dataset: DatasetResult (already preprocessed with PCA).
        config: ModelConfig (model_type must be "svm").
        models_dir: Directory to save model artifacts.
        experiment_id: Unique experiment identifier.

    Returns:
        ClassificationResult with metrics, timing, and model hash.
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    logger.info("  [SVM] Training baseline classifier...")

    # Train
    start_time = time.time()
    svm = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=config.seed, probability=True)
    svm.fit(dataset.X_train, dataset.y_train)
    training_time = time.time() - start_time

    # Predict + metrics
    infer_start = time.time()
    y_pred = svm.predict(dataset.X_test)
    inference_time = time.time() - infer_start

    metrics = compute_metrics(dataset.y_test, y_pred)

    # Save model
    model_filename = f"svm_baseline_{experiment_id}.pkl"
    model_path = models_dir / model_filename
    with open(model_path, "wb") as f:
        pickle.dump(svm, f)

    model_hash = _file_hash(str(model_path))

    logger.info(
        "  [SVM] Done. Accuracy=%.4f, F1=%.4f (%.2fs train, %.3fs infer)",
        metrics["accuracy"],
        metrics["f1_weighted"],
        training_time,
        inference_time,
    )

    return ClassificationResult(
        model_type="svm",
        model_path=str(model_path),
        model_hash=model_hash,
        accuracy=metrics["accuracy"],
        f1_weighted=metrics["f1_weighted"],
        f1_macro=metrics["f1_macro"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        confusion_matrix=metrics["confusion_matrix"],
        training_time_s=round(training_time, 3),
        inference_time_s=round(inference_time, 3),
        y_pred=y_pred,
        hyperparameters={
            "kernel": "rbf",
            "C": 1.0,
            "gamma": "scale",
            "seed": config.seed,
        },
        dataset_metadata=dataset.metadata,
    )


def _file_hash(filepath: str) -> str:
    """SHA-256 hash of file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

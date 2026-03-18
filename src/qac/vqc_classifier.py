"""QAC — Variational Quantum Classifier (VQC).

Implements VQC from qiskit-machine-learning with configurable
feature map, ansatz, and optimizer via factory pattern.

This is the PRIMARY quantum model for Bloco 3.
"""
from __future__ import annotations

import hashlib
import logging
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np

from qac.ansatz_factory import AnsatzFactory
from qac.config import ModelConfig
from qac.data_loader import DatasetResult
from qac.evaluation import ClassificationResult, compute_metrics
from qac.feature_map_factory import FeatureMapFactory
from qac.optimizer_factory import OptimizerFactory
from qac.preprocessing import normalize_features

logger = logging.getLogger(__name__)


def train_vqc(
    dataset: DatasetResult,
    config: ModelConfig,
    models_dir: str | Path = "models",
    experiment_id: str = "",
) -> ClassificationResult:
    """Train Variational Quantum Classifier.

    Uses factories for feature_map, ansatz, and optimizer creation.
    Returns typed ClassificationResult.

    Args:
        dataset: DatasetResult with PCA-reduced features.
        config: ModelConfig with VQC hyperparameters.
        models_dir: Directory to save model artifacts.
        experiment_id: Unique experiment identifier.

    Returns:
        ClassificationResult with metrics, timing, and model hash.

    Raises:
        ValueError: If feature_dim != n_qubits.
    """
    from qiskit_machine_learning.algorithms import VQC

    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    if dataset.feature_dim != config.n_qubits:
        raise ValueError(
            f"Feature dimension ({dataset.feature_dim}) must match "
            f"n_qubits ({config.n_qubits}). Apply PCA first."
        )

    logger.info(
        "  [VQC] Training: ansatz=%s, optimizer=%s, n_qubits=%d, max_iter=%d",
        config.ansatz_type,
        config.optimizer_type,
        config.n_qubits,
        config.max_iter,
    )

    # Normalize features to [0, 2π]
    X_train = normalize_features(dataset.X_train)
    X_test = normalize_features(dataset.X_test)

    # Create components via factories
    feature_map = FeatureMapFactory.create(config.feature_map_type, config.n_qubits)
    ansatz = AnsatzFactory.create(config.ansatz_type, config.n_qubits)
    optimizer = OptimizerFactory.create(config.optimizer_type, config.max_iter)

    # Training history callback
    history: list[dict[str, Any]] = []

    def callback(weights, obj_func_eval):
        history.append({"iteration": len(history), "objective": float(obj_func_eval)})

    # Create and train VQC
    start_time = time.time()

    vqc = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
        callback=callback,
    )

    # One-hot encode labels for VQC
    num_classes = dataset.num_classes
    y_train_onehot = np.eye(num_classes)[dataset.y_train]

    vqc.fit(X_train, y_train_onehot)
    training_time = time.time() - start_time

    # Evaluate
    infer_start = time.time()
    y_pred_onehot = vqc.predict(X_test)
    y_pred = (
        np.argmax(y_pred_onehot, axis=1)
        if y_pred_onehot.ndim > 1
        else y_pred_onehot
    )
    inference_time = time.time() - infer_start

    metrics = compute_metrics(dataset.y_test, y_pred)

    # Save model
    model_filename = f"vqc_{experiment_id}.pkl"
    model_path = models_dir / model_filename
    model_data = {
        "weights": vqc.weights if hasattr(vqc, "weights") else None,
        "hyperparameters": config.to_dict(),
        "training_history": history,
    }
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)

    model_hash = _file_hash(str(model_path))

    logger.info(
        "  [VQC] Done. Accuracy=%.4f, F1=%.4f (%.2fs train, %.3fs infer)",
        metrics["accuracy"],
        metrics["f1_weighted"],
        training_time,
        inference_time,
    )

    return ClassificationResult(
        model_type="vqc",
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
        hyperparameters=config.to_dict(),
        dataset_metadata=dataset.metadata,
    )


def _file_hash(filepath: str) -> str:
    """SHA-256 hash of file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

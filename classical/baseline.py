"""
QAC — Classical Baselines.

Provides SVM and CNN baseline models for comparison with quantum classifiers.
All models are saved with hash and metadata for auditability.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.svm import SVC

from classical.data_loader import DatasetResult


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Compute standard classification metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def train_svm(
    dataset: DatasetResult,
    seed: int = 42,
    kernel: str = "rbf",
    C: float = 1.0,
    gamma: str = "scale",
    models_dir: str | Path = "models",
    experiment_id: str = "",
) -> dict[str, Any]:
    """
    Train SVM baseline classifier.

    Returns dict with model path, hash, and metrics.
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Train
    start_time = time.time()
    svm = SVC(kernel=kernel, C=C, gamma=gamma, random_state=seed, probability=True)
    svm.fit(dataset.X_train, dataset.y_train)
    training_time = time.time() - start_time

    # Predict + metrics
    infer_start = time.time()
    y_pred = svm.predict(dataset.X_test)
    inference_time = time.time() - infer_start

    metrics = compute_metrics(dataset.y_test, y_pred)
    metrics["training_time_s"] = round(training_time, 3)
    metrics["inference_time_s"] = round(inference_time, 3)

    # Save model
    model_filename = f"svm_baseline_{experiment_id}.pkl"
    model_path = models_dir / model_filename
    with open(model_path, "wb") as f:
        pickle.dump(svm, f)

    # Compute model hash
    model_hash = _file_hash(str(model_path))

    return {
        "model_type": "svm",
        "model_path": str(model_path),
        "model_hash": model_hash,
        "metrics": metrics,
        "hyperparameters": {
            "kernel": kernel,
            "C": C,
            "gamma": gamma,
            "seed": seed,
        },
        "dataset_info": dataset.metadata,
    }


def train_cnn(
    dataset: DatasetResult,
    seed: int = 42,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    models_dir: str | Path = "models",
    experiment_id: str = "",
) -> dict[str, Any]:
    """
    Train CNN baseline (ResNet-18 fine-tuned) using PyTorch.

    Returns dict with model path, hash, and metrics.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    np.random.seed(seed)

    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = dataset.num_classes

    # Determine image size from flattened features
    # For RGB: features = H * W * 3
    n_features = dataset.X_train.shape[1]
    img_size = int(np.sqrt(n_features / 3))
    if img_size * img_size * 3 != n_features:
        # Not a perfect RGB image — use simple MLP instead
        return _train_mlp(
            dataset, seed, epochs, batch_size, learning_rate,
            models_dir, experiment_id
        )

    # Reshape to (N, 3, H, W)
    X_train = dataset.X_train.reshape(-1, img_size, img_size, 3).transpose(0, 3, 1, 2)
    X_test = dataset.X_test.reshape(-1, img_size, img_size, 3).transpose(0, 3, 1, 2)

    train_ds = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(dataset.y_train),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # Simple CNN (lighter than ResNet for speed)
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(64, num_classes),
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Train
    start_time = time.time()
    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    training_time = time.time() - start_time

    # Evaluate
    model.eval()
    infer_start = time.time()
    with torch.no_grad():
        X_test_t = torch.FloatTensor(X_test).to(device)
        outputs = model(X_test_t)
        y_pred = outputs.argmax(dim=1).cpu().numpy()
    inference_time = time.time() - infer_start

    metrics = compute_metrics(dataset.y_test, y_pred)
    metrics["training_time_s"] = round(training_time, 3)
    metrics["inference_time_s"] = round(inference_time, 3)

    # Save
    model_filename = f"cnn_baseline_{experiment_id}.pt"
    model_path = models_dir / model_filename
    torch.save(model.state_dict(), model_path)
    model_hash = _file_hash(str(model_path))

    return {
        "model_type": "cnn",
        "model_path": str(model_path),
        "model_hash": model_hash,
        "metrics": metrics,
        "hyperparameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "seed": seed,
        },
        "dataset_info": dataset.metadata,
    }


def _train_mlp(
    dataset: DatasetResult,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    models_dir: Path,
    experiment_id: str,
) -> dict[str, Any]:
    """Fallback MLP for non-image features."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_features = dataset.X_train.shape[1]
    num_classes = dataset.num_classes

    model = nn.Sequential(
        nn.Linear(n_features, 256), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(128, num_classes),
    ).to(device)

    train_ds = TensorDataset(
        torch.FloatTensor(dataset.X_train),
        torch.LongTensor(dataset.y_train),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    start_time = time.time()
    model.train()
    for _ in range(epochs):
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
    training_time = time.time() - start_time

    model.eval()
    infer_start = time.time()
    with torch.no_grad():
        y_pred = model(torch.FloatTensor(dataset.X_test).to(device)).argmax(1).cpu().numpy()
    inference_time = time.time() - infer_start

    metrics = compute_metrics(dataset.y_test, y_pred)
    metrics["training_time_s"] = round(training_time, 3)
    metrics["inference_time_s"] = round(inference_time, 3)

    model_path = models_dir / f"mlp_baseline_{experiment_id}.pt"
    import torch
    torch.save(model.state_dict(), model_path)

    return {
        "model_type": "mlp",
        "model_path": str(model_path),
        "model_hash": _file_hash(str(model_path)),
        "metrics": metrics,
        "hyperparameters": {"epochs": epochs, "batch_size": batch_size, "lr": learning_rate, "seed": seed},
        "dataset_info": dataset.metadata,
    }


def evaluate_model(
    model_path: str | Path,
    model_type: str,
    dataset: DatasetResult,
    split: str = "test",
) -> dict[str, Any]:
    """Evaluate a saved model on a given split."""
    X = {"train": dataset.X_train, "val": dataset.X_val, "test": dataset.X_test}[split]
    y = {"train": dataset.y_train, "val": dataset.y_val, "test": dataset.y_test}[split]

    infer_start = time.time()

    if model_type == "svm":
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        y_pred = model.predict(X)
    elif model_type in ("cnn", "mlp"):
        import torch
        import torch.nn as nn

        # Load model architecture info from metadata if needed
        # For now, do prediction with loaded state dict
        # This is a simplified approach — full implementation would store architecture
        raise NotImplementedError("CNN/MLP evaluation requires architecture info — use during training")
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    inference_time = time.time() - infer_start

    metrics = compute_metrics(y, y_pred)
    metrics["inference_time_s"] = round(inference_time, 3)
    metrics["split"] = split

    return metrics


def compare_models(metrics_list: list[dict[str, Any]]) -> dict[str, Any]:
    """Generate comparison table between models."""
    table = []
    for m in metrics_list:
        table.append({
            "model_type": m.get("model_type", "unknown"),
            "accuracy": m.get("metrics", {}).get("accuracy", 0),
            "f1_weighted": m.get("metrics", {}).get("f1_weighted", 0),
            "f1_macro": m.get("metrics", {}).get("f1_macro", 0),
            "training_time_s": m.get("metrics", {}).get("training_time_s", 0),
            "inference_time_s": m.get("metrics", {}).get("inference_time_s", 0),
        })

    table.sort(key=lambda x: x["accuracy"], reverse=True)
    best = table[0] if table else None

    # Check quantum vs classical
    quantum_models = [t for t in table if t["model_type"] in ("qsvm", "vqc", "data_reupload")]
    classical_models = [t for t in table if t["model_type"] in ("svm", "cnn", "mlp")]

    quantum_vs_baseline = {}
    if quantum_models and classical_models:
        best_quantum = max(quantum_models, key=lambda x: x["accuracy"])
        best_classical = max(classical_models, key=lambda x: x["accuracy"])
        quantum_vs_baseline = {
            "best_quantum": best_quantum,
            "best_classical": best_classical,
            "quantum_advantage": best_quantum["accuracy"] > best_classical["accuracy"],
            "accuracy_delta": best_quantum["accuracy"] - best_classical["accuracy"],
        }

    return {
        "comparison_table": table,
        "best_model": best,
        "quantum_vs_baseline": quantum_vs_baseline,
    }


def _file_hash(filepath: str) -> str:
    """SHA-256 hash of file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

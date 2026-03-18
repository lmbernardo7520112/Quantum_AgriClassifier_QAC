"""QAC — Preprocessing module.

Handles PCA dimensionality reduction and quantum feature normalization.
Separated from data_loader following clean architecture principles.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from qac.data_loader import DatasetResult


def apply_pca(dataset: DatasetResult, n_components: int, seed: int = 42) -> DatasetResult:
    """Apply PCA dimensionality reduction for quantum pipelines.

    Reduces features to n_components (must be ≤ 10 for Bloco 3 compliance).

    Args:
        dataset: DatasetResult with raw flattened features.
        n_components: Number of PCA components (≤10).
        seed: Random seed for PCA.

    Returns:
        New DatasetResult with PCA-reduced features.
    """
    if n_components > 10:
        raise ValueError(
            f"n_components must be ≤ 10 (Bloco 3 constraint), got {n_components}"
        )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(dataset.X_train)
    X_val_scaled = scaler.transform(dataset.X_val)
    X_test_scaled = scaler.transform(dataset.X_test)

    pca = PCA(n_components=n_components, random_state=seed)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    explained_var = float(pca.explained_variance_ratio_.sum())

    return DatasetResult(
        X_train=X_train_pca,
        X_val=X_val_pca,
        X_test=X_test_pca,
        y_train=dataset.y_train,
        y_val=dataset.y_val,
        y_test=dataset.y_test,
        class_names=dataset.class_names,
        dataset_hash=dataset.dataset_hash,
        metadata={
            **dataset.metadata,
            "pca_components": n_components,
            "pca_explained_variance_ratio": round(explained_var, 4),
        },
    )


def normalize_features(X: np.ndarray) -> np.ndarray:
    """Normalize features to [0, 2π] range for quantum angle encoding.

    Args:
        X: Feature array of shape (n_samples, n_features).

    Returns:
        Normalized array with values in [0, 2π].
    """
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range == 0] = 1.0  # avoid division by zero
    return (X - X_min) / X_range * 2 * np.pi

"""Tests for QAC preprocessing — PCA and normalization."""
import numpy as np
import pytest

from qac.data_loader import DatasetResult
from qac.preprocessing import apply_pca, normalize_features


def _make_synthetic_dataset(n_samples=50, n_features=100, n_classes=2, seed=42):
    """Create a synthetic DatasetResult for testing."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features).astype(np.float32)
    y = rng.randint(0, n_classes, n_samples)

    # Split 60/20/20
    n_train = int(n_samples * 0.6)
    n_val = int(n_samples * 0.2)

    return DatasetResult(
        X_train=X[:n_train],
        X_val=X[n_train:n_train + n_val],
        X_test=X[n_train + n_val:],
        y_train=y[:n_train],
        y_val=y[n_train:n_train + n_val],
        y_test=y[n_train + n_val:],
        class_names=tuple(f"class_{i}" for i in range(n_classes)),
        dataset_hash="test_hash",
        metadata={"test": True},
    )


class TestPCA:
    def test_reduces_dimensionality(self):
        dataset = _make_synthetic_dataset(n_features=100)
        reduced = apply_pca(dataset, n_components=8)
        assert reduced.feature_dim == 8
        assert reduced.X_train.shape[1] == 8
        assert reduced.X_val.shape[1] == 8
        assert reduced.X_test.shape[1] == 8

    def test_preserves_labels(self):
        dataset = _make_synthetic_dataset()
        reduced = apply_pca(dataset, n_components=4)
        np.testing.assert_array_equal(reduced.y_train, dataset.y_train)
        np.testing.assert_array_equal(reduced.y_test, dataset.y_test)

    def test_metadata_updated(self):
        dataset = _make_synthetic_dataset()
        reduced = apply_pca(dataset, n_components=8)
        assert reduced.metadata["pca_components"] == 8
        assert "pca_explained_variance_ratio" in reduced.metadata

    def test_exceeds_10_raises(self):
        dataset = _make_synthetic_dataset(n_features=20)
        with pytest.raises(ValueError, match="≤ 10"):
            apply_pca(dataset, n_components=11)

    def test_preserves_sample_count(self):
        dataset = _make_synthetic_dataset(n_samples=50)
        reduced = apply_pca(dataset, n_components=4)
        assert len(reduced.y_train) == len(dataset.y_train)
        assert len(reduced.y_test) == len(dataset.y_test)


class TestNormalize:
    def test_output_range(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        X_norm = normalize_features(X)
        assert X_norm.min() >= 0.0
        assert X_norm.max() <= 2 * np.pi + 1e-6

    def test_constant_feature(self):
        """Constant columns should not cause division by zero."""
        X = np.array([[1.0, 5.0], [1.0, 10.0], [1.0, 15.0]])
        X_norm = normalize_features(X)
        assert not np.any(np.isnan(X_norm))
        assert not np.any(np.isinf(X_norm))

    def test_shape_preserved(self):
        X = np.random.randn(20, 8)
        X_norm = normalize_features(X)
        assert X_norm.shape == X.shape

"""Tests for QAC config module — invariants, serialization, validation.

Mirrors test_config.py from vqe_knapsack_local.
"""
import json
import tempfile
from pathlib import Path

import pytest

from qac.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    DEFAULT_DATA_CONFIG,
    DEFAULT_EXPERIMENT_CONFIG,
    DEFAULT_VQC_CONFIG,
    DEFAULT_SVM_CONFIG,
    FAST_EXPERIMENT_CONFIG,
)


class TestDataConfig:
    """Test DataConfig invariants and serialization."""

    def test_valid_creation(self):
        """Valid DataConfig should not raise."""
        cfg = DataConfig(
            dataset_name="eurosat_rgb",
            dataset_path="datasets/eurosat",
            classes=("AnnualCrop", "SeaLake"),
        )
        assert cfg.dataset_name == "eurosat_rgb"
        assert cfg.n_pca_components == 8

    def test_empty_classes_raises(self):
        with pytest.raises(ValueError, match="classes must be non-empty"):
            DataConfig(
                dataset_name="eurosat_rgb",
                dataset_path="test",
                classes=(),
            )

    def test_pca_exceeds_10_raises(self):
        with pytest.raises(ValueError, match="n_pca_components must be in"):
            DataConfig(
                dataset_name="eurosat_rgb",
                dataset_path="test",
                classes=("A",),
                n_pca_components=11,
            )

    def test_pca_zero_raises(self):
        with pytest.raises(ValueError):
            DataConfig(
                dataset_name="eurosat_rgb",
                dataset_path="test",
                classes=("A",),
                n_pca_components=0,
            )

    def test_max_samples_one_raises(self):
        with pytest.raises(ValueError, match="max_samples must be >= 2"):
            DataConfig(
                dataset_name="eurosat_rgb",
                dataset_path="test",
                classes=("A",),
                max_samples=1,
            )

    def test_to_dict(self):
        d = DEFAULT_DATA_CONFIG.to_dict()
        assert d["dataset_name"] == "eurosat_rgb"
        assert d["classes"] == ["AnnualCrop", "SeaLake"]
        assert d["n_pca_components"] == 8

    def test_frozen(self):
        with pytest.raises(AttributeError):
            DEFAULT_DATA_CONFIG.seed = 99  # type: ignore


class TestModelConfig:
    """Test ModelConfig invariants."""

    def test_valid_vqc(self):
        cfg = ModelConfig(model_type="vqc", n_qubits=8)
        assert cfg.ansatz_type == "real_amplitudes"

    def test_valid_svm(self):
        cfg = ModelConfig(model_type="svm")
        assert cfg.model_type == "svm"

    def test_invalid_model_type_raises(self):
        with pytest.raises(ValueError, match="model_type"):
            ModelConfig(model_type="cnn")

    def test_vqc_qubits_exceeds_10_raises(self):
        with pytest.raises(ValueError, match="n_qubits must be in"):
            ModelConfig(model_type="vqc", n_qubits=12)

    def test_vqc_invalid_ansatz_raises(self):
        with pytest.raises(ValueError, match="ansatz_type"):
            ModelConfig(model_type="vqc", ansatz_type="unknown")

    def test_vqc_invalid_optimizer_raises(self):
        with pytest.raises(ValueError, match="optimizer_type"):
            ModelConfig(model_type="vqc", optimizer_type="adam")

    def test_vqc_invalid_feature_map_raises(self):
        with pytest.raises(ValueError, match="feature_map_type"):
            ModelConfig(model_type="vqc", feature_map_type="unknown")

    def test_max_iter_zero_raises(self):
        with pytest.raises(ValueError, match="max_iter"):
            ModelConfig(model_type="vqc", max_iter=0)

    def test_to_dict_roundtrip(self):
        d = DEFAULT_VQC_CONFIG.to_dict()
        assert d["model_type"] == "vqc"
        assert d["n_qubits"] == 8


class TestExperimentConfig:
    """Test ExperimentConfig composition and JSON serialization."""

    def test_valid_creation(self):
        cfg = DEFAULT_EXPERIMENT_CONFIG
        assert len(cfg.models) == 2
        assert cfg.output_dir == "outputs"

    def test_empty_models_raises(self):
        with pytest.raises(ValueError, match="models must be non-empty"):
            ExperimentConfig(data=DEFAULT_DATA_CONFIG, models=())

    def test_to_dict(self):
        d = DEFAULT_EXPERIMENT_CONFIG.to_dict()
        assert "data" in d
        assert "models" in d
        assert len(d["models"]) == 2

    def test_json_roundtrip(self):
        """Test from_json loads correctly."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(DEFAULT_EXPERIMENT_CONFIG.to_dict(), f)
            f.flush()
            loaded = ExperimentConfig.from_json(f.name)

        assert loaded.data.dataset_name == "eurosat_rgb"
        assert len(loaded.models) == 2
        assert loaded.models[0].model_type == "svm"
        assert loaded.models[1].model_type == "vqc"

    def test_defaults_are_valid(self):
        """All default configs should be valid."""
        assert DEFAULT_EXPERIMENT_CONFIG.data.n_pca_components <= 10
        assert FAST_EXPERIMENT_CONFIG.data.n_pca_components <= 10

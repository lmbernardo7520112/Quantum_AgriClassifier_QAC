"""QAC — Immutable configuration dataclasses for experiment specification.

All configs are frozen to guarantee reproducibility.
Invariants are enforced in __post_init__.

Mirrors the design of KnapsackConfig in vqe_knapsack_local.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DataConfig:
    """Configuration for dataset loading and preprocessing.

    Invariants:
    - max_samples >= 2
    - n_pca_components in [1, 10]  (Bloco 3: ≤10 features)
    - img_size > 0
    - classes must be non-empty
    """

    dataset_name: str
    dataset_path: str
    classes: tuple[str, ...]
    max_samples: int = 100
    img_size: int = 64
    n_pca_components: int = 8
    seed: int = 42

    def __post_init__(self) -> None:
        if not self.classes:
            raise ValueError("classes must be non-empty")
        if self.max_samples < 2:
            raise ValueError(f"max_samples must be >= 2, got {self.max_samples}")
        if not 1 <= self.n_pca_components <= 10:
            raise ValueError(
                f"n_pca_components must be in [1, 10] (Bloco 3 constraint), "
                f"got {self.n_pca_components}"
            )
        if self.img_size <= 0:
            raise ValueError(f"img_size must be > 0, got {self.img_size}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "dataset_path": self.dataset_path,
            "classes": list(self.classes),
            "max_samples": self.max_samples,
            "img_size": self.img_size,
            "n_pca_components": self.n_pca_components,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for a single quantum or classical model.

    Invariants:
    - n_qubits in [1, 10]
    - max_iter >= 1
    - ansatz_type in known set
    - feature_map_type in known set
    - optimizer_type in known set
    """

    model_type: str  # "vqc" or "svm"
    n_qubits: int = 8
    ansatz_type: str = "real_amplitudes"
    feature_map_type: str = "zz"
    optimizer_type: str = "cobyla"
    max_iter: int = 100
    seed: int = 42

    VALID_MODEL_TYPES = ("vqc", "svm")
    VALID_ANSATZ_TYPES = ("real_amplitudes", "efficient_su2")
    VALID_FEATURE_MAP_TYPES = ("zz", "z", "pauli")
    VALID_OPTIMIZER_TYPES = ("cobyla", "spsa")

    def __post_init__(self) -> None:
        if self.model_type not in self.VALID_MODEL_TYPES:
            raise ValueError(
                f"model_type must be one of {self.VALID_MODEL_TYPES}, got '{self.model_type}'"
            )
        if self.model_type == "vqc":
            if not 1 <= self.n_qubits <= 10:
                raise ValueError(f"n_qubits must be in [1, 10], got {self.n_qubits}")
            if self.ansatz_type not in self.VALID_ANSATZ_TYPES:
                raise ValueError(
                    f"ansatz_type must be one of {self.VALID_ANSATZ_TYPES}, "
                    f"got '{self.ansatz_type}'"
                )
            if self.feature_map_type not in self.VALID_FEATURE_MAP_TYPES:
                raise ValueError(
                    f"feature_map_type must be one of {self.VALID_FEATURE_MAP_TYPES}, "
                    f"got '{self.feature_map_type}'"
                )
            if self.optimizer_type not in self.VALID_OPTIMIZER_TYPES:
                raise ValueError(
                    f"optimizer_type must be one of {self.VALID_OPTIMIZER_TYPES}, "
                    f"got '{self.optimizer_type}'"
                )
            if self.max_iter < 1:
                raise ValueError(f"max_iter must be >= 1, got {self.max_iter}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "n_qubits": self.n_qubits,
            "ansatz_type": self.ansatz_type,
            "feature_map_type": self.feature_map_type,
            "optimizer_type": self.optimizer_type,
            "max_iter": self.max_iter,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level experiment configuration.

    Composes DataConfig with one or more ModelConfigs.
    Invariants:
    - models must be non-empty
    """

    data: DataConfig
    models: tuple[ModelConfig, ...]
    output_dir: str = "outputs"

    def __post_init__(self) -> None:
        if not self.models:
            raise ValueError("models must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "data": self.data.to_dict(),
            "models": [m.to_dict() for m in self.models],
            "output_dir": self.output_dir,
        }

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        """Load ExperimentConfig from a JSON file."""
        with open(path) as f:
            raw = json.load(f)

        data_raw = raw["data"]
        data_config = DataConfig(
            dataset_name=data_raw["dataset_name"],
            dataset_path=data_raw["dataset_path"],
            classes=tuple(data_raw["classes"]),
            max_samples=data_raw.get("max_samples", 100),
            img_size=data_raw.get("img_size", 64),
            n_pca_components=data_raw.get("n_pca_components", 8),
            seed=data_raw.get("seed", 42),
        )

        model_configs = tuple(
            ModelConfig(
                model_type=m["model_type"],
                n_qubits=m.get("n_qubits", 8),
                ansatz_type=m.get("ansatz_type", "real_amplitudes"),
                feature_map_type=m.get("feature_map_type", "zz"),
                optimizer_type=m.get("optimizer_type", "cobyla"),
                max_iter=m.get("max_iter", 100),
                seed=m.get("seed", 42),
            )
            for m in raw["models"]
        )

        return cls(
            data=data_config,
            models=model_configs,
            output_dir=raw.get("output_dir", "outputs"),
        )


# ── Default Configurations ──────────────────────────────────────────────────

DEFAULT_DATA_CONFIG = DataConfig(
    dataset_name="eurosat_rgb",
    dataset_path="datasets/eurosat",
    classes=("AnnualCrop", "SeaLake"),
    max_samples=100,
    n_pca_components=8,
    seed=42,
)

DEFAULT_VQC_CONFIG = ModelConfig(
    model_type="vqc",
    n_qubits=8,
    ansatz_type="real_amplitudes",
    feature_map_type="zz",
    optimizer_type="cobyla",
    max_iter=100,
    seed=42,
)

DEFAULT_SVM_CONFIG = ModelConfig(
    model_type="svm",
    seed=42,
)

DEFAULT_EXPERIMENT_CONFIG = ExperimentConfig(
    data=DEFAULT_DATA_CONFIG,
    models=(DEFAULT_SVM_CONFIG, DEFAULT_VQC_CONFIG),
    output_dir="outputs",
)

# Fast configuration for CI/testing
FAST_DATA_CONFIG = DataConfig(
    dataset_name="eurosat_rgb",
    dataset_path="datasets/eurosat",
    classes=("AnnualCrop", "SeaLake"),
    max_samples=20,
    img_size=32,
    n_pca_components=4,
    seed=42,
)

FAST_VQC_CONFIG = ModelConfig(
    model_type="vqc",
    n_qubits=4,
    ansatz_type="real_amplitudes",
    feature_map_type="zz",
    optimizer_type="cobyla",
    max_iter=20,
    seed=42,
)

FAST_EXPERIMENT_CONFIG = ExperimentConfig(
    data=FAST_DATA_CONFIG,
    models=(ModelConfig(model_type="svm", seed=42), FAST_VQC_CONFIG),
    output_dir="outputs",
)

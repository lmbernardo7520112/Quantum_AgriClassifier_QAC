"""QAC — Data Loader.

Loads and splits datasets for the QAC pipeline.
Supports EuroSAT RGB with class filtering for Bloco 3 compliance.

All datasets are hashed (SHA-256) for reproducibility tracking.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from qac.config import DataConfig


@dataclass(frozen=True)
class DatasetResult:
    """Structured dataset result for pipeline consumption.

    Immutable — guarantees no accidental mutation during experiments.
    """

    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    class_names: tuple[str, ...]
    dataset_hash: str
    metadata: dict[str, Any]

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def feature_dim(self) -> int:
        return int(self.X_train.shape[1]) if self.X_train.ndim > 1 else 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_classes": self.num_classes,
            "class_names": list(self.class_names),
            "dataset_hash": self.dataset_hash,
            "feature_dim": self.feature_dim,
            "splits": {
                "train": len(self.y_train),
                "val": len(self.y_val),
                "test": len(self.y_test),
            },
            "metadata": self.metadata,
        }


def load_eurosat_rgb(config: DataConfig) -> DatasetResult:
    """Load EuroSAT RGB dataset with class filtering.

    Expected structure: dataset_path/{class_name}/*.jpg

    Args:
        config: DataConfig specifying classes, max_samples, img_size, and seed.

    Returns:
        DatasetResult with flattened RGB images and stratified train/val/test splits.
    """
    dataset_path = Path(config.dataset_path)

    # Find EuroSAT subdirectory if needed
    if not any(dataset_path.iterdir()):
        raise FileNotFoundError(f"Empty dataset directory: {dataset_path}")

    # Check if classes exist as subdirectories
    available_dirs = {d.name: d for d in sorted(dataset_path.iterdir()) if d.is_dir()}

    # If the path has a single subdirectory containing the class dirs, descend
    if not any(c in available_dirs for c in config.classes):
        for subdir in dataset_path.iterdir():
            if subdir.is_dir():
                sub_available = {d.name: d for d in sorted(subdir.iterdir()) if d.is_dir()}
                if any(c in sub_available for c in config.classes):
                    dataset_path = subdir
                    available_dirs = sub_available
                    break

    # Filter to requested classes
    selected_classes = [c for c in config.classes if c in available_dirs]
    if not selected_classes:
        raise FileNotFoundError(
            f"None of the requested classes {config.classes} found in {dataset_path}. "
            f"Available: {list(available_dirs.keys())}"
        )

    images: list[np.ndarray] = []
    labels: list[str] = []
    max_per_class = config.max_samples // len(selected_classes)

    for class_name in selected_classes:
        class_dir = available_dirs[class_name]
        class_count = 0
        for img_file in sorted(class_dir.iterdir()):
            if img_file.suffix.lower() in (".jpg", ".jpeg", ".png"):
                try:
                    img = Image.open(img_file).convert("RGB").resize(
                        (config.img_size, config.img_size)
                    )
                    img_array = np.array(img, dtype=np.float32).flatten() / 255.0
                    images.append(img_array)
                    labels.append(class_name)
                    class_count += 1
                except Exception:
                    continue
                if class_count >= max_per_class:
                    break

    if len(images) < 2:
        raise ValueError(f"Not enough images loaded: {len(images)}")

    X = np.array(images)
    label_encoder = LabelEncoder()
    label_encoder.fit(selected_classes)
    y = label_encoder.transform(labels)

    # Deterministic SHA-256 hash
    dataset_hash = _hash_arrays(X, y)

    # Stratified split: 60% train / 20% val / 20% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=config.seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=config.seed, stratify=y_temp
    )

    return DatasetResult(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        class_names=tuple(selected_classes),
        dataset_hash=dataset_hash,
        metadata={
            "dataset": config.dataset_name,
            "img_size": config.img_size,
            "total_samples": len(y),
            "classes": list(selected_classes),
            "seed": config.seed,
        },
    )


def load_dataset(config: DataConfig) -> DatasetResult:
    """Unified dataset loader driven by DataConfig.

    Args:
        config: DataConfig specifying the dataset to load.

    Returns:
        DatasetResult ready for preprocessing.
    """
    if config.dataset_name == "eurosat_rgb":
        return load_eurosat_rgb(config)
    else:
        raise ValueError(
            f"Unknown dataset: {config.dataset_name}. Supported: eurosat_rgb"
        )


def _hash_arrays(*arrays: np.ndarray) -> str:
    """Compute deterministic SHA-256 hash over numpy arrays."""
    sha256 = hashlib.sha256()
    for arr in arrays:
        sha256.update(arr.tobytes())
    return sha256.hexdigest()

"""
QAC — Data Loader.

Loads, preprocesses, and hashes datasets for the QAC pipeline.
Supports:
- EuroSAT (Multispectral 13-band .tif + RGB .jpg)
- PlantVillage (HuggingFace datasets + local fallback)
- Callisto (placeholder)

All datasets are hashed (SHA-256) and registered as resource.dataset.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ─────────────────── PlantVillage Class Names ──────────

PLANTVILLAGE_CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

EUROSAT_CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]


class DatasetResult:
    """Structured dataset result for pipeline consumption."""

    def __init__(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
        class_names: list[str],
        dataset_hash: str,
        metadata: dict[str, Any],
    ):
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        self.class_names = class_names
        self.dataset_hash = dataset_hash
        self.metadata = metadata

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def feature_dim(self) -> int:
        return self.X_train.shape[1] if self.X_train.ndim > 1 else 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "dataset_hash": self.dataset_hash,
            "feature_dim": self.feature_dim,
            "splits": {
                "train": len(self.y_train),
                "val": len(self.y_val),
                "test": len(self.y_test),
            },
            "metadata": self.metadata,
        }


# ─────────────────── EuroSAT RGB Loader ────────────────

def load_eurosat_rgb(
    dataset_path: str | Path,
    seed: int = 42,
    img_size: int = 64,
    max_samples: int | None = None,
) -> DatasetResult:
    """
    Load EuroSAT RGB dataset from extracted directory.

    Expected structure: dataset_path/{class_name}/*.jpg
    """
    dataset_path = Path(dataset_path)
    images, labels = [], []
    class_names = sorted([d.name for d in dataset_path.iterdir() if d.is_dir()])

    if not class_names:
        raise FileNotFoundError(f"No class directories found in {dataset_path}")

    label_encoder = LabelEncoder()
    label_encoder.fit(class_names)

    max_per_class = max_samples // len(class_names) if max_samples else None

    for class_dir in sorted(dataset_path.iterdir()):
        if not class_dir.is_dir():
            continue
        class_label = class_dir.name
        class_count = 0
        for img_file in sorted(class_dir.iterdir()):
            if img_file.suffix.lower() in (".jpg", ".jpeg", ".png"):
                try:
                    img = Image.open(img_file).convert("RGB").resize((img_size, img_size))
                    img_array = np.array(img, dtype=np.float32).flatten() / 255.0
                    images.append(img_array)
                    labels.append(class_label)
                    class_count += 1
                except Exception:
                    continue
                if max_per_class and class_count >= max_per_class:
                    break
        if max_samples and len(images) >= max_samples:
            break

    X = np.array(images)
    y = label_encoder.transform(labels)

    # Hash for reproducibility
    hash_str = _hash_arrays(X, y)

    # Split 60/20/20
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp
    )

    return DatasetResult(
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=y_train, y_val=y_val, y_test=y_test,
        class_names=class_names,
        dataset_hash=hash_str,
        metadata={
            "dataset": "eurosat_rgb",
            "img_size": img_size,
            "total_samples": len(y),
            "seed": seed,
        },
    )


# ─────────────────── EuroSAT Multispectral Loader ──────

def load_eurosat_ms(
    dataset_path: str | Path,
    seed: int = 42,
    max_samples: int | None = None,
) -> DatasetResult:
    """
    Load EuroSAT Multispectral dataset (13 bands, .tif files).

    Expected structure: dataset_path/{class_name}/*.tif
    Uses rasterio if available, falls back to tifffile.
    """
    dataset_path = Path(dataset_path)
    images, labels = [], []
    class_names = sorted([d.name for d in dataset_path.iterdir() if d.is_dir()])

    if not class_names:
        raise FileNotFoundError(f"No class directories found in {dataset_path}")

    label_encoder = LabelEncoder()
    label_encoder.fit(class_names)

    try:
        import rasterio
        USE_RASTERIO = True
    except ImportError:
        try:
            import tifffile
            USE_RASTERIO = False
        except ImportError:
            raise ImportError("Install rasterio or tifffile for EuroSAT MS: pip install rasterio")

    for class_dir in sorted(dataset_path.iterdir()):
        if not class_dir.is_dir():
            continue
        class_label = class_dir.name
        for tif_file in sorted(class_dir.iterdir()):
            if tif_file.suffix.lower() in (".tif", ".tiff"):
                try:
                    if USE_RASTERIO:
                        with rasterio.open(tif_file) as src:
                            img = src.read()  # (bands, H, W)
                    else:
                        img = tifffile.imread(str(tif_file))
                        if img.ndim == 3 and img.shape[2] <= 13:
                            img = img.transpose(2, 0, 1)  # (H,W,C) → (C,H,W)

                    # Flatten: (bands, H, W) → (bands*H*W,)
                    img_flat = img.astype(np.float32).flatten()
                    images.append(img_flat)
                    labels.append(class_label)
                except Exception:
                    continue
                if max_samples and len(images) >= max_samples:
                    break
        if max_samples and len(images) >= max_samples:
            break

    X = np.array(images)
    y = label_encoder.transform(labels)

    hash_str = _hash_arrays(X, y)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp
    )

    return DatasetResult(
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=y_train, y_val=y_val, y_test=y_test,
        class_names=class_names,
        dataset_hash=hash_str,
        metadata={
            "dataset": "eurosat_ms",
            "bands": 13,
            "total_samples": len(y),
            "seed": seed,
        },
    )


# ─────────────────── PlantVillage Loader ───────────────

def load_plantvillage_hf(
    seed: int = 42,
    max_samples: int | None = None,
    img_size: int = 64,
) -> DatasetResult:
    """
    Load PlantVillage via HuggingFace datasets (primary mode).
    Uses leaf-grouping aware 80/20 train/test splits.
    """
    from datasets import load_dataset

    ds = load_dataset("mohanty/PlantVillage", "color")

    # Process train split
    X_train_list, y_train_list = [], []
    for i, example in enumerate(ds["train"]):
        if max_samples and i >= int(max_samples * 0.8):
            break
        img = example["image"].convert("RGB").resize((img_size, img_size))
        X_train_list.append(np.array(img, dtype=np.float32).flatten() / 255.0)
        y_train_list.append(example["label"])

    # Process test split
    X_test_list, y_test_list = [], []
    for i, example in enumerate(ds["test"]):
        if max_samples and i >= int(max_samples * 0.2):
            break
        img = example["image"].convert("RGB").resize((img_size, img_size))
        X_test_list.append(np.array(img, dtype=np.float32).flatten() / 255.0)
        y_test_list.append(example["label"])

    X_train_full = np.array(X_train_list)
    y_train_full = np.array(y_train_list)
    X_test = np.array(X_test_list)
    y_test = np.array(y_test_list)

    # Split train into train/val (75/25 of train = 60/20 overall)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.25, random_state=seed, stratify=y_train_full
    )

    hash_str = _hash_arrays(X_train, y_train)

    return DatasetResult(
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=y_train, y_val=y_val, y_test=y_test,
        class_names=PLANTVILLAGE_CLASSES,
        dataset_hash=hash_str,
        metadata={
            "dataset": "plantvillage",
            "source": "huggingface",
            "img_size": img_size,
            "num_classes": 38,
            "total_samples": len(y_train) + len(y_val) + len(y_test),
            "seed": seed,
        },
    )


def load_plantvillage_local(
    dataset_path: str | Path,
    seed: int = 42,
    max_samples: int | None = None,
    img_size: int = 64,
) -> DatasetResult:
    """
    Load PlantVillage from local directory (fallback mode).

    Expected structure: dataset_path/raw/color/{class_name}/*.jpg
    """
    dataset_path = Path(dataset_path)
    color_dir = dataset_path / "raw" / "color"

    if not color_dir.exists():
        # Try direct path if already pointing to color dir
        color_dir = dataset_path
        if not any(color_dir.iterdir()):
            raise FileNotFoundError(f"PlantVillage color directory not found in {dataset_path}")

    images, labels = [], []
    class_dirs = sorted([d for d in color_dir.iterdir() if d.is_dir()])

    if not class_dirs:
        raise FileNotFoundError(f"No class directories in {color_dir}")

    class_names = [d.name for d in class_dirs]
    label_encoder = LabelEncoder()
    label_encoder.fit(class_names)

    for class_dir in class_dirs:
        class_label = class_dir.name
        for img_file in sorted(class_dir.iterdir()):
            if img_file.suffix.lower() in (".jpg", ".jpeg", ".png"):
                try:
                    img = Image.open(img_file).convert("RGB").resize((img_size, img_size))
                    img_array = np.array(img, dtype=np.float32).flatten() / 255.0
                    images.append(img_array)
                    labels.append(class_label)
                except Exception:
                    continue
                if max_samples and len(images) >= max_samples:
                    break
        if max_samples and len(images) >= max_samples:
            break

    X = np.array(images)
    y = label_encoder.transform(labels)

    hash_str = _hash_arrays(X, y)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp
    )

    return DatasetResult(
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=y_train, y_val=y_val, y_test=y_test,
        class_names=class_names,
        dataset_hash=hash_str,
        metadata={
            "dataset": "plantvillage",
            "source": "local",
            "img_size": img_size,
            "num_classes": len(class_names),
            "total_samples": len(y),
            "seed": seed,
        },
    )


# ─────────────────── PCA Preprocessing ─────────────────

def apply_pca(
    dataset: DatasetResult,
    n_components: int,
    seed: int = 42,
) -> DatasetResult:
    """
    Apply PCA dimensionality reduction for quantum pipelines.
    Reduces features to n_components (must be ≤ n_qubits).
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(dataset.X_train)
    X_val_scaled = scaler.transform(dataset.X_val)
    X_test_scaled = scaler.transform(dataset.X_test)

    pca = PCA(n_components=n_components, random_state=seed)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    explained_var = pca.explained_variance_ratio_.sum()

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
            "pca_explained_variance_ratio": float(explained_var),
        },
    )


# ─────────────────── Unified Loader ────────────────────

def load_dataset(
    dataset_name: str,
    dataset_path: str | Path | None = None,
    seed: int = 42,
    max_samples: int | None = None,
    img_size: int = 64,
) -> DatasetResult:
    """
    Unified dataset loader.

    Args:
        dataset_name: One of 'eurosat_rgb', 'eurosat_ms', 'plantvillage'
        dataset_path: Path to dataset (required for eurosat, optional for plantvillage)
        seed: Random seed for splits
        max_samples: Limit number of samples (for development)
        img_size: Target image size (for RGB datasets)
    """
    if dataset_name == "eurosat_rgb":
        if not dataset_path:
            raise ValueError("dataset_path required for eurosat_rgb")
        return load_eurosat_rgb(dataset_path, seed=seed, max_samples=max_samples, img_size=img_size)

    elif dataset_name == "eurosat_ms":
        if not dataset_path:
            raise ValueError("dataset_path required for eurosat_ms")
        return load_eurosat_ms(dataset_path, seed=seed, max_samples=max_samples)

    elif dataset_name == "plantvillage":
        if dataset_path:
            # Try local first
            try:
                return load_plantvillage_local(
                    dataset_path, seed=seed, max_samples=max_samples, img_size=img_size
                )
            except Exception:
                pass
        # Fall back to HuggingFace
        try:
            return load_plantvillage_hf(seed=seed, max_samples=max_samples, img_size=img_size)
        except Exception as e:
            if dataset_path:
                # Try local as last resort
                return load_plantvillage_local(
                    dataset_path, seed=seed, max_samples=max_samples, img_size=img_size
                )
            raise ValueError(f"Cannot load PlantVillage: {e}. Provide dataset_path for local loading.")

    elif dataset_name == "callisto":
        raise NotImplementedError("Callisto dataset loader not yet implemented")

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from: eurosat_rgb, eurosat_ms, plantvillage, callisto")


# ─────────────────── Helpers ───────────────────────────

def _hash_arrays(*arrays: np.ndarray) -> str:
    """Compute deterministic SHA-256 hash over numpy arrays."""
    sha256 = hashlib.sha256()
    for arr in arrays:
        sha256.update(arr.tobytes())
    return sha256.hexdigest()

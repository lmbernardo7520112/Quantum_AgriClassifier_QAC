"""
QAC — HuggingFace Dataset Upload/Maintenance Script.

Handles zipping, uploading, and updating the PlantVillage dataset on
HuggingFace Hub. Run this script after modifying metadata or adding images.

Prerequisites:
    pip install huggingface_hub
    huggingface-cli login

Usage:
    python scripts/upload_to_hf.py [--repo-id mohanty/PlantVillage] [--data-dir /path/to/raw]
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import zipfile
from pathlib import Path


def compute_dir_hash(dirpath: Path) -> str:
    """Compute SHA-256 hash of all files in directory."""
    sha256 = hashlib.sha256()
    for fpath in sorted(dirpath.rglob("*")):
        if fpath.is_file():
            sha256.update(fpath.relative_to(dirpath).as_posix().encode())
            with open(fpath, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256.update(chunk)
    return sha256.hexdigest()


def zip_data(data_dir: Path, output_path: Path) -> str:
    """Zip the data directory and return the zip hash."""
    print(f"📦 Zipping {data_dir} → {output_path}")
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fpath in sorted(data_dir.rglob("*")):
            if fpath.is_file():
                arcname = fpath.relative_to(data_dir.parent).as_posix()
                zf.write(fpath, arcname)
    print(f"✅ Created zip: {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")
    return compute_dir_hash(data_dir)


def upload_to_hf(
    repo_id: str,
    data_zip: Path,
    readme_path: Path | None = None,
):
    """Upload files to HuggingFace Hub."""
    from huggingface_hub import HfApi

    api = HfApi()

    # Upload data zip
    print(f"📤 Uploading {data_zip.name} to {repo_id}...")
    api.upload_file(
        path_or_fileobj=str(data_zip),
        path_in_repo="data.zip",
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"✅ Uploaded data.zip")

    # Upload README if provided
    if readme_path and readme_path.exists():
        print(f"📤 Uploading README...")
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
        )
        print(f"✅ Uploaded README.md")


def main():
    parser = argparse.ArgumentParser(description="Upload PlantVillage dataset to HuggingFace Hub")
    parser.add_argument(
        "--repo-id", default="mohanty/PlantVillage",
        help="HuggingFace repo ID (default: mohanty/PlantVillage)",
    )
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path(r"C:\Users\USER\Downloads\Quantum_AgriClassifier_QAC_dataset\PlantVillage-Dataset\raw"),
        help="Path to raw data directory",
    )
    parser.add_argument(
        "--readme", type=Path,
        default=Path(r"C:\Users\USER\Downloads\Quantum_AgriClassifier_QAC_dataset\PlantVillage-Dataset\README_HF.md"),
        help="Path to HuggingFace README",
    )
    parser.add_argument(
        "--output-zip", type=Path, default=None,
        help="Output zip path (default: data.zip in project root)",
    )
    args = parser.parse_args()

    # Validate
    if not args.data_dir.exists():
        print(f"❌ Data directory not found: {args.data_dir}")
        sys.exit(1)

    # Determine output zip path
    output_zip = args.output_zip or (Path(__file__).parent.parent / "data.zip")

    # Zip
    data_hash = zip_data(args.data_dir, output_zip)
    print(f"📊 Data hash: {data_hash}")

    # Upload
    try:
        upload_to_hf(args.repo_id, output_zip, args.readme if args.readme.exists() else None)
        print(f"\n🎉 Successfully uploaded to https://huggingface.co/datasets/{args.repo_id}")
    except ImportError:
        print("❌ huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        print("💡 Make sure you're logged in: huggingface-cli login")
        sys.exit(1)

    # Cleanup
    if output_zip.exists():
        output_zip.unlink()
        print(f"🧹 Cleaned up {output_zip}")


if __name__ == "__main__":
    main()

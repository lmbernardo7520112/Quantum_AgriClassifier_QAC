#!/usr/bin/env python3
"""
QAC — VQE Phase 2 Execution Script.

Controlled parametric study: VQE binary classification on PlantVillage
(Tomato___healthy vs Tomato___Bacterial_spot) with systematic parameter variation.

This script runs INDEPENDENTLY of the MCP server.
It uses the modules directly and updates the registry atomically.

Governed by: docs/VQE_PHASE2_EXPERIMENTAL_PROTOCOL.md

USAGE:
    # Dry run (validate data loading and grid only):
    PYTHONPATH=. python scripts/run_vqe_phase2.py --dry-run

    # Full execution:
    PYTHONPATH=. python scripts/run_vqe_phase2.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# ─── Project paths ───
# Auto-detect: use env vars or resolve from script location
_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get(
    "QAC_PROJECT_ROOT",
    str(_SCRIPT_DIR.parent),
))
DATASET_ROOT = Path(os.environ.get(
    "QAC_DATASET_ROOT",
    str(PROJECT_ROOT / "datasets" / "raw"),
))
REGISTRY_PATH = PROJECT_ROOT / "registry"
MODELS_DIR = PROJECT_ROOT / "models"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments" / "vqe_phase2"
REPORTS_DIR = PROJECT_ROOT / "docs"

os.environ["QAC_PROJECT_ROOT"] = str(PROJECT_ROOT)
os.environ["QAC_DATASET_ROOT"] = str(DATASET_ROOT)

sys.path.insert(0, str(PROJECT_ROOT))

# ─── Constants ───
SEED = 42
N_QUBITS = 8
MAX_SAMPLES = 500
PHASE_ID = "VQE_PHASE2"
PHASE1_EXPERIMENT_ID = "VQE_PHASE1_BINARY"


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


def sha256_file(filepath: str) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def update_registry(filename: str, key: str, entry: dict) -> None:
    """Atomically update a registry JSON file."""
    path = REGISTRY_PATH / filename
    data = json.loads(path.read_text(encoding="utf-8"))
    collection_key = "experiments" if "experiment" in filename else "resources"
    data[collection_key][key] = entry
    data["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def load_phase1_reference() -> dict | None:
    """Load Phase I results from the registry for comparison."""
    experiments_file = REGISTRY_PATH / "experiments.json"
    if not experiments_file.exists():
        return None
    data = json.loads(experiments_file.read_text(encoding="utf-8"))
    exp = data.get("experiments", {}).get(PHASE1_EXPERIMENT_ID)
    if exp and exp.get("output"):
        return exp["output"]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="VQE Phase II — Parametric Study")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate data loading and grid without running experiments")
    parser.add_argument("--max-samples", type=int, default=MAX_SAMPLES,
                        help=f"Maximum samples per class (default: {MAX_SAMPLES})")
    parser.add_argument("--n-qubits", type=int, default=N_QUBITS,
                        help=f"Number of qubits / PCA components (default: {N_QUBITS})")
    args = parser.parse_args()

    start_time = time.time()
    log("=" * 70)
    log("VQE PHASE II — ESTUDO PARAMÉTRICO CONTROLADO")
    log("=" * 70)
    log(f"Project root:  {PROJECT_ROOT}")
    log(f"Dataset root:  {DATASET_ROOT}")
    log(f"Registry path: {REGISTRY_PATH}")
    log(f"Models dir:    {MODELS_DIR}")
    log(f"Experiments:   {EXPERIMENTS_DIR}")
    log(f"Dry run:       {args.dry_run}")
    log("")

    # ─── Step 1: Load Phase I reference ───
    log("─── Step 1: Loading Phase I reference ───")
    phase1_ref = load_phase1_reference()
    if phase1_ref:
        log(f"  Phase I accuracy: {phase1_ref.get('accuracy', 'N/A')}")
        log(f"  Phase I status:   COMPLETED_WITH_LIMITATIONS")
        baseline_accuracy = float(phase1_ref.get("accuracy", 0.64))
    else:
        log("  ⚠️  Phase I results not found in registry, using default baseline 0.64")
        baseline_accuracy = 0.64
    log("")

    from classical.data_loader import DatasetResult, apply_pca, _hash_arrays, PLANTVILLAGE_CLASSES
    from PIL import Image
    from sklearn.model_selection import train_test_split
    
    # ─── MONKEYPATCH: Loading only 2 target classes directly ───
    # Phase I bypassed data_loader.py to specifically load matched samples.
    # We do the exact same logic here to maintain data equivalence.
    import classical.data_loader
    
    def patched_load_plantvillage_hf(seed=42, max_samples=500, **kwargs):
        color_dir = DATASET_ROOT / "raw" / "color" # Extracted path
        target_classes = ["Tomato___healthy", "Tomato___Bacterial_spot"]
        samples_per_class = max_samples // 2
        
        images, labels = [], []
        for cls_idx, cls_name in enumerate(target_classes):
            cls_dir = color_dir / cls_name
            if not cls_dir.exists():
                raise FileNotFoundError(f"Missing {cls_dir}")
            img_files = sorted([f for f in cls_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")])
            
            np.random.seed(seed)
            selected = np.random.choice(len(img_files), size=min(samples_per_class, len(img_files)), replace=False)
            for idx in selected:
                try:
                    img = Image.open(img_files[idx]).convert("RGB").resize((64, 64))
                    images.append(np.array(img, dtype=np.float32).flatten() / 255.0)
                    labels.append(cls_idx)
                except Exception:
                    continue
                    
        X_all = np.array(images)
        y_all = np.array(labels)
        
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_all, y_all, test_size=0.4, random_state=seed, stratify=y_all
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp
        )
        hash_str = _hash_arrays(X_train, y_train)
        
        return DatasetResult(
            X_train=X_train, X_val=X_val, X_test=X_test,
            y_train=y_train, y_val=y_val, y_test=y_test,
            class_names=target_classes, dataset_hash=hash_str,
            metadata={"dataset": "plantvillage", "source": "local_subset", "img_size": 64, "num_classes": 2, "total_samples": len(y_all), "seed": seed}
        )
        
    classical.data_loader.load_plantvillage_hf = patched_load_plantvillage_hf
    # ────────────────────────────────────────────────────────────────────────────────

    # Load the exactly balanced target class dataset
    dataset = classical.data_loader.load_plantvillage_hf(
        seed=SEED,
        max_samples=args.max_samples,
    )

    log(f"  Classes: {dataset.class_names}")
    log(f"  Train: {dataset.X_train.shape}")
    log(f"  Test:  {dataset.X_test.shape}")
    log(f"  Hash:  {dataset.dataset_hash}")
    log("")

    # ─── Step 3: Apply PCA ───
    log(f"─── Step 3: Applying PCA (n_components={args.n_qubits}) ───")
    dataset = apply_pca(dataset, n_components=args.n_qubits)
    log(f"  Train (PCA): {dataset.X_train.shape}")
    log(f"  Test  (PCA): {dataset.X_test.shape}")
    log("")

    # ─── Step 4: Build parametric grid ───
    log("─── Step 4: Building parametric grid ───")
    from quantum.vqe_parametric_runner import build_default_grid, ParametricConfig

    grid = build_default_grid()
    # Override n_qubits
    for config in grid:
        config.n_qubits = args.n_qubits

    seeds = [42, 123, 7]
    total_experiments = len(grid) * len(seeds)

    log(f"  Configurations: {len(grid)}")
    log(f"  Seeds per config: {seeds}")
    log(f"  Total experiments: {total_experiments}")
    log("")

    for i, config in enumerate(grid, 1):
        log(f"  [{i:2d}] {config.config_id}")
    log("")

    if args.dry_run:
        log("=" * 70)
        log("DRY RUN COMPLETE — Data loaded and grid validated successfully")
        log("=" * 70)
        log(f"To execute the full study, run:")
        log(f"  PYTHONPATH=. python scripts/run_vqe_phase2.py")
        return

    # ─── Step 5: Create output directories ───
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ─── Step 6: Run parametric grid ───
    log("─── Step 6: Running parametric grid ───")
    log(f"  ⏱  Starting {total_experiments} experiments...")
    log("")

    from quantum.vqe_parametric_runner import run_parametric_grid

    results = run_parametric_grid(
        dataset=dataset,
        grid=grid,
        seeds=seeds,
        models_dir=str(MODELS_DIR),
        results_dir=str(EXPERIMENTS_DIR),
        phase_id=PHASE_ID,
        log_fn=log,
    )

    completed = [r for r in results if r.status == "COMPLETED"]
    failed = [r for r in results if r.status == "FAILED"]

    log("")
    log(f"  ✅ Completed: {len(completed)}/{total_experiments}")
    log(f"  ❌ Failed:    {len(failed)}/{total_experiments}")
    log("")

    # ─── Step 7: Update registry ───
    log("─── Step 7: Updating registry ───")

    for result in completed:
        # Register experiment
        experiment_entry = {
            "experiment_id": result.experiment_id,
            "tool_name": "tool.train_vqe_classifier",
            "status": "COMPLETED",
            "input": {
                "dataset_name": "plantvillage",
                "classes_filter": ["Tomato___healthy", "Tomato___Bacterial_spot"],
                "max_samples": args.max_samples,
                "n_qubits": args.n_qubits,
                "seed": result.seed,
                "ansatz": result.config.ansatz_type,
                "ansatz_reps": result.config.ansatz_reps,
                "optimizer": result.config.optimizer_type,
                "max_iter": result.config.max_iter,
                "coupling_strength": result.config.coupling_strength,
                "transverse_field": result.config.transverse_field,
                "backend": result.config.backend,
                "phase": "VQE_PHASE2",
            },
            "output": {
                "accuracy": result.accuracy,
                "f1_score": result.f1_score,
                "delta_E": result.delta_E,
                "cohens_d": result.cohens_d,
                "p_ttest": result.p_value_ttest,
                "p_perm": result.p_value_permutation,
                "convergence": result.convergence,
                "vqe_energies": result.vqe_energies,
                "training_time_s": result.training_time_s,
            },
            "error": None,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "context": {
                "seed": result.seed,
                "dataset_hash": dataset.dataset_hash,
                "backend": result.config.backend,
            },
            "resources_created": [
                {"type": "resource.model", "path": result.model_path},
                {"type": "resource.metrics", "id": result.experiment_id},
            ],
        }
        update_registry("experiments.json", result.experiment_id, experiment_entry)

        # Register model
        model_entry = {
            "resource_id": f"model-{result.experiment_id}",
            "resource_type": "resource.model",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "file_path": result.model_path,
            "file_hash": result.model_hash,
            "experiment_id": result.experiment_id,
            "metadata": {
                "model_type": "vqe_classifier",
                "optimizer": result.config.optimizer_type,
                "ansatz_reps": result.config.ansatz_reps,
                "coupling_strength": result.config.coupling_strength,
                "transverse_field": result.config.transverse_field,
                "seed": result.seed,
                "accuracy": result.accuracy,
                "phase": "VQE_PHASE2",
            },
        }
        update_registry("models.json", f"model-{result.experiment_id}", model_entry)

        # Register metrics
        metrics_entry = {
            "resource_id": f"metrics-{result.experiment_id}",
            "resource_type": "resource.metrics",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "experiment_id": result.experiment_id,
            "metadata": {
                "accuracy": result.accuracy,
                "f1_score": result.f1_score,
                "delta_E": result.delta_E,
                "cohens_d": result.cohens_d,
                "p_ttest": result.p_value_ttest,
                "p_perm": result.p_value_permutation,
                "var_intra_class0": result.var_intra_class0,
                "var_intra_class1": result.var_intra_class1,
                "training_time_s": result.training_time_s,
                "inference_time_s": result.inference_time_s,
                "phase": "VQE_PHASE2",
            },
        }
        update_registry("metrics.json", f"metrics-{result.experiment_id}", metrics_entry)

    log(f"  Registered {len(completed)} experiments in registry")
    log("")

    # ─── Step 8: Consolidate and analyze ───
    log("─── Step 8: Consolidating results and generating report ───")

    from quantum.vqe_analysis import (
        consolidate_results,
        evaluate_superiority_criteria,
        generate_report_markdown,
    )

    consolidated = consolidate_results(str(EXPERIMENTS_DIR))
    evaluation = evaluate_superiority_criteria(
        consolidated,
        baseline_accuracy=baseline_accuracy,
    )

    phase1_ref_metrics = None
    if phase1_ref:
        phase1_ref_metrics = {
            "accuracy": phase1_ref.get("accuracy"),
            "f1_score": phase1_ref.get("f1_score"),
            "delta_E": phase1_ref.get("delta_E"),
            "p_ttest": phase1_ref.get("p_ttest"),
            "p_perm": phase1_ref.get("p_perm"),
        }

    report_md = generate_report_markdown(
        consolidated=consolidated,
        evaluation=evaluation,
        phase1_reference=phase1_ref_metrics,
    )

    # Save report
    report_path = REPORTS_DIR / "VQE_PHASE2_FINAL_REPORT.md"
    report_path.write_text(report_md, encoding="utf-8")
    log(f"  Report saved: {report_path}")

    # ─── Step 9: Final summary ───
    elapsed = time.time() - start_time
    log("")
    log("=" * 70)
    log(f"VQE PHASE II — CONCLUÍDO")
    log("=" * 70)
    log(f"  Tempo total:        {elapsed:.1f}s")
    log(f"  Experimentos:       {len(completed)}/{total_experiments}")
    log(f"  Melhor accuracy:    {evaluation.get('best_accuracy', 'N/A')}")
    log(f"  Melhor config:      {evaluation.get('best_config', 'N/A')}")
    log(f"  Veredito:           {evaluation.get('verdict', 'N/A')}")
    log(f"  Critérios:          {evaluation.get('n_passed', 0)}/{evaluation.get('n_total', 0)} passed")
    log(f"  Relatório:          {report_path}")
    log("=" * 70)


if __name__ == "__main__":
    main()

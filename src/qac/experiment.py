"""QAC — Experiment Runner.

Orchestrates the full classification pipeline:
1. Load dataset → 2. Preprocess (PCA) → 3. Train models → 4. Evaluate → 5. Compare

Mirrors ExperimentRunner from vqe_knapsack_local.
"""
from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from qac.classical_baseline import train_svm
from qac.config import ExperimentConfig, ModelConfig
from qac.data_loader import DatasetResult, load_dataset
from qac.evaluation import (
    ClassificationResult,
    ComparisonResult,
    compare_results,
    plot_comparison,
    plot_confusion_matrix,
)
from qac.preprocessing import apply_pca
from qac.vqc_classifier import train_vqc

logger = logging.getLogger(__name__)


class ClassifierExperiment:
    """Runs the full grid of classification experiments.

    Args:
        config: ExperimentConfig defining dataset, models, and output.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self._config = config
        self._output_dir = Path(config.output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> ComparisonResult:
        """Execute all experiments and return comparison.

        Pipeline:
            1. Load dataset from DataConfig
            2. Apply PCA preprocessing
            3. For each ModelConfig: train and evaluate
            4. Compare all results
            5. Save results to CSV/JSON

        Returns:
            ComparisonResult with all model results and comparison analysis.
        """
        total_start = time.time()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Step 1: Load dataset
        logger.info("=" * 60)
        logger.info("QAC Experiment Pipeline Starting")
        logger.info("=" * 60)
        logger.info("Step 1: Loading dataset '%s'...", self._config.data.dataset_name)

        raw_dataset = load_dataset(self._config.data)

        logger.info(
            "  Loaded %d samples, %d classes: %s",
            len(raw_dataset.y_train) + len(raw_dataset.y_val) + len(raw_dataset.y_test),
            raw_dataset.num_classes,
            raw_dataset.class_names,
        )

        # Step 2: Apply PCA
        logger.info(
            "Step 2: Applying PCA (%d components)...",
            self._config.data.n_pca_components,
        )
        dataset = apply_pca(
            raw_dataset,
            n_components=self._config.data.n_pca_components,
            seed=self._config.data.seed,
        )
        logger.info(
            "  PCA explained variance: %.2f%%",
            dataset.metadata.get("pca_explained_variance_ratio", 0) * 100,
        )

        # Step 3: Train all models
        logger.info("Step 3: Training %d models...", len(self._config.models))
        results: list[ClassificationResult] = []

        for i, model_config in enumerate(self._config.models):
            experiment_id = f"{timestamp}_{model_config.model_type}_{i}"
            logger.info(
                "  Model %d/%d: %s",
                i + 1,
                len(self._config.models),
                model_config.model_type,
            )

            result = self._train_model(dataset, model_config, experiment_id)
            results.append(result)

        # Step 4: Compare
        logger.info("Step 4: Comparing models...")
        comparison = compare_results(results)

        logger.info(
            "  Best model: %s (accuracy=%.4f)",
            comparison.best_model.model_type,
            comparison.best_model.accuracy,
        )
        logger.info(
            "  Quantum advantage: %s (Δ=%.4f)",
            comparison.quantum_advantage,
            comparison.accuracy_delta,
        )

        # Step 5: Save results
        logger.info("Step 5: Saving results...")
        self._save_results(results, comparison, timestamp, dataset)

        total_time = time.time() - total_start
        logger.info("=" * 60)
        logger.info("Pipeline completed in %.2fs", total_time)
        logger.info("=" * 60)

        return comparison

    def _train_model(
        self,
        dataset: DatasetResult,
        config: ModelConfig,
        experiment_id: str,
    ) -> ClassificationResult:
        """Train a single model based on its type."""
        models_dir = self._output_dir / "models"

        if config.model_type == "svm":
            return train_svm(
                dataset=dataset,
                config=config,
                models_dir=models_dir,
                experiment_id=experiment_id,
            )
        elif config.model_type == "vqc":
            return train_vqc(
                dataset=dataset,
                config=config,
                models_dir=models_dir,
                experiment_id=experiment_id,
            )
        else:
            raise ValueError(f"Unknown model type: {config.model_type}")

    def _save_results(
        self,
        results: list[ClassificationResult],
        comparison: ComparisonResult,
        timestamp: str,
        dataset: DatasetResult,
    ) -> None:
        """Save experiment results to CSV and JSON."""
        # CSV
        csv_path = self._output_dir / f"results_{timestamp}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "model_type", "accuracy", "f1_weighted", "f1_macro",
                    "precision", "recall", "training_time_s", "inference_time_s",
                    "model_hash",
                ],
            )
            writer.writeheader()
            for r in results:
                writer.writerow({
                    "model_type": r.model_type,
                    "accuracy": round(r.accuracy, 4),
                    "f1_weighted": round(r.f1_weighted, 4),
                    "f1_macro": round(r.f1_macro, 4),
                    "precision": round(r.precision, 4),
                    "recall": round(r.recall, 4),
                    "training_time_s": r.training_time_s,
                    "inference_time_s": r.inference_time_s,
                    "model_hash": r.model_hash,
                })

        # JSON (full details)
        json_path = self._output_dir / f"results_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(
                {
                    "timestamp": timestamp,
                    "config": self._config.to_dict(),
                    "dataset": dataset.to_dict(),
                    "comparison": comparison.to_dict(),
                    "results": [r.to_dict() for r in results],
                },
                f,
                indent=2,
            )

        # Plots
        try:
            plot_comparison(
                results,
                output_path=str(self._output_dir / f"comparison_{timestamp}.png"),
            )
            for r in results:
                plot_confusion_matrix(
                    r,
                    class_names=dataset.class_names,
                    output_path=str(
                        self._output_dir / f"cm_{r.model_type}_{timestamp}.png"
                    ),
                )
        except Exception as e:
            logger.warning("Could not generate plots: %s", e)

        logger.info("  Results saved to %s", self._output_dir)

"""QAC — Experiment CLI runner.

Loads experiment config from JSON and runs the full pipeline.
Mirrors scripts/run_experiments.py from vqe_knapsack_local.

Usage:
    python scripts/run_experiment.py --config configs/default.json
"""
from __future__ import annotations

import argparse
import logging
import sys

from qac.config import ExperimentConfig
from qac.experiment import ClassifierExperiment


def main() -> None:
    """Entry point for CLI experiment runner."""
    parser = argparse.ArgumentParser(
        description="QAC — Run classification experiments"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment JSON config file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load config
    config = ExperimentConfig.from_json(args.config)
    logger = logging.getLogger("qac")
    logger.info("Loaded config from %s", args.config)
    logger.info("Dataset: %s | Models: %d", config.data.dataset_name, len(config.models))

    # Run experiment
    experiment = ClassifierExperiment(config)
    comparison = experiment.run()

    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    for r in comparison.results:
        print(
            f"  {r.model_type.upper():>6}: accuracy={r.accuracy:.4f}  "
            f"f1={r.f1_weighted:.4f}  time={r.training_time_s:.1f}s"
        )
    print(f"\n  Best: {comparison.best_model.model_type.upper()} ({comparison.best_model.accuracy:.4f})")
    print(f"  Quantum advantage: {'YES' if comparison.quantum_advantage else 'NO'} (Δ={comparison.accuracy_delta:+.4f})")
    print("=" * 60)


if __name__ == "__main__":
    main()

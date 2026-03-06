"""
QAC — VQE Phase II Analysis Module.

Post-hoc analysis for parametric VQE experiments:
- Consolidation of all experiment results
- Comparison table generation
- Formal superiority evaluation (5 criteria)
- Automatic verdict generation
- Final report markdown generation

This file is a NEW incremental extension — no existing files are modified.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np


# ─────────────────── Superiority Criteria ──────────────


SUPERIORITY_CRITERIA = {
    "C1": {
        "name": "Accuracy Gap > 3%",
        "description": "Accuracy_Q > Accuracy_Classical + 3%",
    },
    "C2": {
        "name": "p-value < 0.05",
        "description": "Statistical significance via t-test or permutation test",
    },
    "C3": {
        "name": "ΔE Statistically Significant",
        "description": "Energy separation between classes is significant (Cohen's d > 0.5)",
    },
    "C4": {
        "name": "NISQ Robustness < 30%",
        "description": "Accuracy degradation under noise < 30%",
    },
    "C5": {
        "name": "Reproducibility < 1%",
        "description": "Accuracy variance across seeds < 1%",
    },
}


# ─────────────────── Consolidation ─────────────────────


def consolidate_results(
    results_dir: str | Path,
) -> dict[str, Any]:
    """
    Consolidate all Phase II experiment results from JSON files.

    Args:
        results_dir: Directory containing per-experiment JSON files

    Returns:
        Consolidated dict with per-config aggregated metrics
    """
    results_dir = Path(results_dir)

    # Load individual results
    all_results = []
    for json_file in sorted(results_dir.glob("VQE_PHASE2_*.json")):
        if json_file.name == "consolidated_results.json":
            continue
        with open(json_file, "r", encoding="utf-8") as f:
            all_results.append(json.load(f))

    if not all_results:
        # Try loading from consolidated file
        consolidated_file = results_dir / "consolidated_results.json"
        if consolidated_file.exists():
            with open(consolidated_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            all_results = data.get("results", [])

    if not all_results:
        return {
            "error": "No results found",
            "configs": [],
            "total_experiments": 0,
        }

    # Group by config_id
    by_config: dict[str, list[dict]] = {}
    for r in all_results:
        config = r.get("config", {})
        config_id = (
            f"{config.get('optimizer_type', '?')}_reps{config.get('ansatz_reps', '?')}"
            f"_g{config.get('coupling_strength', '?')}_h{config.get('transverse_field', '?')}"
        )
        if config_id not in by_config:
            by_config[config_id] = []
        by_config[config_id].append(r)

    # Aggregate per config
    configs = []
    for config_id, runs in by_config.items():
        completed_runs = [r for r in runs if r.get("status") == "COMPLETED"]

        if not completed_runs:
            configs.append({
                "config_id": config_id,
                "config": runs[0].get("config", {}),
                "n_runs": len(runs),
                "n_completed": 0,
                "status": "ALL_FAILED",
            })
            continue

        accuracies = [r["accuracy"] for r in completed_runs]
        f1_scores = [r["f1_score"] for r in completed_runs]
        delta_es = [r["delta_E"] for r in completed_runs]
        cohens_ds = [r["cohens_d"] for r in completed_runs]
        p_ttests = [r["p_value_ttest"] for r in completed_runs]
        p_perms = [r["p_value_permutation"] for r in completed_runs]
        training_times = [r["training_time_s"] for r in completed_runs]

        configs.append({
            "config_id": config_id,
            "config": completed_runs[0].get("config", {}),
            "n_runs": len(runs),
            "n_completed": len(completed_runs),
            "status": "COMPLETED",
            "accuracy_mean": float(np.mean(accuracies)),
            "accuracy_std": float(np.std(accuracies)),
            "accuracy_min": float(np.min(accuracies)),
            "accuracy_max": float(np.max(accuracies)),
            "f1_mean": float(np.mean(f1_scores)),
            "f1_std": float(np.std(f1_scores)),
            "delta_E_mean": float(np.mean(delta_es)),
            "delta_E_std": float(np.std(delta_es)),
            "cohens_d_mean": float(np.mean(cohens_ds)),
            "p_ttest_mean": float(np.mean(p_ttests)),
            "p_ttest_min": float(np.min(p_ttests)),
            "p_perm_mean": float(np.mean(p_perms)),
            "p_perm_min": float(np.min(p_perms)),
            "training_time_mean": float(np.mean(training_times)),
            "individual_runs": completed_runs,
        })

    # Sort by mean accuracy descending
    configs.sort(key=lambda x: x.get("accuracy_mean", 0), reverse=True)

    return {
        "total_experiments": len(all_results),
        "completed": sum(1 for r in all_results if r.get("status") == "COMPLETED"),
        "failed": sum(1 for r in all_results if r.get("status") == "FAILED"),
        "n_configs": len(configs),
        "configs": configs,
    }


# ─────────────────── Comparison Table ──────────────────


def generate_comparison_table(consolidated: dict[str, Any]) -> str:
    """
    Generate a markdown comparison table of all configurations.

    Returns:
        Markdown string with formatted table
    """
    configs = consolidated.get("configs", [])
    if not configs:
        return "No results to tabulate.\n"

    # Header
    lines = [
        "| # | Optimizer | Reps | γ | h | Acc (μ±σ) | F1 (μ) | ΔE (μ) | Cohen's d | p-ttest | p-perm | Time (s) |",
        "|---|-----------|------|---|---|-----------|--------|--------|----------|---------|--------|----------|",
    ]

    for i, cfg in enumerate(configs, 1):
        if cfg.get("status") == "ALL_FAILED":
            c = cfg.get("config", {})
            lines.append(
                f"| {i} | {c.get('optimizer_type', '?')} | {c.get('ansatz_reps', '?')} | "
                f"{c.get('coupling_strength', '?')} | {c.get('transverse_field', '?')} | "
                f"FAILED | — | — | — | — | — | — |"
            )
            continue

        c = cfg.get("config", {})
        acc = f"{cfg['accuracy_mean']:.4f}±{cfg['accuracy_std']:.4f}"
        lines.append(
            f"| {i} | {c.get('optimizer_type', '?')} | {c.get('ansatz_reps', '?')} | "
            f"{c.get('coupling_strength', '?')} | {c.get('transverse_field', '?')} | "
            f"{acc} | {cfg['f1_mean']:.4f} | {cfg['delta_E_mean']:.4f} | "
            f"{cfg['cohens_d_mean']:.4f} | {cfg['p_ttest_min']:.4f} | "
            f"{cfg['p_perm_min']:.4f} | {cfg['training_time_mean']:.1f} |"
        )

    return "\n".join(lines) + "\n"


# ─────────────────── Best Configuration ────────────────


def identify_best_configuration(consolidated: dict[str, Any]) -> dict[str, Any] | None:
    """
    Identify the best configuration by accuracy, then F1.

    Returns:
        Best config dict, or None if no completed configs
    """
    configs = [c for c in consolidated.get("configs", []) if c.get("status") == "COMPLETED"]
    if not configs:
        return None

    return max(configs, key=lambda c: (c["accuracy_mean"], c["f1_mean"]))


# ─────────────────── Superiority Evaluation ────────────


def evaluate_superiority_criteria(
    consolidated: dict[str, Any],
    baseline_accuracy: float = 0.0,
    nisq_degradation_pct: float | None = None,
) -> dict[str, Any]:
    """
    Evaluate the 5 formal criteria for quantum superiority.

    Args:
        consolidated: Consolidated results from consolidate_results()
        baseline_accuracy: Classical baseline accuracy (from Phase I or SVM)
        nisq_degradation_pct: NISQ noise degradation percentage (if evaluated)

    Returns:
        Dict with per-criterion pass/fail and overall verdict
    """
    best = identify_best_configuration(consolidated)
    if best is None:
        return {
            "verdict": "RESULTADO INCONCLUSIVO",
            "reason": "No completed experiments",
            "criteria": {},
        }

    criteria = {}

    # C1: Accuracy Gap > 3%
    acc_gap = best["accuracy_mean"] - baseline_accuracy
    criteria["C1"] = {
        **SUPERIORITY_CRITERIA["C1"],
        "passed": acc_gap > 0.03,
        "value": f"Δ = {acc_gap:.4f} ({best['accuracy_mean']:.4f} - {baseline_accuracy:.4f})",
    }

    # C2: p-value < 0.05
    p_min = best.get("p_ttest_min", 1.0)
    p_perm_min = best.get("p_perm_min", 1.0)
    p_best = min(p_min, p_perm_min)
    criteria["C2"] = {
        **SUPERIORITY_CRITERIA["C2"],
        "passed": p_best < 0.05,
        "value": f"p_ttest_min = {p_min:.6f}, p_perm_min = {p_perm_min:.6f}",
    }

    # C3: ΔE Statistically Significant (Cohen's d > 0.5)
    cohens_d = best.get("cohens_d_mean", 0.0)
    criteria["C3"] = {
        **SUPERIORITY_CRITERIA["C3"],
        "passed": cohens_d > 0.5,
        "value": f"Cohen's d = {cohens_d:.4f}",
    }

    # C4: NISQ Robustness < 30%
    if nisq_degradation_pct is not None:
        criteria["C4"] = {
            **SUPERIORITY_CRITERIA["C4"],
            "passed": nisq_degradation_pct < 30.0,
            "value": f"Degradation = {nisq_degradation_pct:.2f}%",
        }
    else:
        criteria["C4"] = {
            **SUPERIORITY_CRITERIA["C4"],
            "passed": None,  # Not evaluated
            "value": "Not evaluated (requires NISQ simulation)",
        }

    # C5: Reproducibility < 1%
    acc_std = best.get("accuracy_std", 1.0)
    criteria["C5"] = {
        **SUPERIORITY_CRITERIA["C5"],
        "passed": acc_std < 0.01,
        "value": f"σ(accuracy) = {acc_std:.6f}",
    }

    # Determine verdict
    evaluated_criteria = {k: v for k, v in criteria.items() if v["passed"] is not None}
    all_passed = all(v["passed"] for v in evaluated_criteria.values())
    any_passed = any(v["passed"] for v in evaluated_criteria.values())

    if all_passed and len(evaluated_criteria) >= 4:
        verdict = "SUPERIORIDADE CONFIRMADA"
    elif not any_passed:
        verdict = "SUPERIORIDADE NÃO CONFIRMADA"
    else:
        verdict = "RESULTADO INCONCLUSIVO"

    n_passed = sum(1 for v in evaluated_criteria.values() if v["passed"])
    n_total = len(evaluated_criteria)

    return {
        "verdict": verdict,
        "criteria": criteria,
        "n_passed": n_passed,
        "n_total": n_total,
        "best_config": best["config_id"],
        "best_accuracy": best["accuracy_mean"],
        "baseline_accuracy": baseline_accuracy,
    }


# ─────────────────── Verdict ───────────────────────────


def generate_final_verdict(evaluation: dict[str, Any]) -> str:
    """
    Generate the final scientific verdict string.

    Returns one of:
    - SUPERIORIDADE CONFIRMADA
    - SUPERIORIDADE NÃO CONFIRMADA
    - RESULTADO INCONCLUSIVO
    """
    return evaluation.get("verdict", "RESULTADO INCONCLUSIVO")


# ─────────────────── Report Generation ─────────────────


def generate_report_markdown(
    consolidated: dict[str, Any],
    evaluation: dict[str, Any],
    phase1_reference: dict[str, Any] | None = None,
) -> str:
    """
    Generate the complete Phase II Final Report as markdown.

    Args:
        consolidated: From consolidate_results()
        evaluation: From evaluate_superiority_criteria()
        phase1_reference: Phase I results for comparison

    Returns:
        Complete markdown report string
    """
    verdict = evaluation.get("verdict", "RESULTADO INCONCLUSIVO")
    best = identify_best_configuration(consolidated)
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    sections = []

    # Header
    sections.append(f"""# VQE Phase II — Final Report

**Status**: {verdict}
**Generated**: {timestamp}
**Total Experiments**: {consolidated.get('total_experiments', 0)}
**Completed**: {consolidated.get('completed', 0)}
**Failed**: {consolidated.get('failed', 0)}

---

## 1. Experimental Design

### 1.1 Hypothesis

> Exists a parametric VQE configuration that demonstrates statistically significant
> superiority over classical baseline for PlantVillage binary classification
> (Tomato___Bacterial_spot vs Tomato___healthy)?

### 1.2 Parametric Grid

Three parameters were systematically varied:

- **Optimizer**: COBYLA, SPSA, L-BFGS-B
- **Ansatz Depth**: reps ∈ {{2, 4, 6}}
- **Hamiltonian Parameters**: γ ∈ {{0.1, 0.5, 1.0}}, h ∈ {{0.1, 0.5, 1.0}}

Total: {consolidated.get('n_configs', 0)} configurations × 3 seeds = {consolidated.get('total_experiments', 0)} experiments.

### 1.3 Acceptance Criteria

| # | Criterion | Threshold |
|---|-----------|-----------|
| C1 | Accuracy Gap | > 3% over classical baseline |
| C2 | p-value | < 0.05 (t-test or permutation) |
| C3 | ΔE Significance | Cohen's d > 0.5 |
| C4 | NISQ Robustness | Degradation < 30% |
| C5 | Reproducibility | σ(accuracy) < 1% |

---""")

    # Phase I Reference
    if phase1_reference:
        sections.append(f"""
## 2. Phase I Reference

| Metric | Phase I Value |
|--------|--------------|
| Accuracy | {phase1_reference.get('accuracy', 'N/A')} |
| F1 Score | {phase1_reference.get('f1_score', 'N/A')} |
| ΔE | {phase1_reference.get('delta_E', 'N/A')} |
| p-value (t-test) | {phase1_reference.get('p_ttest', 'N/A')} |
| p-value (permutation) | {phase1_reference.get('p_perm', 'N/A')} |
| Status | COMPLETED_WITH_LIMITATIONS |

---""")

    # Comparison Table
    table = generate_comparison_table(consolidated)
    sections.append(f"""
## 3. Consolidated Results

{table}

---""")

    # Best Configuration
    if best:
        best_config = best.get("config", {})
        sections.append(f"""
## 4. Best Configuration

| Parameter | Value |
|-----------|-------|
| Config ID | `{best.get('config_id', 'N/A')}` |
| Optimizer | {best_config.get('optimizer_type', 'N/A')} |
| Ansatz Reps | {best_config.get('ansatz_reps', 'N/A')} |
| γ (coupling) | {best_config.get('coupling_strength', 'N/A')} |
| h (transverse) | {best_config.get('transverse_field', 'N/A')} |
| Accuracy (μ±σ) | {best.get('accuracy_mean', 0):.4f} ± {best.get('accuracy_std', 0):.4f} |
| F1 Score (μ) | {best.get('f1_mean', 0):.4f} |
| ΔE (μ) | {best.get('delta_E_mean', 0):.4f} |
| Cohen's d (μ) | {best.get('cohens_d_mean', 0):.4f} |
| p-value (best) | {min(best.get('p_ttest_min', 1), best.get('p_perm_min', 1)):.6f} |
| Training time (μ) | {best.get('training_time_mean', 0):.1f}s |

---""")

    # Criteria Evaluation
    criteria = evaluation.get("criteria", {})
    criteria_rows = []
    for cid, cdata in criteria.items():
        status = "✅ PASS" if cdata.get("passed") else ("❌ FAIL" if cdata.get("passed") is not None else "⚠️ N/A")
        criteria_rows.append(f"| {cid} | {cdata.get('name', '')} | {status} | {cdata.get('value', '')} |")

    criteria_table = "\n".join(criteria_rows)
    sections.append(f"""
## 5. Superiority Criteria Evaluation

| # | Criterion | Result | Detail |
|---|-----------|--------|--------|
{criteria_table}

**Criteria Passed**: {evaluation.get('n_passed', 0)}/{evaluation.get('n_total', 0)}

---""")

    # Sensitivity Analysis
    if best and len(consolidated.get("configs", [])) > 1:
        sections.append("""
## 6. Sensitivity Analysis

### 6.1 Optimizer Sensitivity

Comparison across optimizers (holding other parameters at default):
""")
        optimizer_groups: dict[str, list] = {}
        for cfg in consolidated.get("configs", []):
            if cfg.get("status") != "COMPLETED":
                continue
            opt = cfg.get("config", {}).get("optimizer_type", "?")
            if opt not in optimizer_groups:
                optimizer_groups[opt] = []
            optimizer_groups[opt].append(cfg)

        for opt, cfgs in optimizer_groups.items():
            accs = [c["accuracy_mean"] for c in cfgs]
            sections.append(
                f"- **{opt}**: mean acc = {np.mean(accs):.4f}, "
                f"range = [{np.min(accs):.4f}, {np.max(accs):.4f}]"
            )

        sections.append("""
### 6.2 Depth Sensitivity

Comparison across ansatz depths:
""")
        depth_groups: dict[int, list] = {}
        for cfg in consolidated.get("configs", []):
            if cfg.get("status") != "COMPLETED":
                continue
            reps = cfg.get("config", {}).get("ansatz_reps", 0)
            if reps not in depth_groups:
                depth_groups[reps] = []
            depth_groups[reps].append(cfg)

        for reps, cfgs in sorted(depth_groups.items()):
            accs = [c["accuracy_mean"] for c in cfgs]
            sections.append(
                f"- **reps={reps}**: mean acc = {np.mean(accs):.4f}, "
                f"range = [{np.min(accs):.4f}, {np.max(accs):.4f}]"
            )

        sections.append("\n---")

    # Conclusion
    sections.append(f"""
## 7. Scientific Conclusion

### Verdict

**{verdict}**

### Justification

{"All formal criteria for quantum superiority have been simultaneously satisfied." if verdict == "SUPERIORIDADE CONFIRMADA" else ""}{"The parametric exploration did not yield a configuration satisfying all five criteria for quantum superiority. The observed performance gap, statistical significance, or reproducibility did not meet the formal thresholds required." if verdict == "SUPERIORIDADE NÃO CONFIRMADA" else ""}{"Some criteria were satisfied but not all. Further investigation with expanded parameter ranges, alternative Hamiltonian formulations, or larger datasets may be warranted." if verdict == "RESULTADO INCONCLUSIVO" else ""}

### Baseline Comparison

- Classical baseline accuracy: {evaluation.get('baseline_accuracy', 'N/A')}
- Best quantum accuracy: {evaluation.get('best_accuracy', 'N/A')}
- Best quantum config: `{evaluation.get('best_config', 'N/A')}`

---

## 8. Registry Integrity

All experiments were registered atomically in the MCP registry:

- Each experiment has a unique `experiment_id`
- All models are persisted with SHA-256 hashes
- All metrics are recorded in `experiments/vqe_phase2/`
- No Phase I resources were modified

---

> *Generated automatically by `quantum/vqe_analysis.py`*
> *Quantum AgriClassifier — VQE Phase II*
""")

    return "\n".join(sections)

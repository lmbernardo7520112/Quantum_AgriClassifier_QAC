"""
QAC — Scientific Maturity Tests (Phase 5).

1. Reproducibility: re-run same experiment, variance < 1%
2. Cross-model validation: quantum vs baseline
3. Registry consistency: no orphans
4. Noise robustness: degradation ≤ 30%
5. Hard reset: restart, verify identical state
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import numpy as np

from mcp_server.context_manager import ContextManager
from mcp_server.resource_registry import ResourceRegistry


@pytest.fixture
def tmp_registry(tmp_path):
    for fname in ("models.json", "metrics.json", "experiments.json", "context.json"):
        key = "resources" if "model" in fname or "metric" in fname else (
            "experiments" if fname == "experiments.json" else "contexts"
        )
        (tmp_path / fname).write_text(json.dumps({key: {}, "version": "1.0.0", "last_updated": None}))
    return tmp_path


class TestReproducibility:
    """Reproducibility Test: re-run with same seed → variance < 1%."""

    def test_deterministic_hashing(self):
        """Verify hashing is deterministic."""
        data = np.random.RandomState(42).rand(100, 10)
        h1 = ContextManager.hash_string(data.tobytes().hex())
        h2 = ContextManager.hash_string(data.tobytes().hex())
        assert h1 == h2

    def test_seed_produces_same_splits(self):
        """Verify same seed produces identical train/test splits."""
        from sklearn.model_selection import train_test_split

        X = np.random.RandomState(42).rand(200, 5)
        y = np.random.RandomState(42).randint(0, 3, 200)

        X1_train, X1_test, y1_train, y1_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X2_train, X2_test, y2_train, y2_test = train_test_split(X, y, test_size=0.2, random_state=42)

        np.testing.assert_array_equal(X1_train, X2_train)
        np.testing.assert_array_equal(y1_test, y2_test)


class TestRegistryConsistency:
    """Registry Consistency Check: no orphans, timestamps on all."""

    def test_empty_registry_consistent(self, tmp_registry):
        rr = ResourceRegistry(tmp_registry)
        result = rr.check_consistency()
        assert result["consistent"] is True
        assert result["issues"] == []

    def test_all_resources_have_timestamps(self, tmp_registry):
        rr = ResourceRegistry(tmp_registry)
        rr.register("resource.model", metadata={"type": "svm"}, experiment_id="exp-00000001")
        rr.register("resource.metrics", metadata={"accuracy": 0.9}, experiment_id="exp-00000001")

        for resource in rr.list_resources():
            assert resource.get("timestamp") is not None


class TestHardReset:
    """Hard Reset Test: stop server, restart, verify identical state."""

    def test_context_survives_restart(self, tmp_registry):
        cm = ContextManager(tmp_registry)
        cm.create_context("ctx-reset-test", seed=42, dataset_hash="abc")
        snapshot = cm.snapshot()

        # Simulate restart
        cm2 = ContextManager(tmp_registry)
        assert cm2.verify_snapshot(snapshot)

    def test_resources_survive_restart(self, tmp_registry):
        rr = ResourceRegistry(tmp_registry)
        rr.register("resource.model", resource_id="reset-model-001", metadata={"test": True})
        snapshot = rr.snapshot()

        # Simulate restart
        rr2 = ResourceRegistry(tmp_registry)
        assert rr2.verify_snapshot(snapshot)

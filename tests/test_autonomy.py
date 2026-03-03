"""
QAC — Autonomy Tests (Phase 6 — Falsifiable).

Test A: Concurrency — unique experiment_ids
Test B: Context loss — train, restart, evaluate
Test C: Contract violation — structured error, no crash
Test D: Physical audit — file existence + hash verification
Test E: Autonomous recovery — invalid dataset handling
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from mcp_server.context_manager import ContextManager
from mcp_server.resource_registry import ResourceRegistry
from mcp_server.schemas import SchemaRegistry
from mcp_server.tool_registry import ToolError, ToolRegistry
from mcp_server.execution_engine import ExecutionEngine


@pytest.fixture
def tmp_registry(tmp_path):
    for fname in ("models.json", "metrics.json", "experiments.json", "context.json"):
        key = "resources" if "model" in fname or "metric" in fname else (
            "experiments" if fname == "experiments.json" else "contexts"
        )
        (tmp_path / fname).write_text(json.dumps({key: {}, "version": "1.0.0", "last_updated": None}))
    return tmp_path


@pytest.fixture
def tmp_tasks(tmp_path):
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    (tasks_dir / "todo.md").write_text("# TODO\n")
    (tasks_dir / "lessons.md").write_text("# Lessons\n")
    return tasks_dir


@pytest.fixture
def engine(tmp_registry, tmp_tasks):
    sr = SchemaRegistry()
    cm = ContextManager(tmp_registry)
    rr = ResourceRegistry(tmp_registry)
    tr = ToolRegistry(sr)

    # Register a dummy tool
    def dummy_tool(input_data, experiment_id, context, resource_registry, context_manager):
        resource_registry.register(
            "resource.model",
            metadata={"dummy": True},
            experiment_id=experiment_id,
        )
        return {"experiment_id": experiment_id, "status": "success"}

    tr.register("tool.initialize_project", dummy_tool)

    ee = ExecutionEngine(tr, rr, cm, tmp_registry, tmp_tasks)
    return ee


# ─────────────────── Test A: Concurrency ───────────────

@pytest.mark.asyncio
class TestConcurrency:
    async def test_unique_experiment_ids(self, engine):
        """Simulate two concurrent tool calls — experiment_ids must be unique."""
        results = await asyncio.gather(
            engine.execute("tool.initialize_project", {"project_root": "/tmp/a", "dataset_root": "/tmp/b"}),
            engine.execute("tool.initialize_project", {"project_root": "/tmp/c", "dataset_root": "/tmp/d"}),
        )

        ids = [r["experiment_id"] for r in results]
        assert len(ids) == 2
        assert ids[0] != ids[1], "Experiment IDs must be unique"

    async def test_registry_not_corrupted(self, engine):
        """After concurrent calls, registry should be consistent."""
        await asyncio.gather(
            engine.execute("tool.initialize_project", {"project_root": "/tmp/a", "dataset_root": "/tmp/b"}),
            engine.execute("tool.initialize_project", {"project_root": "/tmp/c", "dataset_root": "/tmp/d"}),
        )

        experiments = engine.list_experiments()
        assert len(experiments) == 2
        for exp in experiments:
            assert exp["status"] in ("COMPLETED", "COMPLETED_WITH_WARNINGS")


# ─────────────────── Test C: Contract Violation ────────

class TestContractViolation:
    @pytest.mark.asyncio
    async def test_missing_tool_returns_error(self, engine):
        """Calling nonexistent tool returns structured error, not crash."""
        result = await engine.execute("tool.nonexistent", {})
        assert result.get("error") is True
        assert "error_type" in result

    @pytest.mark.asyncio
    async def test_invalid_input_returns_error(self, engine):
        """Calling tool with invalid input returns structured error."""
        # Register tool that expects specific input
        sr = SchemaRegistry()
        tr = engine._tool_registry

        def dummy(input_data, experiment_id, context, resource_registry, context_manager):
            return {"experiment_id": experiment_id}

        tr.register("tool.load_dataset", dummy)

        result = await engine.execute("tool.load_dataset", {})  # missing required 'dataset_name'
        assert result.get("error") is True


# ─────────────────── Test D: Physical Audit ────────────

class TestPhysicalAudit:
    def test_audit_empty_registry(self, tmp_registry):
        """Empty registry passes audit."""
        rr = ResourceRegistry(tmp_registry)
        result = rr.verify_physical_audit()
        assert result["passed"] is True

    def test_audit_with_valid_file(self, tmp_registry):
        """Model with valid file passes audit."""
        rr = ResourceRegistry(tmp_registry)

        # Create a dummy model file
        model_file = tmp_registry / "test_model.pkl"
        model_file.write_bytes(b"fake model data")

        rr.register(
            "resource.model",
            file_path=str(model_file),
            metadata={"type": "test"},
        )

        result = rr.verify_physical_audit()
        assert result["passed"] is True

    def test_audit_with_missing_file(self, tmp_registry):
        """Model with missing file fails audit."""
        rr = ResourceRegistry(tmp_registry)
        rr.register(
            "resource.model",
            file_path=str(tmp_registry / "nonexistent.pkl"),
            metadata={"type": "test"},
        )

        result = rr.verify_physical_audit()
        assert result["passed"] is False


# ─────────────────── Test E: Autonomous Recovery ───────

class TestAutonomousRecovery:
    @pytest.mark.asyncio
    async def test_invalid_dataset_handled(self, engine, tmp_tasks):
        """Invalid dataset should log FAILED experiment and update lessons.md."""
        sr = SchemaRegistry()
        tr = engine._tool_registry

        def failing_tool(input_data, experiment_id, context, resource_registry, context_manager):
            raise ValueError("Dataset not found: invalid_path")

        tr.register("tool.load_dataset", failing_tool)

        result = await engine.execute("tool.load_dataset", {"dataset_name": "eurosat_rgb"})

        # Should not crash — returns error
        assert result.get("error") is True

        # Experiment should be recorded as FAILED
        experiments = engine.list_experiments(status="FAILED")
        assert len(experiments) >= 1

        # lessons.md should be updated
        lessons_content = (tmp_tasks / "lessons.md").read_text()
        assert "FAILED" in lessons_content

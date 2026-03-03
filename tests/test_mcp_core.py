"""
QAC — Unit tests for MCP core infrastructure.

Tests schemas, context manager, resource registry, tool registry,
and execution engine independently.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from mcp_server.schemas import SchemaRegistry
from mcp_server.context_manager import ContextManager
from mcp_server.resource_registry import ResourceRegistry
from mcp_server.tool_registry import ToolError, ToolRegistry


@pytest.fixture
def tmp_registry(tmp_path):
    """Create temp registry directory with initial JSON files."""
    for fname in ("models.json", "metrics.json", "experiments.json", "context.json"):
        key = "resources" if fname != "experiments.json" and fname != "context.json" else (
            "experiments" if fname == "experiments.json" else "contexts"
        )
        (tmp_path / fname).write_text(json.dumps({key: {}, "version": "1.0.0", "last_updated": None}))
    return tmp_path


@pytest.fixture
def tmp_tasks(tmp_path):
    """Create temp tasks directory."""
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    (tasks_dir / "todo.md").write_text("# TODO\n")
    (tasks_dir / "lessons.md").write_text("# Lessons\n")
    return tasks_dir


# ─────────────────── SchemaRegistry Tests ──────────────

class TestSchemaRegistry:
    def test_list_tools(self):
        sr = SchemaRegistry()
        tools = sr.list_tools()
        assert len(tools) >= 10
        names = [t["name"] for t in tools]
        assert "tool.initialize_project" in names
        assert "tool.load_dataset" in names
        assert "tool.train_qsvm" in names

    def test_validate_input_valid(self):
        sr = SchemaRegistry()
        errors = sr.validate_input("tool.initialize_project", {
            "project_root": "/tmp/test",
            "dataset_root": "/tmp/data",
        })
        assert errors == []

    def test_validate_input_invalid(self):
        sr = SchemaRegistry()
        errors = sr.validate_input("tool.load_dataset", {})  # missing required field
        assert len(errors) > 0

    def test_validate_unknown_tool(self):
        sr = SchemaRegistry()
        errors = sr.validate_input("tool.nonexistent", {})
        assert "Unknown tool" in errors[0]

    def test_get_preconditions(self):
        sr = SchemaRegistry()
        pre = sr.get_preconditions("tool.load_dataset")
        assert "project_initialized" in pre

    def test_get_postconditions(self):
        sr = SchemaRegistry()
        post = sr.get_postconditions("tool.run_baseline")
        assert "resource.model_registered" in post


# ─────────────────── ContextManager Tests ──────────────

class TestContextManager:
    def test_create_and_get(self, tmp_registry):
        cm = ContextManager(tmp_registry)
        ctx = cm.create_context("ctx-001", seed=42, dataset_hash="abc123")
        assert ctx["context_id"] == "ctx-001"
        assert ctx["seed"] == 42

        retrieved = cm.get_context("ctx-001")
        assert retrieved["seed"] == 42

    def test_persistence(self, tmp_registry):
        cm = ContextManager(tmp_registry)
        cm.create_context("ctx-persist", seed=99)

        # Reload
        cm2 = ContextManager(tmp_registry)
        ctx = cm2.get_context("ctx-persist")
        assert ctx is not None
        assert ctx["seed"] == 99

    def test_snapshot_verify(self, tmp_registry):
        cm = ContextManager(tmp_registry)
        cm.create_context("ctx-snap", seed=42)
        snapshot = cm.snapshot()

        assert cm.verify_snapshot(snapshot)

    def test_hash_string(self):
        h1 = ContextManager.hash_string("hello")
        h2 = ContextManager.hash_string("hello")
        h3 = ContextManager.hash_string("world")
        assert h1 == h2
        assert h1 != h3


# ─────────────────── ResourceRegistry Tests ────────────

class TestResourceRegistry:
    def test_register_and_get(self, tmp_registry):
        rr = ResourceRegistry(tmp_registry)
        res = rr.register("resource.model", metadata={"type": "svm"})
        assert res["resource_type"] == "resource.model"
        assert "resource_id" in res

        retrieved = rr.get("resource.model", res["resource_id"])
        assert retrieved is not None

    def test_list_resources(self, tmp_registry):
        rr = ResourceRegistry(tmp_registry)
        rr.register("resource.model", metadata={"a": 1})
        rr.register("resource.model", metadata={"a": 2})
        models = rr.list_resources("resource.model")
        assert len(models) >= 2

    def test_persistence(self, tmp_registry):
        rr = ResourceRegistry(tmp_registry)
        res = rr.register("resource.model", resource_id="model-test-001", metadata={"x": 1})

        rr2 = ResourceRegistry(tmp_registry)
        retrieved = rr2.get("resource.model", "model-test-001")
        assert retrieved is not None

    def test_invalid_type(self, tmp_registry):
        rr = ResourceRegistry(tmp_registry)
        with pytest.raises(ValueError):
            rr.register("invalid.type")

    def test_consistency_check(self, tmp_registry):
        rr = ResourceRegistry(tmp_registry)
        result = rr.check_consistency()
        assert result["consistent"] is True

    def test_snapshot(self, tmp_registry):
        rr = ResourceRegistry(tmp_registry)
        rr.register("resource.model", metadata={"test": True})
        snap = rr.snapshot()
        assert rr.verify_snapshot(snap)


# ─────────────────── ToolRegistry Tests ────────────────

class TestToolRegistry:
    def test_register_and_get(self):
        sr = SchemaRegistry()
        tr = ToolRegistry(sr)

        def dummy_tool(**kwargs):
            return {"result": "ok"}

        tr.register("tool.initialize_project", dummy_tool)
        assert tr.is_registered("tool.initialize_project")

    def test_contract_violation(self):
        """Test C — structured error on contract violation."""
        sr = SchemaRegistry()
        tr = ToolRegistry(sr)

        with pytest.raises(ToolError) as exc_info:
            tr.validate_and_check("tool.nonexistent", {})

        error = exc_info.value.to_dict()
        assert error["error"] is True
        assert error["error_type"] == "TOOL_NOT_FOUND"

    def test_schema_validation_error(self):
        sr = SchemaRegistry()
        tr = ToolRegistry(sr)

        def dummy(**kwargs):
            return {}

        tr.register("tool.load_dataset", dummy)

        with pytest.raises(ToolError) as exc_info:
            tr.validate_and_check("tool.load_dataset", {})  # missing required

        assert exc_info.value.error_type == "SCHEMA_VALIDATION_ERROR"

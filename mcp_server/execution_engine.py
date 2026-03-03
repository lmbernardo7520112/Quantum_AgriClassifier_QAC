"""
QAC — Execution Engine.

Orchestrates tool execution with full lifecycle management:
PENDING → RUNNING → COMPLETED | FAILED

Supports:
- Invariant 1 (Persistence): every execution produces ≥1 resource
- Invariant 2 (Determinism): records seed, hash, backend, model_version
- Test A (Concurrency): UUID-based experiment_id with collision detection
- Test E (Recovery): failed experiments logged, lessons.md updated
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

from mcp_server.context_manager import ContextManager
from mcp_server.resource_registry import ResourceRegistry
from mcp_server.tool_registry import ToolError, ToolRegistry


class ExecutionEngine:
    """Orchestrates tool execution with experiment lifecycle management."""

    def __init__(
        self,
        tool_registry: ToolRegistry,
        resource_registry: ResourceRegistry,
        context_manager: ContextManager,
        registry_path: str | Path,
        tasks_path: str | Path,
    ) -> None:
        self._tool_registry = tool_registry
        self._resource_registry = resource_registry
        self._context_manager = context_manager
        self._registry_path = Path(registry_path)
        self._tasks_path = Path(tasks_path)
        self._experiments_file = self._registry_path / "experiments.json"
        self._experiments: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self._load_experiments()

    # ─────────────────── Persistence ───────────────────

    def _load_experiments(self) -> None:
        """Load experiment ledger from disk (Invariant 5)."""
        if self._experiments_file.exists():
            with open(self._experiments_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._experiments = data.get("experiments", {})

    def _save_experiments(self) -> None:
        """Persist experiment ledger to disk atomically."""
        tmp_file = self._experiments_file.with_suffix(".tmp")
        data = {
            "experiments": self._experiments,
            "version": "1.0.0",
            "last_updated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        if os.name == "nt" and self._experiments_file.exists():
            self._experiments_file.unlink()
        tmp_file.rename(self._experiments_file)

    # ─────────────────── Experiment ID ─────────────────

    def _generate_experiment_id(self) -> str:
        """Generate unique experiment ID with collision detection (Test A)."""
        for _ in range(100):  # safety limit
            exp_id = f"exp-{uuid.uuid4().hex[:8]}"
            if exp_id not in self._experiments:
                return exp_id
        raise RuntimeError("Failed to generate unique experiment_id after 100 attempts")

    # ─────────────────── Core Execution ────────────────

    async def execute(
        self,
        tool_name: str,
        input_data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Execute a tool with full lifecycle management.

        Returns structured result with experiment_id, or structured error.
        Never crashes (Test C — Contract Violation).
        """
        async with self._lock:
            experiment_id = self._generate_experiment_id()

        # Record experiment as PENDING
        experiment = {
            "experiment_id": experiment_id,
            "tool_name": tool_name,
            "status": "PENDING",
            "input": input_data,
            "output": None,
            "error": None,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "completed_at": None,
            "context": {},
            "resources_created": [],
        }

        async with self._lock:
            self._experiments[experiment_id] = experiment
            self._save_experiments()

        try:
            # Validate schema and preconditions (Test C)
            self._tool_registry.validate_and_check(tool_name, input_data)

            # Create execution context (Invariant 2)
            context = self._context_manager.create_context(
                context_id=experiment_id,
                seed=input_data.get("seed", 42),
                dataset_hash=input_data.get("dataset_hash", ""),
                backend=input_data.get("backend", "aer_statevector"),
            )
            experiment["context"] = context
            experiment["status"] = "RUNNING"

            async with self._lock:
                self._experiments[experiment_id] = experiment
                self._save_experiments()

            # Execute the tool
            tool_fn = self._tool_registry.get(tool_name)
            if tool_fn is None:
                raise ToolError(tool_name, "TOOL_NOT_FOUND", f"Tool '{tool_name}' not found")

            # Support both sync and async tool functions
            if asyncio.iscoroutinefunction(tool_fn):
                result = await tool_fn(
                    input_data=input_data,
                    experiment_id=experiment_id,
                    context=context,
                    resource_registry=self._resource_registry,
                    context_manager=self._context_manager,
                )
            else:
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: tool_fn(
                        input_data=input_data,
                        experiment_id=experiment_id,
                        context=context,
                        resource_registry=self._resource_registry,
                        context_manager=self._context_manager,
                    ),
                )

            # Inject experiment_id into result
            result["experiment_id"] = experiment_id

            # Validate output
            output_errors = self._tool_registry.validate_output(tool_name, result)
            if output_errors:
                experiment["status"] = "COMPLETED_WITH_WARNINGS"
                experiment["output"] = result
                experiment["output"]["_output_validation_warnings"] = output_errors
            else:
                experiment["status"] = "COMPLETED"
                experiment["output"] = result

            experiment["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")

        except ToolError as e:
            experiment["status"] = "FAILED"
            experiment["error"] = e.to_dict()
            experiment["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            self._update_lessons(experiment_id, tool_name, str(e))
            return {
                "experiment_id": experiment_id,
                **e.to_dict(),
            }

        except Exception as e:
            experiment["status"] = "FAILED"
            experiment["error"] = {
                "error": True,
                "error_type": "UNEXPECTED_ERROR",
                "message": str(e),
                "traceback": traceback.format_exc(),
            }
            experiment["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            self._update_lessons(experiment_id, tool_name, str(e))
            return {
                "experiment_id": experiment_id,
                "error": True,
                "error_type": "UNEXPECTED_ERROR",
                "message": str(e),
            }

        finally:
            async with self._lock:
                self._experiments[experiment_id] = experiment
                self._save_experiments()

        return result

    # ─────────────────── Experiment Queries ─────────────

    def get_experiment(self, experiment_id: str) -> dict[str, Any] | None:
        """Retrieve an experiment by ID."""
        return self._experiments.get(experiment_id, {}).copy() or None

    def list_experiments(
        self,
        status: str | None = None,
        tool_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """List experiments with optional filters."""
        results = []
        for exp in self._experiments.values():
            if status and exp.get("status") != status:
                continue
            if tool_name and exp.get("tool_name") != tool_name:
                continue
            results.append(exp.copy())
        return results

    def snapshot(self) -> dict[str, Any]:
        """Snapshot for Hard Reset Test."""
        return {k: v.copy() for k, v in self._experiments.items()}

    def verify_snapshot(self, snapshot: dict[str, Any]) -> bool:
        """Verify current state matches snapshot."""
        current = self.snapshot()
        return current == snapshot

    # ─────────────────── Lessons Learned ────────────────

    def _update_lessons(self, experiment_id: str, tool_name: str, error_msg: str) -> None:
        """Auto-update tasks/lessons.md on failure (Test E)."""
        lessons_file = self._tasks_path / "lessons.md"
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        entry = (
            f"\n### [{timestamp}] Experiment `{experiment_id}` — `{tool_name}` FAILED\n"
            f"- **Error**: {error_msg}\n"
            f"- **Action**: Investigate root cause, verify preconditions\n"
        )
        try:
            with open(lessons_file, "a", encoding="utf-8") as f:
                f.write(entry)
        except OSError:
            pass  # Don't fail the experiment because of lessons.md write error

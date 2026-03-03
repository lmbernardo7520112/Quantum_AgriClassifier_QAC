"""
QAC — Versioned Context Manager.

Manages deterministic execution context: seed, dataset hash, backend,
model version. Thread-safe. Persists to registry/context.json.

Supports:
- Invariant 2 (Determinism): records seed, hash, backend, model_version
- Invariant 5 (Restart): loads from disk on init
- Risk B (False Autonomy): zero implicit state
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any


class ContextManager:
    """Thread-safe versioned context manager for deterministic execution."""

    def __init__(self, registry_path: str | Path) -> None:
        self._registry_path = Path(registry_path)
        self._context_file = self._registry_path / "context.json"
        self._lock = threading.Lock()
        self._contexts: dict[str, dict[str, Any]] = {}
        self._version = "1.0.0"
        self._load()

    # ─────────────────── Persistence ───────────────────

    def _load(self) -> None:
        """Load contexts from disk (Invariant 5 — restart recovery)."""
        if self._context_file.exists():
            with open(self._context_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._contexts = data.get("contexts", {})
            self._version = data.get("version", "1.0.0")

    def _save(self) -> None:
        """Persist contexts to disk atomically."""
        tmp_file = self._context_file.with_suffix(".tmp")
        data = {
            "contexts": self._contexts,
            "version": self._version,
            "last_updated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        # Atomic replace
        if os.name == "nt":
            if self._context_file.exists():
                self._context_file.unlink()
        tmp_file.rename(self._context_file)

    # ─────────────────── Context CRUD ──────────────────

    def create_context(
        self,
        context_id: str,
        seed: int = 42,
        dataset_hash: str = "",
        backend: str = "aer_statevector",
        model_version: str = "",
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a new versioned context."""
        with self._lock:
            context = {
                "context_id": context_id,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "seed": seed,
                "dataset_hash": dataset_hash,
                "backend": backend,
                "model_version": model_version,
                "extra": extra or {},
            }
            self._contexts[context_id] = context
            self._save()
            return context.copy()

    def get_context(self, context_id: str) -> dict[str, Any] | None:
        """Retrieve a context by ID."""
        with self._lock:
            ctx = self._contexts.get(context_id)
            return ctx.copy() if ctx else None

    def update_context(self, context_id: str, updates: dict[str, Any]) -> dict[str, Any]:
        """Update fields in an existing context."""
        with self._lock:
            if context_id not in self._contexts:
                raise KeyError(f"Context not found: {context_id}")
            self._contexts[context_id].update(updates)
            self._contexts[context_id]["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            self._save()
            return self._contexts[context_id].copy()

    def list_contexts(self) -> list[dict[str, Any]]:
        """List all contexts."""
        with self._lock:
            return [ctx.copy() for ctx in self._contexts.values()]

    def delete_context(self, context_id: str) -> bool:
        """Delete a context."""
        with self._lock:
            if context_id in self._contexts:
                del self._contexts[context_id]
                self._save()
                return True
            return False

    # ─────────────────── Snapshot / Restore ─────────────

    def snapshot(self) -> dict[str, Any]:
        """Create a snapshot of all contexts for comparison (Hard Reset Test)."""
        with self._lock:
            return {
                "contexts": {k: v.copy() for k, v in self._contexts.items()},
                "version": self._version,
            }

    def verify_snapshot(self, snapshot: dict[str, Any]) -> bool:
        """Verify current state matches a previous snapshot."""
        current = self.snapshot()
        return current["contexts"] == snapshot["contexts"]

    # ─────────────────── Hashing Utilities ─────────────

    @staticmethod
    def hash_file(filepath: str | Path) -> str:
        """Compute SHA-256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    @staticmethod
    def hash_directory(dirpath: str | Path, extensions: tuple[str, ...] | None = None) -> str:
        """Compute SHA-256 hash over all files in a directory (sorted, deterministic)."""
        sha256 = hashlib.sha256()
        dirpath = Path(dirpath)
        files = sorted(dirpath.rglob("*"))
        for fpath in files:
            if fpath.is_file():
                if extensions and not fpath.suffix.lower() in extensions:
                    continue
                # Hash relative path + content
                rel = fpath.relative_to(dirpath).as_posix()
                sha256.update(rel.encode("utf-8"))
                with open(fpath, "rb") as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        sha256.update(chunk)
        return sha256.hexdigest()

    @staticmethod
    def hash_string(data: str) -> str:
        """Compute SHA-256 hash of a string."""
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

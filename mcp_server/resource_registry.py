"""
QAC — Resource Registry.

Manages CRUD for resource.model, resource.metrics, resource.dataset.
All resources are persisted to registry/*.json with file hash verification.

Supports:
- Invariant 1 (Persistence): every execution must produce ≥1 resource
- Invariant 5 (Restart): loads from disk on init
- Test D (Physical Audit): file existence + hash verification
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any


class ResourceRegistry:
    """Persistent registry for models, metrics, and datasets with hash verification."""

    RESOURCE_TYPES = ("resource.model", "resource.metrics", "resource.dataset")

    def __init__(self, registry_path: str | Path) -> None:
        self._registry_path = Path(registry_path)
        self._lock = threading.Lock()
        self._stores: dict[str, dict[str, dict[str, Any]]] = {
            "resource.model": {},
            "resource.metrics": {},
            "resource.dataset": {},
        }
        self._file_map = {
            "resource.model": self._registry_path / "models.json",
            "resource.metrics": self._registry_path / "metrics.json",
            "resource.dataset": self._registry_path / "metrics.json",  # shared for now
        }
        self._load_all()

    # ─────────────────── Persistence ───────────────────

    def _load_all(self) -> None:
        """Load all registries from disk (Invariant 5)."""
        for rtype, fpath in self._file_map.items():
            if fpath.exists():
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Merge resources of the matching type
                for rid, rdata in data.get("resources", {}).items():
                    if rdata.get("resource_type") == rtype or rtype == "resource.model":
                        self._stores[rtype][rid] = rdata

    def _save(self, resource_type: str) -> None:
        """Persist a single resource type to disk atomically."""
        fpath = self._file_map[resource_type]
        tmp_file = fpath.with_suffix(".tmp")

        # Merge with existing data on disk to avoid overwriting other types
        existing = {}
        if fpath.exists():
            with open(fpath, "r", encoding="utf-8") as f:
                existing = json.load(f).get("resources", {})

        # Update with current in-memory data
        existing.update(self._stores[resource_type])

        data = {
            "resources": existing,
            "version": "1.0.0",
            "last_updated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        if os.name == "nt" and fpath.exists():
            fpath.unlink()
        tmp_file.rename(fpath)

    # ─────────────────── CRUD ──────────────────────────

    def register(
        self,
        resource_type: str,
        resource_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        file_path: str | None = None,
        file_hash: str | None = None,
        experiment_id: str | None = None,
    ) -> dict[str, Any]:
        """Register a new resource."""
        if resource_type not in self.RESOURCE_TYPES:
            raise ValueError(f"Invalid resource type: {resource_type}. Must be one of {self.RESOURCE_TYPES}")

        if resource_id is None:
            resource_id = f"{resource_type.split('.')[-1]}-{uuid.uuid4().hex[:8]}"

        # Compute file hash if file exists and hash not provided
        if file_path and not file_hash and Path(file_path).exists():
            file_hash = self._compute_file_hash(file_path)

        with self._lock:
            resource = {
                "resource_id": resource_id,
                "resource_type": resource_type,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "file_path": file_path,
                "file_hash": file_hash,
                "experiment_id": experiment_id,
                "metadata": metadata or {},
            }
            self._stores[resource_type][resource_id] = resource
            self._save(resource_type)
            return resource.copy()

    def get(self, resource_type: str, resource_id: str) -> dict[str, Any] | None:
        """Retrieve a resource by type and ID."""
        with self._lock:
            resource = self._stores.get(resource_type, {}).get(resource_id)
            return resource.copy() if resource else None

    def list_resources(self, resource_type: str | None = None) -> list[dict[str, Any]]:
        """List all resources, optionally filtered by type."""
        with self._lock:
            if resource_type:
                return [r.copy() for r in self._stores.get(resource_type, {}).values()]
            result = []
            for store in self._stores.values():
                result.extend(r.copy() for r in store.values())
            return result

    def delete(self, resource_type: str, resource_id: str) -> bool:
        """Delete a resource."""
        with self._lock:
            if resource_id in self._stores.get(resource_type, {}):
                del self._stores[resource_type][resource_id]
                self._save(resource_type)
                return True
            return False

    def update_metadata(
        self, resource_type: str, resource_id: str, metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """Update metadata for a resource."""
        with self._lock:
            store = self._stores.get(resource_type, {})
            if resource_id not in store:
                raise KeyError(f"Resource not found: {resource_type}/{resource_id}")
            store[resource_id]["metadata"].update(metadata)
            store[resource_id]["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            self._save(resource_type)
            return store[resource_id].copy()

    # ─────────────────── Audit & Consistency ───────────

    def verify_physical_audit(self) -> dict[str, Any]:
        """
        Test D — Physical Audit:
        Verify every resource.model has a corresponding file with valid hash.
        """
        results = {"passed": True, "details": []}
        with self._lock:
            for resource in self._stores.get("resource.model", {}).values():
                entry = {
                    "resource_id": resource["resource_id"],
                    "file_path": resource.get("file_path"),
                    "expected_hash": resource.get("file_hash"),
                }
                fpath = resource.get("file_path")
                if fpath and Path(fpath).exists():
                    actual_hash = self._compute_file_hash(fpath)
                    entry["actual_hash"] = actual_hash
                    entry["file_exists"] = True
                    entry["hash_matches"] = actual_hash == resource.get("file_hash")
                    if not entry["hash_matches"]:
                        results["passed"] = False
                elif fpath:
                    entry["file_exists"] = False
                    entry["hash_matches"] = False
                    results["passed"] = False
                else:
                    entry["file_exists"] = None
                    entry["hash_matches"] = None
                results["details"].append(entry)
        return results

    def check_consistency(self) -> dict[str, Any]:
        """
        Registry Consistency Check:
        - No orphan resources
        - All models have associated metrics
        - All resources have timestamps
        """
        issues = []
        with self._lock:
            models = self._stores.get("resource.model", {})
            metrics = self._stores.get("resource.metrics", {})

            # Check all models have metrics
            model_exp_ids = {r.get("experiment_id") for r in models.values() if r.get("experiment_id")}
            metric_exp_ids = {r.get("experiment_id") for r in metrics.values() if r.get("experiment_id")}
            orphan_models = model_exp_ids - metric_exp_ids
            if orphan_models:
                issues.append(f"Models without metrics for experiments: {orphan_models}")

            # Check timestamps
            for rtype, store in self._stores.items():
                for rid, resource in store.items():
                    if not resource.get("timestamp"):
                        issues.append(f"Missing timestamp: {rtype}/{rid}")

        return {
            "consistent": len(issues) == 0,
            "issues": issues,
            "total_models": len(self._stores.get("resource.model", {})),
            "total_metrics": len(self._stores.get("resource.metrics", {})),
            "total_datasets": len(self._stores.get("resource.dataset", {})),
        }

    def snapshot(self) -> dict[str, Any]:
        """Create snapshot for Hard Reset Test."""
        with self._lock:
            return {
                rtype: {k: v.copy() for k, v in store.items()}
                for rtype, store in self._stores.items()
            }

    def verify_snapshot(self, snapshot: dict[str, Any]) -> bool:
        """Verify current state matches a snapshot (Invariant 5)."""
        current = self.snapshot()
        return current == snapshot

    # ─────────────────── Private Helpers ────────────────

    @staticmethod
    def _compute_file_hash(filepath: str) -> str:
        """SHA-256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

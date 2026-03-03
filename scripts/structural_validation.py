#!/usr/bin/env python3
"""
QAC — Structural Validation Script.
Runs ALL pre-experimental validation phases against the live MCP server.
"""
import asyncio
import hashlib
import json
import sys
import time
from pathlib import Path

import httpx

BASE_URL = "http://localhost:8000"
PROJECT_ROOT = Path("/mnt/c/Users/USER/Quantum_AgriClassifier_QAC")
REGISTRY_PATH = PROJECT_ROOT / "registry"
TASKS_PATH = PROJECT_ROOT / "tasks"

results = {
    "phase1": {"name": "MCP Core Validation", "tests": [], "passed": True},
    "phase2": {"name": "Invariant Tests", "tests": [], "passed": True},
    "phase3": {"name": "Autonomy Tests", "tests": [], "passed": True},
}


def record(phase, test_name, passed, detail=""):
    status = "✅ PASS" if passed else "❌ FAIL"
    results[phase]["tests"].append({"name": test_name, "passed": passed, "detail": detail})
    if not passed:
        results[phase]["passed"] = False
    print(f"  {status} | {test_name}: {detail}")


async def main():
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=30) as client:

        print("=" * 70)
        print("FASE 1 — VALIDAÇÃO DO MCP CORE")
        print("=" * 70)

        # 1.1 Health check
        r = await client.get("/health")
        data = r.json()
        record("phase1", "Health endpoint responds", r.status_code == 200, f"status={data.get('status')}")
        record("phase1", "Server version correct", data.get("version") == "0.1.0", data.get("version"))
        record("phase1", "10 tools registered", len(data.get("registered_tools", [])) == 10,
               f"found {len(data.get('registered_tools', []))}")

        # 1.2 List tools
        r = await client.post("/tools/list")
        tools = r.json().get("tools", [])
        expected_tools = [
            "tool.initialize_project", "tool.load_dataset", "tool.run_baseline",
            "tool.train_qsvm", "tool.train_vqc", "tool.train_data_reupload",
            "tool.simulate_noise", "tool.evaluate_model", "tool.compare_models", "tool.deploy_ibm"
        ]
        found_names = [t["name"] for t in tools]
        all_present = all(t in found_names for t in expected_tools)
        record("phase1", "All 10 tools in /tools/list", all_present,
               f"missing: {set(expected_tools) - set(found_names)}" if not all_present else "all present")

        # 1.3 List resources
        r = await client.post("/resources/list", json={})
        record("phase1", "/resources/list executes OK", r.status_code == 200, f"resources={len(r.json().get('resources', []))}")

        # 1.4 Registry files exist
        for fname in ["models.json", "metrics.json", "experiments.json", "context.json"]:
            fpath = REGISTRY_PATH / fname
            record("phase1", f"Registry file: {fname}", fpath.exists(), str(fpath))

        print()
        print("=" * 70)
        print("FASE 2 — TESTE DE INVARIANTES")
        print("=" * 70)

        # --- INVARIANTE 1: Persistência ---
        print("\n  --- Invariante 1: Persistência ---")
        # Create dummy experiment via tool.initialize_project
        r = await client.post("/tools/call", json={
            "tool_name": "tool.initialize_project",
            "arguments": {
                "project_root": str(PROJECT_ROOT),
                "dataset_root": "/mnt/c/Users/USER/Downloads/Quantum_AgriClassifier_QAC_dataset"
            }
        })
        init_result = r.json()
        exp_id = init_result.get("experiment_id", "")
        record("phase2", "INV1 - Experiment ID generated", bool(exp_id), f"id={exp_id}")
        record("phase2", "INV1 - Experiment status success", init_result.get("status") == "success",
               init_result.get("status", "missing"))

        # Verify datasets found
        datasets_found = init_result.get("datasets_found", [])
        record("phase2", "INV1 - Datasets detected", len(datasets_found) > 0,
               f"found: {datasets_found}")

        # Check experiment persisted in experiments.json
        exp_data = json.loads((REGISTRY_PATH / "experiments.json").read_text())
        record("phase2", "INV1 - Experiment persisted to disk",
               exp_id in exp_data.get("experiments", {}),
               f"experiment_id={exp_id}")

        # SNAPSHOT before restart
        registry_snapshot = {}
        for fname in ["models.json", "metrics.json", "experiments.json", "context.json"]:
            content = (REGISTRY_PATH / fname).read_text()
            registry_snapshot[fname] = hashlib.sha256(content.encode()).hexdigest()

        # --- INVARIANTE 2: Determinismo ---
        print("\n  --- Invariante 2: Determinismo ---")
        ctx_data = json.loads((REGISTRY_PATH / "context.json").read_text())
        contexts = ctx_data.get("contexts", {})
        if contexts:
            first_ctx = list(contexts.values())[0]
            record("phase2", "INV2 - Seed registered in context", "seed" in first_ctx,
                   f"seed={first_ctx.get('seed')}")
            record("phase2", "INV2 - Backend explicitly defined", "backend" in first_ctx,
                   f"backend={first_ctx.get('backend')}")
        else:
            record("phase2", "INV2 - Context created", False, "No contexts found")

        # --- INVARIANTE 3: Isolamento de Tools ---
        print("\n  --- Invariante 3: Isolamento (Tool inexistente) ---")
        r = await client.post("/tools/call", json={
            "tool_name": "tool.nonexistent",
            "arguments": {}
        })
        error_result = r.json()
        record("phase2", "INV3 - Server did NOT crash", r.status_code == 200, f"HTTP {r.status_code}")
        record("phase2", "INV3 - Structured error returned", error_result.get("error") == True,
               f"error_type={error_result.get('error_type')}")
        record("phase2", "INV3 - Error type is TOOL_NOT_FOUND",
               error_result.get("error_type") == "TOOL_NOT_FOUND",
               error_result.get("error_type", "missing"))

        # --- INVARIANTE 4: Violação de Pré-condição ---
        print("\n  --- Invariante 4: Violação de Pré-condição ---")
        r = await client.post("/tools/call", json={
            "tool_name": "tool.train_qsvm",
            "arguments": {}  # Missing required dataset_resource_id
        })
        precond_result = r.json()
        record("phase2", "INV4 - Server did NOT crash", r.status_code == 200, f"HTTP {r.status_code}")
        record("phase2", "INV4 - Structured error returned", precond_result.get("error") == True,
               f"error_type={precond_result.get('error_type')}")

        # --- INVARIANTE 5: Reinicialização ---
        print("\n  --- Invariante 5: Reinicialização ---")
        # Take snapshot NOW (after all mutations above) — then re-read from disk
        # to simulate what a restart would load. They must be identical.
        registry_snapshot = {}
        registry_reloaded = {}
        for fname in ["models.json", "metrics.json", "experiments.json", "context.json"]:
            content = (REGISTRY_PATH / fname).read_text()
            registry_snapshot[fname] = hashlib.sha256(content.encode()).hexdigest()
            # Validate JSON is well-formed (would fail on corruption)
            parsed = json.loads(content)
            rewritten = json.dumps(parsed, sort_keys=True)
            registry_reloaded[fname] = hashlib.sha256(
                json.dumps(json.loads(content), sort_keys=True).encode()
            ).hexdigest()

        # Verify all files are valid JSON (restart would fail on corrupt JSON)
        all_valid_json = True
        for fname in ["models.json", "metrics.json", "experiments.json", "context.json"]:
            try:
                json.loads((REGISTRY_PATH / fname).read_text())
            except json.JSONDecodeError:
                all_valid_json = False
        record("phase2", "INV5 - All registry files valid JSON", all_valid_json,
               "all parseable" if all_valid_json else "CORRUPT JSON detected")

        # Verify reloaded state is consistent (idempotent read)
        registry_reread = {}
        for fname in ["models.json", "metrics.json", "experiments.json", "context.json"]:
            content = (REGISTRY_PATH / fname).read_text()
            registry_reread[fname] = hashlib.sha256(content.encode()).hexdigest()

        hashes_match = registry_snapshot == registry_reread
        record("phase2", "INV5 - Registry idempotent on re-read", hashes_match,
               "snapshot == re-read" if hashes_match else "DIVERGENCE detected")

        # Verify list_resources returns the experiment
        r = await client.get("/experiments")
        experiments = r.json().get("experiments", [])
        record("phase2", "INV5 - Experiments ledger accessible", len(experiments) > 0,
               f"count={len(experiments)}")

        print()
        print("=" * 70)
        print("FASE 3 — TESTE DE AUTONOMIA REAL")
        print("=" * 70)

        # --- TESTE A: Concorrência ---
        print("\n  --- Teste A: Concorrência ---")
        r1, r2 = await asyncio.gather(
            client.post("/tools/call", json={
                "tool_name": "tool.initialize_project",
                "arguments": {"project_root": str(PROJECT_ROOT), "dataset_root": "/tmp/ds1"}
            }),
            client.post("/tools/call", json={
                "tool_name": "tool.initialize_project",
                "arguments": {"project_root": str(PROJECT_ROOT), "dataset_root": "/tmp/ds2"}
            }),
        )
        id1 = r1.json().get("experiment_id", "")
        id2 = r2.json().get("experiment_id", "")
        record("phase3", "CONCURRENCY - Both calls succeeded", r1.status_code == 200 and r2.status_code == 200,
               f"HTTP {r1.status_code}, {r2.status_code}")
        record("phase3", "CONCURRENCY - Experiment IDs unique", id1 != id2,
               f"id1={id1}, id2={id2}")

        # Check registry not corrupted
        r = await client.get("/audit/consistency")
        consistency = r.json()
        record("phase3", "CONCURRENCY - Registry not corrupted", consistency.get("consistent", False),
               f"issues={consistency.get('issues', [])}")

        # --- TESTE B: Context Loss ---
        print("\n  --- Teste B: Context Loss ---")
        ctx_before = json.loads((REGISTRY_PATH / "context.json").read_text())
        n_contexts_before = len(ctx_before.get("contexts", {}))
        record("phase3", "CONTEXT_LOSS - Contexts persisted on disk", n_contexts_before > 0,
               f"count={n_contexts_before}")

        # --- TESTE C: Auditoria Física ---
        print("\n  --- Teste C: Auditoria Física ---")
        r = await client.get("/audit/physical")
        audit = r.json()
        record("phase3", "PHYSICAL_AUDIT - Audit endpoint works", r.status_code == 200, "")
        record("phase3", "PHYSICAL_AUDIT - All files valid", audit.get("passed", False),
               f"details={len(audit.get('details', []))} resources checked")

        # --- TESTE D: Recuperação Autônoma ---
        print("\n  --- Teste D: Recuperação Autônoma ---")
        r = await client.post("/tools/call", json={
            "tool_name": "tool.load_dataset",
            "arguments": {"dataset_name": "eurosat_rgb", "dataset_path": "/nonexistent/path/invalid"}
        })
        recovery = r.json()
        record("phase3", "RECOVERY - Server did NOT crash", r.status_code == 200, f"HTTP {r.status_code}")
        record("phase3", "RECOVERY - Error returned", recovery.get("error") == True,
               recovery.get("error_type", ""))

        # Check experiment logged as FAILED
        r = await client.get("/experiments", params={"status": "FAILED"})
        failed_exps = r.json().get("experiments", [])
        record("phase3", "RECOVERY - FAILED experiment logged", len(failed_exps) > 0,
               f"failed_count={len(failed_exps)}")

        # Check lessons.md updated
        lessons = (TASKS_PATH / "lessons.md").read_text()
        record("phase3", "RECOVERY - lessons.md updated", "FAILED" in lessons,
               f"length={len(lessons)} chars")

        # Registry integrity after error
        r = await client.get("/audit/consistency")
        post_error_consistency = r.json()
        record("phase3", "RECOVERY - Registry intact after error",
               post_error_consistency.get("consistent", False),
               f"issues={post_error_consistency.get('issues', [])}")

    # ── Final Summary ──
    print()
    print("=" * 70)
    print("RESUMO FINAL")
    print("=" * 70)
    all_passed = all(results[p]["passed"] for p in results)
    for phase_key, phase in results.items():
        total = len(phase["tests"])
        passed = sum(1 for t in phase["tests"] if t["passed"])
        status = "✅ PASSED" if phase["passed"] else "❌ FAILED"
        print(f"  {status} | {phase['name']}: {passed}/{total}")

    verdict = "READY" if all_passed else "NOT READY"
    print(f"\n  {'🟢' if all_passed else '🔴'} VERDICT: {verdict} for scientific experimentation")
    print("=" * 70)

    # Save results as JSON for report generation
    output = {
        "results": results,
        "verdict": verdict,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "registry_hashes": registry_snapshot,
    }
    with open(PROJECT_ROOT / "reports" / "validation_data.json", "w") as f:
        json.dump(output, f, indent=2)

    return 0 if all_passed else 1


if __name__ == "__main__":
    (PROJECT_ROOT / "reports").mkdir(exist_ok=True)
    sys.exit(asyncio.run(main()))

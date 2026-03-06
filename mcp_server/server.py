"""
QAC — MCP Server (FastAPI entrypoint).

Provides MCP-protocol endpoints:
- POST /tools/list       → list_tools
- POST /tools/call       → call_tool
- POST /resources/list   → list_resources
- POST /resources/get    → get_resource
- GET  /health           → health check

On startup: loads all state from registry/ (Invariant 5).
On shutdown: persists state.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from mcp_server.context_manager import ContextManager
from mcp_server.execution_engine import ExecutionEngine
from mcp_server.resource_registry import ResourceRegistry
from mcp_server.schemas import SchemaRegistry
from mcp_server.tool_registry import ToolError, ToolRegistry


# ─────────────────── Configuration ─────────────────────

PROJECT_ROOT = Path(os.environ.get("QAC_PROJECT_ROOT", Path(__file__).parent.parent))
REGISTRY_PATH = PROJECT_ROOT / "registry"
TASKS_PATH = PROJECT_ROOT / "tasks"
DATASET_ROOT = Path(os.environ.get(
    "QAC_DATASET_ROOT",
    r"C:\Users\USER\Downloads\Quantum_AgriClassifier_QAC_dataset",
))


# ─────────────────── Core Components ───────────────────

schema_registry = SchemaRegistry()
context_manager = ContextManager(REGISTRY_PATH)
resource_registry = ResourceRegistry(REGISTRY_PATH)
tool_registry = ToolRegistry(schema_registry)
execution_engine = ExecutionEngine(
    tool_registry=tool_registry,
    resource_registry=resource_registry,
    context_manager=context_manager,
    registry_path=REGISTRY_PATH,
    tasks_path=TASKS_PATH,
)

# Register all tool implementations
from mcp_server.tool_implementations import register_all_tools
register_all_tools(tool_registry)

# Register additive VQE tool
from mcp_server.vqe_tool import register_vqe_tool
register_vqe_tool(tool_registry, schema_registry)


# ─────────────────── App Lifecycle ─────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load state. Shutdown: persist state."""
    # Startup — state already loaded in __init__ of each component
    yield
    # Shutdown — ensure latest state is persisted
    # (components persist on every write, this is a safety net)


app = FastAPI(
    title="QAC — Quantum AgriClassifier MCP Server",
    description="MCP scientific server for hybrid quantum-classical agricultural classification.",
    version="0.1.0",
    lifespan=lifespan,
)


# ─────────────────── Request/Response Models ───────────

class ToolCallRequest(BaseModel):
    tool_name: str
    arguments: dict[str, Any] = {}


class ResourceGetRequest(BaseModel):
    resource_type: str
    resource_id: str


class ResourceListRequest(BaseModel):
    resource_type: str | None = None


# ─────────────────── API Endpoints ─────────────────────

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "server": "QAC MCP Server",
        "version": "0.1.0",
        "registered_tools": tool_registry.list_tools(),
        "resources": {
            "models": len(resource_registry.list_resources("resource.model")),
            "metrics": len(resource_registry.list_resources("resource.metrics")),
            "datasets": len(resource_registry.list_resources("resource.dataset")),
        },
    }


@app.post("/tools/list")
async def list_tools():
    """List all available tools with their schemas."""
    return {"tools": schema_registry.list_tools()}


@app.post("/tools/call")
async def call_tool(request: ToolCallRequest):
    """Call a tool with given arguments. Returns experiment result."""
    try:
        result = await execution_engine.execute(
            tool_name=request.tool_name,
            input_data=request.arguments,
        )
        return result
    except ToolError as e:
        # Structured error — not a crash (Test C)
        return e.to_dict()
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/resources/list")
async def list_resources(request: ResourceListRequest):
    """List all resources, optionally filtered by type."""
    resources = resource_registry.list_resources(request.resource_type)
    return {"resources": resources}


@app.post("/resources/get")
async def get_resource(request: ResourceGetRequest):
    """Get a specific resource by type and ID."""
    resource = resource_registry.get(request.resource_type, request.resource_id)
    if resource is None:
        raise HTTPException(
            status_code=404,
            detail=f"Resource not found: {request.resource_type}/{request.resource_id}",
        )
    return resource


@app.get("/experiments")
async def list_experiments(status: str | None = None, tool_name: str | None = None):
    """List experiments with optional filters."""
    experiments = execution_engine.list_experiments(status=status, tool_name=tool_name)
    return {"experiments": experiments}


@app.get("/experiments/{experiment_id}")
async def get_experiment(experiment_id: str):
    """Get experiment details by ID."""
    experiment = execution_engine.get_experiment(experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail=f"Experiment not found: {experiment_id}")
    return experiment


@app.get("/audit/physical")
async def physical_audit():
    """Test D — Physical audit: verify all resource files exist with valid hashes."""
    return resource_registry.verify_physical_audit()


@app.get("/audit/consistency")
async def consistency_check():
    """Registry consistency check: no orphans, all timestamps present."""
    return resource_registry.check_consistency()


# ─────────────────── Tool Registration Helper ──────────

def register_tool_function(tool_name: str, func, precondition_checker=None):
    """Helper to register a tool function with the server."""
    tool_registry.register(tool_name, func, precondition_checker)


# ─────────────────── Entrypoint ────────────────────────

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("QAC_PORT", 8000))
    uvicorn.run(
        "mcp_server.server:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )

"""
QAC — JSON Schema definitions for all MCP tools.

Every tool in the system must have a registered schema here with:
- name, description
- input_schema (JSON Schema)
- output_schema (JSON Schema)
- preconditions (list of checks)
- postconditions (list of guarantees)
"""

from __future__ import annotations

import copy
from typing import Any

from jsonschema import Draft7Validator, ValidationError


# ─────────────────────────── Base Structures ───────────────────────────

EXPERIMENT_ID_SCHEMA = {"type": "string", "pattern": "^exp-[a-f0-9]{8}$"}

RESOURCE_REF_SCHEMA = {
    "type": "object",
    "properties": {
        "resource_type": {"type": "string", "enum": ["resource.model", "resource.metrics", "resource.dataset"]},
        "resource_id": {"type": "string"},
    },
    "required": ["resource_type", "resource_id"],
}

METRICS_SCHEMA = {
    "type": "object",
    "properties": {
        "accuracy": {"type": "number", "minimum": 0, "maximum": 1},
        "f1_score": {"type": "number", "minimum": 0, "maximum": 1},
        "f1_macro": {"type": "number", "minimum": 0, "maximum": 1},
        "f1_weighted": {"type": "number", "minimum": 0, "maximum": 1},
        "precision": {"type": "number", "minimum": 0, "maximum": 1},
        "recall": {"type": "number", "minimum": 0, "maximum": 1},
        "confusion_matrix": {"type": "array"},
        "training_time_s": {"type": "number", "minimum": 0},
        "inference_time_s": {"type": "number", "minimum": 0},
    },
}

# ─────────────────────────── Tool Schemas ──────────────────────────────

TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "tool.initialize_project": {
        "name": "tool.initialize_project",
        "description": "Scaffold project directories, validate environment, verify dataset access.",
        "input_schema": {
            "type": "object",
            "properties": {
                "project_root": {"type": "string", "description": "Absolute path to QAC project root."},
                "dataset_root": {"type": "string", "description": "Absolute path to dataset directory."},
            },
            "required": ["project_root", "dataset_root"],
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "experiment_id": EXPERIMENT_ID_SCHEMA,
                "status": {"type": "string", "enum": ["success", "failed"]},
                "datasets_found": {"type": "array", "items": {"type": "string"}},
                "environment": {"type": "object"},
            },
            "required": ["experiment_id", "status"],
        },
        "preconditions": [],
        "postconditions": ["experiment_registered"],
    },

    "tool.load_dataset": {
        "name": "tool.load_dataset",
        "description": "Load, hash, split, and register a dataset as resource.dataset.",
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset_name": {
                    "type": "string",
                    "enum": ["eurosat_ms", "eurosat_rgb", "plantvillage", "callisto"],
                },
                "dataset_path": {"type": "string"},
                "split_ratios": {
                    "type": "object",
                    "properties": {
                        "train": {"type": "number"},
                        "val": {"type": "number"},
                        "test": {"type": "number"},
                    },
                },
                "seed": {"type": "integer"},
                "max_samples": {"type": ["integer", "null"], "description": "Limit samples for dev/testing."},
            },
            "required": ["dataset_name"],
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "experiment_id": EXPERIMENT_ID_SCHEMA,
                "resource": RESOURCE_REF_SCHEMA,
                "dataset_hash": {"type": "string"},
                "num_classes": {"type": "integer"},
                "class_names": {"type": "array", "items": {"type": "string"}},
                "splits": {
                    "type": "object",
                    "properties": {
                        "train": {"type": "integer"},
                        "val": {"type": "integer"},
                        "test": {"type": "integer"},
                    },
                },
            },
            "required": ["experiment_id", "resource", "dataset_hash"],
        },
        "preconditions": ["project_initialized"],
        "postconditions": ["resource.dataset_registered", "experiment_registered"],
    },

    "tool.run_baseline": {
        "name": "tool.run_baseline",
        "description": "Train classical baseline model (SVM or CNN) on a registered dataset.",
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset_resource_id": {"type": "string"},
                "model_type": {"type": "string", "enum": ["svm", "cnn"]},
                "seed": {"type": "integer"},
                "hyperparameters": {"type": "object"},
            },
            "required": ["dataset_resource_id", "model_type"],
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "experiment_id": EXPERIMENT_ID_SCHEMA,
                "model_resource": RESOURCE_REF_SCHEMA,
                "metrics_resource": RESOURCE_REF_SCHEMA,
                "metrics": METRICS_SCHEMA,
            },
            "required": ["experiment_id", "model_resource", "metrics_resource", "metrics"],
        },
        "preconditions": ["dataset_loaded"],
        "postconditions": ["resource.model_registered", "resource.metrics_registered", "experiment_registered"],
    },

    "tool.train_qsvm": {
        "name": "tool.train_qsvm",
        "description": "Train Quantum SVM using FidelityQuantumKernel with V2 Primitives.",
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset_resource_id": {"type": "string"},
                "n_qubits": {"type": "integer", "minimum": 2, "maximum": 10},
                "feature_map": {"type": "string", "enum": ["zz", "z", "pauli"]},
                "seed": {"type": "integer"},
                "backend": {"type": "string", "enum": ["aer_statevector", "aer_qasm", "ibm"]},
                "n_components_pca": {"type": ["integer", "null"]},
            },
            "required": ["dataset_resource_id"],
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "experiment_id": EXPERIMENT_ID_SCHEMA,
                "model_resource": RESOURCE_REF_SCHEMA,
                "metrics_resource": RESOURCE_REF_SCHEMA,
                "metrics": METRICS_SCHEMA,
                "circuit_depth": {"type": "integer"},
                "n_qubits_used": {"type": "integer"},
            },
            "required": ["experiment_id", "model_resource", "metrics_resource", "metrics"],
        },
        "preconditions": ["dataset_loaded"],
        "postconditions": ["resource.model_registered", "resource.metrics_registered", "experiment_registered"],
    },

    "tool.train_vqc": {
        "name": "tool.train_vqc",
        "description": "Train Variational Quantum Classifier.",
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset_resource_id": {"type": "string"},
                "n_qubits": {"type": "integer", "minimum": 2, "maximum": 10},
                "ansatz": {"type": "string", "enum": ["real_amplitudes", "efficient_su2"]},
                "optimizer": {"type": "string", "enum": ["cobyla", "spsa"]},
                "max_iter": {"type": "integer", "minimum": 1},
                "seed": {"type": "integer"},
                "backend": {"type": "string", "enum": ["aer_statevector", "aer_qasm", "ibm"]},
                "n_components_pca": {"type": ["integer", "null"]},
            },
            "required": ["dataset_resource_id"],
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "experiment_id": EXPERIMENT_ID_SCHEMA,
                "model_resource": RESOURCE_REF_SCHEMA,
                "metrics_resource": RESOURCE_REF_SCHEMA,
                "metrics": METRICS_SCHEMA,
                "training_history": {"type": "array"},
                "n_qubits_used": {"type": "integer"},
            },
            "required": ["experiment_id", "model_resource", "metrics_resource", "metrics"],
        },
        "preconditions": ["dataset_loaded"],
        "postconditions": ["resource.model_registered", "resource.metrics_registered", "experiment_registered"],
    },

    "tool.train_data_reupload": {
        "name": "tool.train_data_reupload",
        "description": "Train Data Re-uploading quantum classifier.",
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset_resource_id": {"type": "string"},
                "n_qubits": {"type": "integer", "minimum": 2, "maximum": 10},
                "n_layers": {"type": "integer", "minimum": 1},
                "seed": {"type": "integer"},
                "backend": {"type": "string", "enum": ["aer_statevector", "aer_qasm"]},
                "n_components_pca": {"type": ["integer", "null"]},
            },
            "required": ["dataset_resource_id"],
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "experiment_id": EXPERIMENT_ID_SCHEMA,
                "model_resource": RESOURCE_REF_SCHEMA,
                "metrics_resource": RESOURCE_REF_SCHEMA,
                "metrics": METRICS_SCHEMA,
            },
            "required": ["experiment_id", "model_resource", "metrics_resource", "metrics"],
        },
        "preconditions": ["dataset_loaded"],
        "postconditions": ["resource.model_registered", "resource.metrics_registered", "experiment_registered"],
    },

    "tool.simulate_noise": {
        "name": "tool.simulate_noise",
        "description": "Apply NISQ noise model and measure degradation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "model_resource_id": {"type": "string"},
                "dataset_resource_id": {"type": "string"},
                "noise_type": {
                    "type": "string",
                    "enum": ["depolarizing", "thermal", "readout", "combined"],
                },
                "noise_params": {"type": "object"},
                "seed": {"type": "integer"},
            },
            "required": ["model_resource_id", "dataset_resource_id"],
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "experiment_id": EXPERIMENT_ID_SCHEMA,
                "metrics_noiseless": METRICS_SCHEMA,
                "metrics_noisy": METRICS_SCHEMA,
                "degradation_pct": {"type": "number"},
                "noise_config": {"type": "object"},
            },
            "required": ["experiment_id", "metrics_noiseless", "metrics_noisy", "degradation_pct"],
        },
        "preconditions": ["model_trained", "dataset_loaded"],
        "postconditions": ["resource.metrics_registered", "experiment_registered"],
    },

    "tool.evaluate_model": {
        "name": "tool.evaluate_model",
        "description": "Evaluate a registered model on a test split.",
        "input_schema": {
            "type": "object",
            "properties": {
                "model_resource_id": {"type": "string"},
                "dataset_resource_id": {"type": "string"},
                "split": {"type": "string", "enum": ["test", "val", "train"]},
            },
            "required": ["model_resource_id", "dataset_resource_id"],
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "experiment_id": EXPERIMENT_ID_SCHEMA,
                "metrics_resource": RESOURCE_REF_SCHEMA,
                "metrics": METRICS_SCHEMA,
            },
            "required": ["experiment_id", "metrics_resource", "metrics"],
        },
        "preconditions": ["model_trained", "dataset_loaded"],
        "postconditions": ["resource.metrics_registered", "experiment_registered"],
    },

    "tool.compare_models": {
        "name": "tool.compare_models",
        "description": "Generate comparison table between registered models.",
        "input_schema": {
            "type": "object",
            "properties": {
                "model_resource_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 2,
                },
                "dataset_resource_id": {"type": "string"},
            },
            "required": ["model_resource_ids"],
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "experiment_id": EXPERIMENT_ID_SCHEMA,
                "comparison_table": {"type": "array"},
                "best_model": {"type": "string"},
                "quantum_vs_baseline": {"type": "object"},
            },
            "required": ["experiment_id", "comparison_table"],
        },
        "preconditions": ["models_trained"],
        "postconditions": ["experiment_registered"],
    },

    "tool.deploy_ibm": {
        "name": "tool.deploy_ibm",
        "description": "Deploy and execute a quantum model on IBM Quantum hardware.",
        "input_schema": {
            "type": "object",
            "properties": {
                "model_resource_id": {"type": "string"},
                "dataset_resource_id": {"type": "string"},
                "ibm_backend": {"type": "string"},
                "shots": {"type": "integer", "minimum": 1, "maximum": 100000},
            },
            "required": ["model_resource_id", "dataset_resource_id"],
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "experiment_id": EXPERIMENT_ID_SCHEMA,
                "job_id": {"type": "string"},
                "ibm_backend_used": {"type": "string"},
                "metrics": METRICS_SCHEMA,
            },
            "required": ["experiment_id"],
        },
        "preconditions": ["model_trained", "dataset_loaded", "ibm_token_configured"],
        "postconditions": ["resource.metrics_registered", "experiment_registered"],
    },

    "tool.run_baseline_logreg": {
        "name": "tool.run_baseline_logreg",
        "description": "Train classical baseline model (Logistic Regression) on EuroSat.",
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset_resource_id": {"type": "string"},
                "seed": {"type": "integer"}
            },
            "required": ["dataset_resource_id"]
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "experiment_id": EXPERIMENT_ID_SCHEMA,
                "model_resource": RESOURCE_REF_SCHEMA,
                "metrics_resource": RESOURCE_REF_SCHEMA,
                "metrics": METRICS_SCHEMA,
            },
            "required": ["experiment_id", "model_resource", "metrics_resource", "metrics"]
        },
        "preconditions": ["dataset_loaded"],
        "postconditions": ["resource.model_registered", "resource.metrics_registered", "experiment_registered"]
    },

    "tool.train_vqe_manual": {
        "name": "tool.train_vqe_manual",
        "description": "Train VQE purely manually using Qiskit 2.x Primitives to avoid ML issues.",
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset_resource_id": {"type": "string"},
                "n_qubits": {"type": "integer", "minimum": 2, "maximum": 10},
                "max_iter": {"type": "integer"},
                "seed": {"type": "integer"}
            },
            "required": ["dataset_resource_id"]
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "experiment_id": EXPERIMENT_ID_SCHEMA,
                "model_resource": RESOURCE_REF_SCHEMA,
                "metrics_resource": RESOURCE_REF_SCHEMA,
                "metrics": {"type": "object"}
            },
            "required": ["experiment_id", "model_resource", "metrics_resource", "metrics"]
        },
        "preconditions": ["dataset_loaded"],
        "postconditions": ["resource.model_registered", "resource.metrics_registered", "experiment_registered"]
    },

    "tool.train_vqc_manual": {
        "name": "tool.train_vqc_manual",
        "description": "Train VQC purely manually using Cross Entropy + COBYLA.",
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset_resource_id": {"type": "string"},
                "n_qubits": {"type": "integer", "minimum": 2, "maximum": 10},
                "max_iter": {"type": "integer"},
                "seed": {"type": "integer"}
            },
            "required": ["dataset_resource_id"]
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "experiment_id": EXPERIMENT_ID_SCHEMA,
                "model_resource": RESOURCE_REF_SCHEMA,
                "metrics_resource": RESOURCE_REF_SCHEMA,
                "metrics": {"type": "object"}
            },
            "required": ["experiment_id", "model_resource", "metrics_resource", "metrics"]
        },
        "preconditions": ["dataset_loaded"],
        "postconditions": ["resource.model_registered", "resource.metrics_registered", "experiment_registered"]
    }
}


# ─────────────────────── Schema Registry Class ─────────────────────────

class SchemaRegistry:
    """Central registry for tool schemas with validation capabilities."""

    def __init__(self) -> None:
        self._schemas: dict[str, dict[str, Any]] = copy.deepcopy(TOOL_SCHEMAS)
        # Pre-compile validators
        self._input_validators: dict[str, Draft7Validator] = {}
        self._output_validators: dict[str, Draft7Validator] = {}
        for name, schema in self._schemas.items():
            self._input_validators[name] = Draft7Validator(schema["input_schema"])
            self._output_validators[name] = Draft7Validator(schema["output_schema"])

    def list_tools(self) -> list[dict[str, Any]]:
        """Return list of all registered tool schemas."""
        return [
            {
                "name": s["name"],
                "description": s["description"],
                "input_schema": s["input_schema"],
            }
            for s in self._schemas.values()
        ]

    def get_schema(self, tool_name: str) -> dict[str, Any] | None:
        """Get full schema for a tool."""
        return self._schemas.get(tool_name)

    def validate_input(self, tool_name: str, data: dict[str, Any]) -> list[str]:
        """Validate input data against tool schema. Returns list of errors."""
        validator = self._input_validators.get(tool_name)
        if validator is None:
            return [f"Unknown tool: {tool_name}"]
        errors = []
        for error in validator.iter_errors(data):
            errors.append(f"{error.json_path}: {error.message}")
        return errors

    def validate_output(self, tool_name: str, data: dict[str, Any]) -> list[str]:
        """Validate output data against tool schema. Returns list of errors."""
        validator = self._output_validators.get(tool_name)
        if validator is None:
            return [f"Unknown tool: {tool_name}"]
        errors = []
        for error in validator.iter_errors(data):
            errors.append(f"{error.json_path}: {error.message}")
        return errors

    def get_preconditions(self, tool_name: str) -> list[str]:
        """Get preconditions for a tool."""
        schema = self._schemas.get(tool_name)
        return schema["preconditions"] if schema else []

    def get_postconditions(self, tool_name: str) -> list[str]:
        """Get postconditions for a tool."""
        schema = self._schemas.get(tool_name)
        return schema["postconditions"] if schema else []

    def register_tool(self, schema: dict[str, Any]) -> None:
        """Register a new tool schema dynamically."""
        name = schema["name"]
        self._schemas[name] = copy.deepcopy(schema)
        self._input_validators[name] = Draft7Validator(schema["input_schema"])
        self._output_validators[name] = Draft7Validator(schema["output_schema"])

.PHONY: install test lint run report notebook clean mcp

# Use the pre-existing qiskit-env which has qiskit + jupyter pre-installed
PYTHON    := python3
QENV      := /home/leonardomaximinobernardo/qiskit-env
QPYTHON   := $(QENV)/bin/python
QPIP      := $(QENV)/bin/pip
PYTEST    := $(QENV)/bin/pytest
RUFF      := $(QENV)/bin/ruff
SRC       := $(shell pwd)/src
TESTS     := $(shell pwd)/tests

# PYTHONPATH ensures our src/ is importable without pip install -e
export PYTHONPATH := $(SRC)

## ── Environment ─────────────────────────────────────────────────────────────
install:
	$(QPIP) install -e ".[dev,docs]" -q
	@echo "✅ QAC installed into qiskit-env. Ready to use."

## ── Testing (TDD) ───────────────────────────────────────────────────────────
test:
	PYTHONPATH=$(SRC) $(PYTEST) $(TESTS)/unit/ -v --tb=short

test-unit:
	PYTHONPATH=$(SRC) $(PYTEST) $(TESTS)/unit/ -v --tb=short

test-integration:
	PYTHONPATH=$(SRC) $(PYTEST) $(TESTS)/integration/ -v -s

test-all:
	PYTHONPATH=$(SRC) $(PYTEST) $(TESTS)/ -v --tb=short

## ── Code Quality ────────────────────────────────────────────────────────────
lint:
	$(RUFF) check src/ tests/ scripts/
	$(RUFF) format --check src/ tests/ scripts/

format:
	$(RUFF) format src/ tests/ scripts/
	$(RUFF) check --fix src/ tests/ scripts/

## ── Run Experiments ─────────────────────────────────────────────────────────
run:
	PYTHONPATH=$(SRC) $(QPYTHON) scripts/run_experiment.py --config configs/default.json

run-fast:
	PYTHONPATH=$(SRC) $(QPYTHON) scripts/run_experiment.py --config configs/fast.json

## ── Notebook ────────────────────────────────────────────────────────────────
notebook:
	PYTHONPATH=$(SRC) $(QENV)/bin/jupyter lab notebooks/QAC_Bloco3_Experiment.ipynb

## ── MCP Server ──────────────────────────────────────────────────────────────
mcp:
	PYTHONPATH=$(SRC) $(QPYTHON) mcp_server/server.py

## ── Cleanup ─────────────────────────────────────────────────────────────────
clean:
	rm -rf outputs/*.csv outputs/*.json outputs/*.md outputs/*.png
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

clean-all: clean
	rm -rf .mypy_cache .ruff_cache .pytest_cache

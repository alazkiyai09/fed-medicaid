.PHONY: partition train-central train-federated run-e1 run-e2 run-e3 run-e4 run-e5 run-all test lint clean

PYTHON ?= python

# ── Data Pipeline ──────────────────────────────────────────────────
partition:
	$(PYTHON) -m src.data.partition

# ── Experiments ───────────────────────────────────────────────────
run-e1:
	$(PYTHON) experiments/run_experiment_1.py

run-e2:
	$(PYTHON) experiments/run_experiment_2.py

run-e3:
	$(PYTHON) experiments/run_experiment_3.py

run-e4:
	$(PYTHON) experiments/run_experiment_4.py

run-e5:
	$(PYTHON) experiments/run_experiment_5.py

run-all: run-e1 run-e2 run-e3 run-e4 run-e5

# ── Quality ───────────────────────────────────────────────────────
test:
	$(PYTHON) -m pytest tests/ -v --tb=short

lint:
	$(PYTHON) -m ruff check src/ tests/

# ── Cleanup ───────────────────────────────────────────────────────
clean:
	rm -rf data/partitioned/* data/splits/* results/*
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

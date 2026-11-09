# =============================================================================
# Makefile – medimodal DataOps pipeline
# =============================================================================
# Usage:  make <target>
#   make all         Full setup: Docker + install + pipeline
#   make pipeline    Run the DVC pipeline
#   make ci          Lint + test + pipeline (sample mode)
# =============================================================================

.DEFAULT_GOAL := help
SHELL := /bin/bash
PYTHON := python
PYTEST := python -m pytest
PARAMS := configs/params.yaml

# ── Primary targets ──────────────────────────────────────────────────────────

.PHONY: all
all: docker-up install pipeline  ## Full setup: start services, install deps, run pipeline

.PHONY: pipeline
pipeline:  ## Run the full DVC pipeline
	dvc repro

.PHONY: install
install:  ## Install Python dependencies
	pip install -r requirements.txt

# ── Code quality ─────────────────────────────────────────────────────────────

.PHONY: lint
lint:  ## Run flake8 and black --check
	flake8 src/ flows/ tests/ --max-line-length 120 --ignore E501,W503
	black --check --line-length 120 src/ flows/ tests/

.PHONY: format
format:  ## Auto-format code with black
	black --line-length 120 src/ flows/ tests/

# ── Testing ──────────────────────────────────────────────────────────────────

.PHONY: test
test:  ## Run the full test suite
	$(PYTEST) tests/ -v --tb=short

.PHONY: test-data
test-data:  ## Run data quality tests only
	$(PYTEST) tests/test_data_quality.py -v --tb=short

# ── CI pipeline ──────────────────────────────────────────────────────────────

.PHONY: ci
ci: lint test  ## CI: lint + test + sample pipeline
	dvc repro

# ── Docker ───────────────────────────────────────────────────────────────────

.PHONY: docker-up
docker-up:  ## Start PostgreSQL + MinIO via docker-compose
	docker-compose up -d

.PHONY: docker-down
docker-down:  ## Stop Docker services
	docker-compose down

# ── Reports & UI ─────────────────────────────────────────────────────────────

.PHONY: report
report:  ## Generate quality report (post-checks)
	$(PYTHON) run.py check

.PHONY: ui
ui:  ## Launch the DataOps Control Center Web UI
	$(PYTHON) app.py

# ── Housekeeping ─────────────────────────────────────────────────────────────

.PHONY: clean
clean:  ## Remove staging data and Python caches
	rm -rf data/staging
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov .coverage

.PHONY: clean-all
clean-all: clean  ## Remove staging + DVC cache + mlruns
	rm -rf .dvc/cache .dvc/tmp mlruns

# ── Help ─────────────────────────────────────────────────────────────────────

.PHONY: help
help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

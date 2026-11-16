PY ?= py -3.10
VENV := .venv
BIN := $(VENV)/Scripts

.PHONY: help venv install lint format typecheck test test-cov run migrate clean

help:
	@echo "venv      - create the virtual environment"
	@echo "install   - install the package with dev extras"
	@echo "lint      - run ruff checks"
	@echo "format    - apply ruff formatting and import sorting"
	@echo "typecheck - run mypy"
	@echo "test      - run the test suite"
	@echo "run       - start the API with reload enabled"
	@echo "migrate   - apply database migrations"
	@echo "clean     - remove caches and build artefacts"

venv:
	$(PY) -m venv $(VENV)

install: venv
	$(BIN)/python -m pip install --upgrade pip
	$(BIN)/python -m pip install -e ".[dev]"

lint:
	$(BIN)/ruff check src tests

format:
	$(BIN)/ruff check --fix src tests
	$(BIN)/ruff format src tests

typecheck:
	$(BIN)/mypy src

test:
	$(BIN)/pytest

run:
	$(BIN)/uvicorn itinera.main:app --reload --host 127.0.0.1 --port 8000

migrate:
	$(BIN)/alembic upgrade head

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage dist build
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

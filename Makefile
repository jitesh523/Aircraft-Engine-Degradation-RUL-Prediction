# Makefile for Aircraft Engine RUL Prediction

.PHONY: help setup test test-cov test-quick lint format clean docker-build docker-run docker-up docker-down run-api run-dashboard security-scan check validate-data docs

# ─── Help (default) ───────────────────────────────────────────
help:
	@echo "╔══════════════════════════════════════════════════╗"
	@echo "║  Aircraft Engine RUL Prediction — Makefile       ║"
	@echo "╠══════════════════════════════════════════════════╣"
	@echo "║  setup            Install all dependencies       ║"
	@echo "║  run-api          Start FastAPI server            ║"
	@echo "║  run-dashboard    Launch Streamlit dashboard      ║"
	@echo "║  test             Run all tests (verbose)         ║"
	@echo "║  test-quick       Run tests (fail-fast, quiet)    ║"
	@echo "║  test-cov         Run tests with coverage report  ║"
	@echo "║  lint             Check code style                ║"
	@echo "║  format           Auto-format with Black          ║"
	@echo "║  security-scan    Run Bandit security scan        ║"
	@echo "║  check            Lint + test combined            ║"
	@echo "║  clean            Remove generated files          ║"
	@echo "║  docker-build     Build Docker image              ║"
	@echo "║  docker-run       Run Docker container            ║"
	@echo "║  docker-up        Start with docker-compose       ║"
	@echo "║  docker-down      Stop docker-compose             ║"
	@echo "║  validate-data    Run data validation checks      ║"
	@echo "╚══════════════════════════════════════════════════╝"

# ─── Setup ────────────────────────────────────────────────────
setup:
	python3 -m pip install --upgrade pip
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

# ─── Running ──────────────────────────────────────────────────
run-api:
	uvicorn api:app --reload --host 0.0.0.0 --port 8000

run-dashboard:
	streamlit run dashboard.py

# ─── Testing ──────────────────────────────────────────────────
test:
	pytest tests/ -v --timeout=120

test-cov:
	pytest tests/ -v --timeout=120 --cov=. --cov-report=term-missing --cov-report=html

test-quick:
	pytest tests/ -x -q --timeout=60

# ─── Code Quality ────────────────────────────────────────────
lint:
	flake8 . --count --statistics --exclude=venv --max-line-length=127
	black --check --exclude=venv .
	pylint --recursive=y --ignore=venv . || true

format:
	black . --exclude=venv

security-scan:
	bandit -r . -x ./venv,./tests -ll
	@echo "Security scan complete."

check: lint test
	@echo "All checks passed."

# ─── Cleanup ─────────────────────────────────────────────────
clean:
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf htmlcov
	rm -rf *.egg-info
	rm -rf dist build
	find . -type d -name "__pycache__" -exec rm -rf {} +
	@echo "Cleaned all generated files."

# ─── Docker ──────────────────────────────────────────────────
docker-build:
	docker build -t aircraft-engine-rul .

docker-run:
	docker run -p 8000:8000 aircraft-engine-rul

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

# ─── Validation ──────────────────────────────────────────────
validate-data:
	python data_validator.py

# ─── Documentation ───────────────────────────────────────────
docs:
	@echo "Documentation generation not implemented yet. See README.md"

# Makefile for Aircraft Engine RUL Prediction

.PHONY: setup test test-cov test-quick lint format clean docker-build docker-run run-api run-dashboard security-scan check docs

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
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +

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

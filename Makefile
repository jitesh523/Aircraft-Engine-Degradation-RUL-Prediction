# Makefile for Aircraft Engine RUL Prediction

.PHONY: setup test lint format clean docker-build docker-run docs

# Python Setup
setup:
	python3 -m pip install --upgrade pip
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=. --cov-report=term-missing --cov-report=html

# Code Quality
lint:
	flake8 . --count --statistics --exclude=venv --max-line-length=127
	black --check --exclude=venv .
	pylint --recursive=y --ignore=venv . || true

format:
	black . --exclude=venv

# Cleanup
clean:
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +

# Docker
docker-build:
	docker build -t aircraft-engine-rul .

docker-run:
	docker run -p 8000:8000 aircraft-engine-rul

# Validation
validate-data:
	python data_validator.py

# Documentation
docs:
	@echo "Documentation generation not implemented yet. See README.md"

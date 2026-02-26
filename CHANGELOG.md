# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [2.0.1] — 2026-02-26

### Added
- **CITATION.cff**: Academic citation file — GitHub shows "Cite this repository" button
- **LICENSE**: MIT License file (was referenced in README but missing)
- **pyproject.toml**: Centralized tool config (black, ruff, mypy, pytest, coverage)
- **setup.cfg**: flake8 per-file ignores and isort black-compatible profile
- **.gitattributes**: Consistent LF line endings, binary file markers, Python-aware diffs
- **.dockerignore**: Leaner Docker builds — excludes tests, caches, data, IDE files
- **.github/FUNDING.yml**: GitHub Sponsors button
- **.github/ISSUE_TEMPLATE/bug_report.md**: Structured bug report template
- **.github/ISSUE_TEMPLATE/feature_request.md**: Structured feature request template
- **.github/PULL_REQUEST_TEMPLATE.md**: Structured PR template with testing checklist
- **.github/CODEOWNERS**: Auto-assign PR reviewers by area (core ML, API, infra, tests)
- **.github/labeler.yml**: PR label config mapping file patterns to labels
- **.github/dependabot.yml**: Automated weekly dependency updates (pip + Actions)
- **.github/workflows/stale.yml**: Auto-close stale issues/PRs after 60 days
- **.github/workflows/labeler.yml**: Auto-label PRs based on changed file patterns
- **Makefile**: Default `make help`, `train`, `predict`, `lint-fix`, and `docs` targets
- **README.md**: Quick Start section, table of contents, pre-commit and version badges
- **conftest.py**: Pytest markers, `small_fleet`, `tmp_model_dir`, `mock_config`, `ALL_FEATURE_COLUMNS`
- **.pre-commit-config.yaml**: Ruff pre-commit hook with auto-fix

### Changed
- **.gitignore**: Added patterns for `.mypy_cache/`, `.ruff_cache/`, `build/`, `dist/`, `*.egg-info/`
- **Makefile**: Clean target now removes `.mypy_cache`, `.ruff_cache`, `dist`, `build`, `coverage.xml`
- **requirements.txt**: Organized into 10 logical sections with comments
- **requirements-dev.txt**: Organized into sections; added `ruff>=0.3.0`
- **docker-compose.yml**: Added resource limits (2 CPU / 2GB) and JSON log rotation
- **Dockerfile**: Added OCI labels, non-root `appuser` for security
- **.editorconfig**: Added rules for Dockerfile, .cfg, .cff, and shell scripts
- **pyproject.toml**: Added ruff isort config with known-first-party and per-file-ignores
- **config.py**: Expanded module docstring with env var reference and usage example
- **CITATION.cff**: Bumped version to 2.0.1, updated release date
- **SECURITY.md**: Added v2.0.1 version entry and 2 new best practices
- **CODE_OF_CONDUCT.md**: Updated to Contributor Covenant v2.1
- **CONTRIBUTING.md**: Added `make help` tip, ruff instructions, and First-Time Contributors section
- **MODEL_CARD.md**: Bumped to v2.0.1, added license field and reproducibility section
- **DASHBOARD.md**: Updated to v2.0.1, added env vars table and usage tips
- **API.md**: Added quick curl cheat sheet; bumped version to 2.0.1
- **MLFLOW_GUIDE.md**: Added `make setup` reference, fixed Docker Python version, added tip

## [2.0.0] — 2026-02-18


### Added
- **Phase 10**: Digital Twin simulation, Fleet Risk Monte Carlo, Report Engine
- **Phase 11**: Envelope Analyzer, Engine Similarity Finder, Cost Optimizer
- **Dashboard**: Expanded from 15 to 21 tabs
- **API**: Phase 11 endpoints (`/analyze/similarity`, `/optimize/cost`)
- **API**: Root redirect `GET /` → `/docs`
- **Tests**: 47+ new unit tests across 4 test files
  - `tests/test_phase11_modules.py` — DegradationClusterer, SimilarityFinder, CostOptimizer, EnvelopeAnalyzer
  - `tests/test_remaining_modules.py` — SensorNetwork, DigitalTwin, WhatIfSimulator, ReportEngine, MaintenanceAssistant
  - `tests/test_data_validator_survival.py` — DataValidator, SensorAnomalyDetector, SchemaValidator, SurvivalAnalyzer
- **Infrastructure**: Shared `conftest.py` with reusable test fixtures
- **Dashboard**: Version/status footer in sidebar
- **Dockerfile**: Added `curl` for healthcheck
- **Makefile**: New targets — `run-api`, `run-dashboard`, `test-quick`, `security-scan`, `check`
- **CHANGELOG.md**: This file

### Changed
- `config.py`: Replaced hardcoded `/Users/neha/...` with portable `os.path` and env var fallbacks
- `docker-compose.yml`: Removed deprecated `version` key, added `CMAPSS_DATA_DIR` and `GEMINI_API_KEY` env vars
- `requirements-dev.txt`: Added `pytest-timeout>=2.2.0`
- `.gitignore`: Added `mlruns/`, `reports/`, `*.tflite`, `*.onnx`, `.env`
- `README.md`: Updated to 21 tabs, 48 modules, Phase 10–11 project structure
- `API.md`: Added Phase 11 endpoint documentation
- `SECURITY.md`: Updated to v2.0 scope with 48 modules
- `models/__init__.py`: Added TransformerModel import
- `requirements.txt`: Relaxed version pinning for better compatibility

### Fixed
- `api.py`: Duplicate `ensemble = None` declaration
- `Dockerfile`: Healthcheck used `requests` (not installed); now uses `curl`
- `config.py`: Print statements removed (ran on every import)
- `.DS_Store`: Removed from version control

## [1.0.0] — 2026-01-15

### Added
- Initial release with Phases 1–9
- LSTM, Transformer, and ensemble RUL prediction models
- 15-tab Streamlit dashboard
- FastAPI REST API with Docker support
- CI/CD pipeline with GitHub Actions
- Comprehensive data validation and monitoring

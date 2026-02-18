# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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

# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 2.0.1   | :white_check_mark: Current |
| 2.x     | :white_check_mark: |
| 1.x     | :warning: Security fixes only |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability, please follow these steps:

### How to Report

1. **DO NOT** open a public issue
2. Email security concerns to the project maintainers
3. Include detailed information about the vulnerability:
   - Type of vulnerability
   - Full path to the affected file(s)
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

### What to Expect

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Resolution Timeline**: Depends on severity (Critical: 24-48h, High: 1 week, Medium: 2 weeks, Low: 1 month)

### Scope

This security policy applies to:
- All Python source code (48 modules)
- API endpoints (`api.py`) — including Phase 11 endpoints
- Configuration files (`config.py`, environment variables)
- Docker configurations (`Dockerfile`, `docker-compose.yml`)
- CI/CD pipelines (`.github/workflows/`)
- Dashboard (`dashboard.py`) — 21-tab Streamlit interface
- LLM integration (`llm_assistant.py`) — Gemini API key handling

### Out of Scope

- Issues in dependencies (report to upstream)
- Theoretical vulnerabilities without proof-of-concept
- Social engineering attacks

## Security Best Practices

When contributing to this project:

1. **Never commit secrets** — Use environment variables (`GEMINI_API_KEY`, `CMAPSS_DATA_DIR`)
2. **Validate inputs** — Especially in API endpoints and dashboard forms
3. **Keep dependencies updated** — Run `pip install --upgrade`
4. **Use type hints** — Helps catch errors early
5. **Run security scans** — `make security-scan` or `bandit -r . -x ./venv,./tests`
6. **Avoid hardcoded paths** — Use `os.path` or env vars for portability
7. **Use pre-commit hooks** — Run `pre-commit install` to auto-check before each commit
8. **Pin Docker base images** — Use specific tags (e.g., `python:3.10-slim`) not `latest`

## Acknowledgments

We appreciate responsible disclosure and will acknowledge security researchers who help improve our project security.

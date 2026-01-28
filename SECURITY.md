# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.x     | :white_check_mark: |

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
- All Python source code
- API endpoints (`api.py`)
- Configuration files
- Docker configurations
- CI/CD pipelines

### Out of Scope

- Issues in dependencies (report to upstream)
- Theoretical vulnerabilities without proof-of-concept
- Social engineering attacks

## Security Best Practices

When contributing to this project:

1. **Never commit secrets** - Use environment variables
2. **Validate inputs** - Especially in API endpoints
3. **Keep dependencies updated** - Run `pip install --upgrade`
4. **Use type hints** - Helps catch errors early
5. **Run security scans** - `bandit -r . -x ./venv`

## Acknowledgments

We appreciate responsible disclosure and will acknowledge security researchers who help improve our project security.

# Contributing to Aircraft Engine RUL Prediction

Thank you for your interest in contributing to this project! This document provides guidelines for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md) to maintain a welcoming and inclusive community.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Aircraft-Engine-Degradation-RUL-Prediction.git
   cd Aircraft-Engine-Degradation-RUL-Prediction
   ```

3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/ORIGINAL_OWNER/Aircraft-Engine-Degradation-RUL-Prediction.git
   ```

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Virtual environment tool (venv, conda, etc.)
- Git

### Installation

1. **Create and activate virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

3. **Download the C-MAPSS dataset**:
   - Download from [NASA PCoE Website](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data)
   - Extract to `data/` directory

4. **Verify installation**:
   ```bash
   pytest tests/ -v
   ```

### Makefile Shortcuts

> **Tip:** Run `make` or `make help` to see all available targets.

```bash
make setup           # Install all dependencies
make test            # Run tests with timeout
make test-cov        # Run tests with coverage report
make test-quick      # Quick test run (fail-fast)
make lint            # Run flake8 + black + pylint
make format          # Auto-format with black
make run-api         # Start FastAPI server
make run-dashboard   # Start Streamlit dashboard
make security-scan   # Run bandit security scan
make check           # Run lint + test together
make clean           # Remove build artifacts
make docker-build    # Build Docker image
make docker-up       # Start with docker-compose
```

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:

- **Clear title** describing the issue
- **Detailed description** of the problem
- **Steps to reproduce** the bug
- **Expected vs actual behavior**
- **Environment details** (Python version, OS, etc.)
- **Error messages** or screenshots if applicable

### Suggesting Enhancements

For feature requests:

- **Describe the feature** and its use case
- **Explain why** it would be valuable
- **Provide examples** if possible
- **Consider alternatives** you've thought about

### Code Contributions

1. **Create a branch** for your work:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following our coding standards

3. **Write tests** for new functionality

4. **Run tests** to ensure everything works:
   ```bash
   pytest tests/ -v
   ```

5. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

6. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request** on GitHub

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some modifications:

- **Line length**: Maximum 127 characters
- **Indentation**: 4 spaces (no tabs)
- **Imports**: Organized using isort
- **Formatting**: Use Black for code formatting

### Code Formatting

Before committing, format and lint your code:

```bash
black . --exclude=venv
ruff check . --fix          # fast linting with auto-fix
flake8 . --exclude=venv --max-line-length=127
```

> **Note:** All tool settings are centralized in `pyproject.toml`.

### Documentation

- **Docstrings**: Use Google-style docstrings for all functions and classes
- **Comments**: Explain *why*, not *what*
- **Type hints**: Use type hints for function parameters and return values

Example:

```python
def calculate_rul(time_cycles: np.ndarray, max_cycles: int) -> np.ndarray:
    """
    Calculate Remaining Useful Life for each time step.
    
    Args:
        time_cycles: Array of current time cycle values
        max_cycles: Maximum number of cycles before failure
        
    Returns:
        Array of RUL values (decreasing from max to 0)
    """
    return max_cycles - time_cycles
```

### Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Adding or updating tests
- `refactor:` Code refactoring
- `perf:` Performance improvements
- `chore:` Maintenance tasks

Examples:
```
feat: add Monte Carlo dropout for uncertainty quantification
fix: correct RUL calculation for multiple engines
docs: update README with new feature descriptions
test: add integration tests for LSTM model
```

## Testing Guidelines

### Writing Tests

- **Unit tests**: Test individual functions and classes
- **Integration tests**: Test complete workflows
- **Coverage**: Aim for >80% code coverage
- **Assertions**: Use descriptive assertion messages

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=. --cov-report=term-missing

# Run specific test file
pytest tests/test_data_loader.py -v

# Run specific test
pytest tests/test_data_loader.py::TestDataLoader::test_initialization -v
```

### Test Structure

```python
import pytest

class TestYourModule:
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        return create_sample_data()
    
    def test_your_function(self, sample_data):
        """Test your function behavior"""
        result = your_function(sample_data)
        assert result is not None
        assert len(result) > 0
```

## Pull Request Process

### Before Submitting

- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] Commit messages follow conventions
- [ ] No merge conflicts with main branch

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
How has this been tested?

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] Tests pass locally
- [ ] New tests added
```

### Review Process

1. **Automated checks**: CI/CD pipeline runs tests
2. **Code review**: Maintainers review your code
3. **Feedback**: Address any requested changes
4. **Approval**: Once approved, your PR will be merged

### After Merge

- Delete your feature branch
- Update your local main branch:
  ```bash
  git checkout main
  git pull upstream main
  ```

## Questions?

If you have questions, feel free to:

- Open an issue for discussion
- Reach out to maintainers
- Check existing issues and PRs for similar questions

## First-Time Contributors

Welcome! Here are some great first contributions:

- 📝 Fix typos or improve documentation
- 🧪 Add tests for untested functions
- 🐛 Fix issues labeled [`good first issue`](https://github.com/jitesh523/Aircraft-Engine-Degradation-RUL-Prediction/labels/good%20first%20issue)
- 📊 Improve visualizations or add new dashboard charts
- 🔧 Add type hints to functions missing them

> **Tip:** Run `make help` to explore the project, and `make test-quick` to verify nothing is broken.

Thank you for contributing! 🚀

"""
Shared pytest fixtures for the Aircraft Engine RUL Prediction test suite.
Provides synthetic C-MAPSS-shaped data used across multiple test modules.
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── Custom Markers ──────────────────────────────────────────────────────────
def pytest_configure(config):
    """Register custom markers to avoid warnings."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks integration tests requiring external services")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU resources")


# ─── Constants ───────────────────────────────────────────────────────────────
SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]
OPERATIONAL_SETTINGS = ["setting_1", "setting_2", "setting_3"]
ALL_FEATURE_COLUMNS = OPERATIONAL_SETTINGS + SENSOR_COLUMNS


@pytest.fixture
def small_fleet():
    """Generate a small fleet (3 engines) for fast-running tests."""
    np.random.seed(99)
    rows = []
    for uid in range(1, 4):
        n_cycles = np.random.randint(50, 80)
        for t in range(1, n_cycles + 1):
            row = {'unit_id': uid, 'time_cycles': t, 'RUL': n_cycles - t}
            for s in range(1, 22):
                row[f'sensor_{s}'] = 100 + s * 10 + t * 0.01 * s + np.random.normal(0, 1)
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_fleet():
    """
    Generate synthetic C-MAPSS-shaped data with 10 engines,
    21 sensors, and RUL column.
    """
    np.random.seed(42)
    rows = []
    for uid in range(1, 11):
        n_cycles = np.random.randint(120, 200)
        for t in range(1, n_cycles + 1):
            row = {
                'unit_id': uid,
                'time_cycles': t,
                'RUL': n_cycles - t,
            }
            for s in range(1, 22):
                base = 100 + s * 10
                noise = np.random.normal(0, 1)
                trend = t * 0.01 * s
                row[f'sensor_{s}'] = base + trend + noise
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def single_engine_df(synthetic_fleet):
    """Single engine data sorted by time_cycles."""
    return synthetic_fleet[synthetic_fleet['unit_id'] == 1].sort_values('time_cycles').copy()


@pytest.fixture
def fleet_rul_df():
    """Fleet-level DataFrame with engine_id and rul_pred for cost/scheduling tests."""
    np.random.seed(42)
    return pd.DataFrame({
        'engine_id': [f'E{i}' for i in range(1, 21)],
        'rul_pred': np.concatenate([
            np.random.randint(5, 25, 5),
            np.random.randint(30, 80, 7),
            np.random.randint(80, 180, 8),
        ]).astype(float)
    })


@pytest.fixture
def tmp_model_dir(tmp_path):
    """Provide a clean temporary directory for model save/load tests."""
    model_dir = tmp_path / "models" / "saved"
    model_dir.mkdir(parents=True)
    return model_dir


@pytest.fixture
def mock_config(tmp_path):
    """Provide a lightweight config dict for tests needing config values."""
    return {
        'data_dir': str(tmp_path / "data"),
        'models_dir': str(tmp_path / "models"),
        'results_dir': str(tmp_path / "results"),
        'sequence_length': 30,
        'random_seed': 42,
        'rul_cap': 125,
        'lstm': {
            'lstm_units': [64, 32],
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 5,
            'patience': 3,
        },
    }

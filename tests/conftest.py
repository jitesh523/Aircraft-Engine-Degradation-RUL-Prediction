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

"""
Tests for UncertaintyQuantifier module
"""

import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestUncertaintyQuantifier:
    """Test suite for uncertainty_quantifier.UncertaintyQuantifier."""

    @pytest.fixture
    def uq(self):
        from uncertainty_quantifier import UncertaintyQuantifier
        return UncertaintyQuantifier(n_iterations=50, confidence_level=0.95)

    def test_initialization(self, uq):
        assert uq.n_iterations == 50
        assert uq.confidence_level == 0.95

    def test_get_prediction_intervals(self, uq):
        np.random.seed(42)
        mean = np.random.uniform(30, 120, 20)
        lower = mean - 10
        upper = mean + 10
        df = uq.get_prediction_intervals(mean, lower, upper)
        assert isinstance(df, pd.DataFrame)
        assert 'prediction' in df.columns
        assert 'lower_bound' in df.columns
        assert 'upper_bound' in df.columns
        assert len(df) == 20

    def test_classify_uncertainty(self, uq):
        widths = np.array([5, 15, 30, 50, 80])
        classes = uq.classify_uncertainty(widths)
        assert isinstance(classes, np.ndarray)
        assert len(classes) == 5

    def test_evaluate_uncertainty_calibration(self, uq):
        np.random.seed(42)
        y_true = np.random.uniform(30, 120, 50)
        lower = y_true - 15
        upper = y_true + 15
        result = uq.evaluate_uncertainty_calibration(y_true, lower, upper)
        assert isinstance(result, dict)
        assert 'coverage' in result
        # Coverage is in percentage (100.0 = 100%)
        assert result['coverage'] >= 99.0

    def test_get_high_uncertainty_engines(self, uq):
        np.random.seed(42)
        mean = np.random.uniform(30, 120, 10)
        lower = mean - np.random.uniform(5, 20, 10)
        upper = mean + np.random.uniform(5, 20, 10)
        df = uq.get_prediction_intervals(mean, lower, upper)
        unit_ids = np.arange(1, 11)
        result = uq.get_high_uncertainty_engines(df, unit_ids, top_k=3)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

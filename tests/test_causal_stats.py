"""
Tests for IVEstimator and PowerCalculator modules
"""

import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ============================================================
# IVEstimator Tests
# ============================================================
class TestIVEstimator:
    """Test suite for iv_estimator.IVEstimator."""

    @pytest.fixture
    def estimator(self):
        from iv_estimator import IVEstimator
        return IVEstimator()

    @pytest.fixture
    def iv_data(self):
        """Synthetic data with known causal effect and confounding."""
        np.random.seed(42)
        n = 500
        U = np.random.normal(0, 1, n)        # Confounder
        Z = np.random.normal(0, 1, n)        # Instrument
        X = 0.6 * Z + 0.4 * U + np.random.normal(0, 0.3, n)  # Treatment
        Y = 2.0 * X + U + np.random.normal(0, 0.5, n)         # Outcome
        return pd.DataFrame({'Y': Y, 'X': X, 'Z': Z, 'U': U})

    def test_initialization(self, estimator):
        assert estimator.model is None
        assert estimator.results is None

    def test_estimate_effect(self, estimator, iv_data):
        results = estimator.estimate_effect(
            iv_data, outcome_col='Y', treatment_col='X', instrument_col='Z'
        )
        assert isinstance(results, dict)
        assert 'effect_size' in results
        # True effect is 2.0; IV estimate should be close
        assert 1.0 < results['effect_size'] < 3.5

    def test_estimate_effect_with_controls(self, estimator, iv_data):
        iv_data['ctrl'] = np.random.normal(0, 1, len(iv_data))
        results = estimator.estimate_effect(
            iv_data, outcome_col='Y', treatment_col='X',
            instrument_col='Z', control_cols=['ctrl']
        )
        assert isinstance(results, dict)
        assert 'effect_size' in results

    def test_verify_assumptions(self, estimator, iv_data):
        result = estimator.verify_assumptions(
            iv_data, outcome_col='Y', treatment_col='X', instrument_col='Z'
        )
        assert isinstance(result, dict)
        # Instrument should be relevant (correlated with treatment)
        assert 'relevance_check' in result


# ============================================================
# PowerCalculator Tests
# ============================================================
class TestPowerCalculator:
    """Test suite for power_calculator.PowerCalculator."""

    @pytest.fixture
    def calc(self):
        from power_calculator import PowerCalculator
        return PowerCalculator()

    def test_initialization(self, calc):
        assert calc.power_analysis is not None

    def test_calculate_sample_size(self, calc):
        n = calc.calculate_sample_size(effect_size=0.5, alpha=0.05, power=0.8)
        assert isinstance(n, int)
        assert n > 0
        # Medium effect (d=0.5) needs ~64 per group
        assert 50 < n < 100

    def test_calculate_sample_size_small_effect(self, calc):
        n = calc.calculate_sample_size(effect_size=0.2, alpha=0.05, power=0.8)
        assert n > 200  # Small effect needs big sample

    def test_calculate_sample_size_zero_effect(self, calc):
        n = calc.calculate_sample_size(effect_size=0)
        assert n == float('inf')

    def test_calculate_power(self, calc):
        power = calc.calculate_power(nobs1=100, effect_size=0.5)
        assert isinstance(power, float)
        assert 0.0 < power <= 1.0
        assert power > 0.8  # Should be well powered

    def test_calculate_mde(self, calc):
        mde = calc.calculate_minimum_detectable_effect(nobs1=100, power=0.8)
        assert isinstance(mde, float)
        assert mde > 0

    def test_estimate_cohens_d(self, calc):
        np.random.seed(42)
        g1 = np.random.normal(0, 1, 100)
        g2 = np.random.normal(0.5, 1, 100)
        d = calc.estimate_cohens_d(g1, g2)
        assert isinstance(d, float)
        assert d > 0
        # True d ≈ 0.5
        assert 0.2 < d < 1.0

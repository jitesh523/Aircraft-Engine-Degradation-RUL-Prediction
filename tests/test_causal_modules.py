
"""
Unit tests for Causal Analytics Modules (Phase 5):
- Instrumental Variables (IV) Estimator
- Power Calculator
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from iv_estimator import IVEstimator
from power_calculator import PowerCalculator

class TestIVEstimator:
    """Test suite for IVEstimator"""
    
    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic data with known causal effect"""
        np.random.seed(42)
        n = 1000
        
        # Confounder
        U = np.random.normal(0, 1, n)
        
        # Instrument (affects X, independent of U)
        Z = np.random.normal(0, 1, n)
        
        # Treatment (endogenous: depends on Z and U)
        # 0.8 is strong relevance
        X = 0.8 * Z + 0.5 * U + np.random.normal(0, 0.5, n)
        
        # Outcome
        # True effect of X on Y = 2.0
        Y = 2.0 * X + U + np.random.normal(0, 0.5, n)
        
        return pd.DataFrame({'Y': Y, 'X': X, 'Z': Z, 'U': U})
    
    def test_initialization(self):
        """Test estimator initialization"""
        estimator = IVEstimator()
        assert estimator is not None
        
    def test_estimate_effect(self, synthetic_data):
        """Test IV estimation logic"""
        df = synthetic_data
        estimator = IVEstimator()
        
        results = estimator.estimate_effect(df, outcome_col='Y', treatment_col='X', instrument_col='Z')
        
        assert results is not None
        assert 'effect_size' in results
        assert 'instrument_strength' in results
        
        # Check if estimate is close to true value (2.0)
        # 2SLS should remove bias. OLS would be biased upwards (~2.5) due to U
        assert 1.8 < results['effect_size'] < 2.2
        assert results['instrument_strength'] == 'Strong'
        
    def test_weak_instrument(self):
        """Test behavior with weak instrument"""
        np.random.seed(42)
        n = 500
        z = np.random.normal(0, 1, n)
        # Very weak relationship between Z and X
        x = 0.01 * z + np.random.normal(0, 1, n)
        y = 2 * x + np.random.normal(0, 1, n)
        
        df = pd.DataFrame({'Y': y, 'X': x, 'Z': z})
        
        estimator = IVEstimator()
        results = estimator.estimate_effect(df, 'Y', 'X', 'Z')
        
        # Should likely report Weak instrument
        assert results['instrument_strength'] == 'Weak'

class TestPowerCalculator:
    """Test suite for PowerCalculator"""
    
    def test_sample_size_calculation(self):
        """Test sample size calculation against known standard"""
        calc = PowerCalculator()
        
        # Standard scenario: d=0.5, alpha=0.05, power=0.8
        # Expected n approx 63-64 per group
        n = calc.calculate_sample_size(effect_size=0.5, alpha=0.05, power=0.8)
        
        assert 60 <= n <= 70
        
    def test_power_calculation(self):
        """Test power calculation"""
        calc = PowerCalculator()
        
        # If we have n=64, d=0.5, alpha=0.05, power should be ~0.8
        power = calc.calculate_power(nobs1=64, effect_size=0.5, alpha=0.05)
        
        assert 0.78 <= power <= 0.82
        
    def test_mde_calculation(self):
        """Test Minimum Detectable Effect calculation"""
        calc = PowerCalculator()
        
        mde = calc.calculate_minimum_detectable_effect(nobs1=64, power=0.8, alpha=0.05)
        
        # Should be close to 0.5
        assert 0.45 <= mde <= 0.55

if __name__ == "__main__":
    pytest.main([__file__, '-v'])

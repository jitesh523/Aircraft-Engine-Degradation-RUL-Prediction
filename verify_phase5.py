
import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

from iv_estimator import IVEstimator
from power_calculator import PowerCalculator

class TestPhase5Features(unittest.TestCase):
    
    def test_iv_estimator(self):
        print("\nTesting IV Estimator...")
        # Generate synthetic data with known causal effect
        np.random.seed(42)
        n = 1000
        U = np.random.normal(0, 1, n)
        Z = np.random.normal(0, 1, n)
        X = 0.8 * Z + 0.5 * U + np.random.normal(0, 0.5, n)
        Y = 2.0 * X + U + np.random.normal(0, 0.5, n)
        
        df = pd.DataFrame({'Y': Y, 'X': X, 'Z': Z, 'U': U})
        
        estimator = IVEstimator()
        results = estimator.estimate_effect(df, 'Y', 'X', 'Z')
        
        self.assertIn('effect_size', results)
        print(f"  True Effect: 2.0")
        print(f"  Estimated Effect: {results['effect_size']:.4f}")
        
        # Check if estimate is close to true value (allow some noise)
        self.assertAlmostEqual(results['effect_size'], 2.0, delta=0.2)
        self.assertEqual(results['instrument_strength'], 'Strong')

    def test_power_calculator(self):
        print("\nTesting Power Calculator...")
        calc = PowerCalculator()
        
        # Known standard: d=0.5, power=0.8, alpha=0.05 -> n=64 per group
        n = calc.calculate_sample_size(effect_size=0.5, alpha=0.05, power=0.8)
        print(f"  Sample size for d=0.5, power=0.8: {n}")
        self.assertTrue(60 <= n <= 70)
        
        # Power calculation
        power = calc.calculate_power(nobs1=64, effect_size=0.5, alpha=0.05)
        print(f"  Power for n=64, d=0.5: {power:.4f}")
        self.assertAlmostEqual(power, 0.8, delta=0.05)

    def test_dashboard_imports(self):
        print("\nTesting Dashboard Imports...")
        try:
            from dashboard import show_causal_inference, show_experiment_design, show_drift_monitoring
            print("  Dashboard functions imported successfully.")
        except ImportError as e:
            self.fail(f"Failed to import dashboard functions: {e}")

if __name__ == '__main__':
    unittest.main()

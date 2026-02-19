"""
Tests for ModelMonitor and ConceptDriftDetector modules
"""

import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestModelMonitor:
    """Test suite for model_monitor.ModelMonitor."""

    @pytest.fixture
    def monitor(self):
        from model_monitor import ModelMonitor
        return ModelMonitor(baseline_metrics={'RMSE': 20.0, 'MAE': 15.0})

    @pytest.fixture
    def baseline_data(self):
        np.random.seed(42)
        n = 200
        cols = {f'sensor_{i}': np.random.normal(100 + i, 5, n) for i in range(1, 6)}
        return pd.DataFrame(cols)

    @pytest.fixture
    def drifted_data(self, baseline_data):
        """Data with injected drift in sensor_1."""
        drifted = baseline_data.copy()
        drifted['sensor_1'] += 30  # large shift
        return drifted

    def test_initialization(self, monitor):
        assert monitor.baseline_metrics is not None
        assert monitor.baseline_metrics['RMSE'] == 20.0

    def test_calculate_psi_no_drift(self, monitor):
        np.random.seed(42)
        a = np.random.normal(100, 10, 500)
        b = np.random.normal(100, 10, 500)
        psi = monitor.calculate_psi(a, b)
        assert isinstance(psi, float)
        assert psi < 0.1  # no real drift

    def test_calculate_psi_with_drift(self, monitor):
        np.random.seed(42)
        a = np.random.normal(100, 10, 500)
        b = np.random.normal(130, 10, 500)  # big shift
        psi = monitor.calculate_psi(a, b)
        assert psi > 0.2  # significant drift

    def test_kolmogorov_smirnov_test(self, monitor):
        np.random.seed(42)
        a = np.random.normal(0, 1, 200)
        b = np.random.normal(1, 1, 200)
        drift, stat, pval = monitor.kolmogorov_smirnov_test(a, b)
        assert isinstance(drift, (bool, np.bool_))
        assert drift  # distributions are different

    def test_detect_feature_drift(self, monitor, baseline_data, drifted_data):
        cols = list(baseline_data.columns)
        result = monitor.detect_feature_drift(baseline_data, drifted_data, cols)
        assert isinstance(result, dict)

    def test_monitor_performance(self, monitor):
        np.random.seed(42)
        y_true = np.random.uniform(30, 120, 100)
        y_pred = y_true + np.random.normal(0, 15, 100)
        try:
            result = monitor.monitor_performance(y_true, y_pred)
            assert isinstance(result, dict)
        except ImportError:
            pytest.skip('Evaluator dependency unavailable')

    def test_generate_drift_report(self, monitor, baseline_data, drifted_data):
        cols = list(baseline_data.columns)
        report = monitor.generate_drift_report(baseline_data, drifted_data, feature_cols=cols)
        assert isinstance(report, dict)
        assert 'timestamp' in report


class TestConceptDriftDetector:
    """Test suite for model_monitor.ConceptDriftDetector."""

    @pytest.fixture
    def detector(self):
        from model_monitor import ConceptDriftDetector
        return ConceptDriftDetector(sensitivity='medium')

    def test_initialization(self, detector):
        assert detector is not None

    def test_detect_target_drift(self, detector):
        np.random.seed(42)
        baseline_rul = np.random.uniform(50, 150, 200)
        current_rul = np.random.uniform(20, 100, 200)  # shifted
        result = detector.detect_target_drift(baseline_rul, current_rul)
        assert isinstance(result, dict)
        assert 'drift_detected' in result

    def test_detect_covariate_shift(self, detector):
        np.random.seed(42)
        cols = [f's{i}' for i in range(5)]
        base = pd.DataFrame(np.random.normal(0, 1, (200, 5)), columns=cols)
        curr = pd.DataFrame(np.random.normal(1, 1, (200, 5)), columns=cols)
        result = detector.detect_covariate_shift(base, curr, cols)
        assert isinstance(result, dict)

    def test_calculate_drift_severity(self, detector):
        np.random.seed(42)
        target = detector.detect_target_drift(
            np.random.uniform(50, 150, 200),
            np.random.uniform(20, 100, 200)
        )
        cols = ['s0']
        base = pd.DataFrame({'s0': np.random.normal(0, 1, 200)})
        curr = pd.DataFrame({'s0': np.random.normal(2, 1, 200)})
        covar = detector.detect_covariate_shift(base, curr, cols)
        severity = detector.calculate_drift_severity(target, covar)
        assert isinstance(severity, dict)
        assert 'severity_level' in severity

"""
Tests for ModelComparisonReport module
"""

import sys
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestModelComparisonReport:
    """Test suite for model_comparison.ModelComparisonReport."""

    @pytest.fixture
    def report(self):
        from model_comparison import ModelComparisonReport
        return ModelComparisonReport()

    @pytest.fixture
    def populated_report(self, report):
        np.random.seed(42)
        y_true = np.random.uniform(20, 120, 100)
        y_pred_a = y_true + np.random.normal(0, 10, 100)
        y_pred_b = y_true + np.random.normal(5, 15, 100)
        report.add_model_results('ModelA', y_true, y_pred_a, training_time=10.0)
        report.add_model_results('ModelB', y_true, y_pred_b, training_time=25.0)
        return report

    def test_initialization(self, report):
        assert report.comparison_results == {}

    def test_add_model_results(self, populated_report):
        assert 'ModelA' in populated_report.comparison_results
        assert 'ModelB' in populated_report.comparison_results

    def test_get_comparison_dataframe(self, populated_report):
        df = populated_report.get_comparison_dataframe()
        assert len(df) == 2
        assert 'RMSE' in df.columns or 'rmse' in df.columns

    def test_statistical_comparison(self, populated_report):
        result = populated_report.statistical_comparison('ModelA', 'ModelB', n_bootstrap=100)
        assert isinstance(result, dict)

    def test_find_best_model(self, populated_report):
        best = populated_report.find_best_model(metric='rmse')
        assert isinstance(best, dict)
        assert 'model_name' in best or 'best_model' in best

    def test_generate_error_analysis(self, populated_report):
        analysis = populated_report.generate_error_analysis('ModelA')
        assert isinstance(analysis, dict)

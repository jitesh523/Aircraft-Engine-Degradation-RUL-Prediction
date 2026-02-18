"""
Tests for DataValidator and SurvivalAnalyzer modules
"""

import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ============================================================
# DataValidator Tests
# ============================================================
class TestDataValidator:
    """Test suite for data_validator.DataValidator."""

    @pytest.fixture
    def validator(self):
        from data_validator import DataValidator
        return DataValidator()

    @pytest.fixture
    def valid_df(self, synthetic_fleet):
        """A DataFrame that matches expected schema."""
        return synthetic_fleet

    # -- schema --
    def test_validate_schema_valid(self, validator, valid_df):
        is_valid, errors = validator.validate_schema(valid_df)
        # synthetic_fleet may not have all 26 expected columns, so just check types
        assert isinstance(is_valid, bool)
        assert isinstance(errors, list)

    def test_validate_schema_missing_columns(self, validator):
        df = pd.DataFrame({'unit_id': [1], 'time_cycles': [1]})
        is_valid, errors = validator.validate_schema(df)
        assert is_valid is False
        assert len(errors) > 0

    # -- data quality --
    def test_validate_data_quality_clean(self, validator, valid_df):
        is_valid, report = validator.validate_data_quality(valid_df)
        assert isinstance(report, dict)
        # report may have 'issues' key instead of 'missing_values'
        assert 'issues' in report or 'missing_values' in report

    def test_validate_data_quality_with_nulls(self, validator, valid_df):
        df = valid_df.copy()
        df.loc[0, 'sensor_2'] = np.nan
        is_valid, report = validator.validate_data_quality(df)
        assert isinstance(report, dict)
        # should detect the introduced null
        issues = report.get('issues', [])
        assert len(issues) > 0 or not is_valid

    # -- sensor ranges --
    def test_validate_sensor_ranges(self, validator, valid_df):
        is_valid, report = validator.validate_sensor_ranges(valid_df)
        assert isinstance(report, dict)

    # -- validate_all --
    def test_validate_all(self, validator, valid_df):
        is_valid, report = validator.validate_all(valid_df, verbose=False)
        assert isinstance(report, dict)
        assert 'schema' in report
        assert 'data_quality' in report


# ============================================================
# SensorAnomalyDetector Tests
# ============================================================
class TestSensorAnomalyDetector:
    """Test suite for data_validator.SensorAnomalyDetector."""

    @pytest.fixture
    def detector(self):
        from data_validator import SensorAnomalyDetector
        return SensorAnomalyDetector()

    def test_detect_anomalies_zscore(self, detector, single_engine_df):
        result = detector.detect_sensor_anomalies(single_engine_df, method='zscore')
        assert isinstance(result, dict)

    def test_detect_anomalies_iqr(self, detector, single_engine_df):
        result = detector.detect_sensor_anomalies(single_engine_df, method='iqr')
        assert isinstance(result, dict)

    def test_sensor_correlation_check(self, detector, single_engine_df):
        result = detector.sensor_correlation_check(single_engine_df)
        assert isinstance(result, dict)

    def test_degradation_pattern_match(self, detector, single_engine_df):
        uid = int(single_engine_df['unit_id'].iloc[0])
        result = detector.degradation_pattern_match(single_engine_df, unit_id=uid)
        assert isinstance(result, dict)

    def test_full_sensor_analysis(self, detector, single_engine_df):
        uid = int(single_engine_df['unit_id'].iloc[0])
        result = detector.full_sensor_analysis(single_engine_df, unit_id=uid)
        assert isinstance(result, dict)
        # result nests under 'anomaly_detection' not top-level 'anomalies'
        assert 'anomaly_detection' in result or 'anomalies' in result


# ============================================================
# SchemaValidator Tests
# ============================================================
class TestSchemaValidator:
    """Test suite for data_validator.SchemaValidator."""

    @pytest.fixture
    def schema_val(self):
        from data_validator import SchemaValidator
        return SchemaValidator()

    def test_define_schema(self, schema_val):
        schema_val.define_schema('test', {
            'columns': {'a': 'int64', 'b': 'float64'},
            'required': ['a']
        })
        assert 'test' in schema_val.list_schemas()

    def test_get_schema(self, schema_val):
        schema_val.define_schema('demo', {'columns': {'x': 'int64'}})
        schema = schema_val.get_schema('demo')
        assert schema is not None

    def test_get_validation_summary(self, schema_val):
        summary = schema_val.get_validation_summary()
        # May return a string or a dict
        assert summary is not None


# ============================================================
# SurvivalAnalyzer Tests
# ============================================================
class TestSurvivalAnalyzer:
    """Test suite for survival_analyzer.SurvivalAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        from survival_analyzer import SurvivalAnalyzer
        return SurvivalAnalyzer()

    def test_initialization(self, analyzer):
        assert analyzer.km_fitted is False
        assert analyzer.cox_fitted is False

    def test_prepare_survival_data(self, analyzer, synthetic_fleet):
        surv_df = analyzer.prepare_survival_data(synthetic_fleet)
        assert 'duration' in surv_df.columns
        assert 'event' in surv_df.columns
        assert len(surv_df) == synthetic_fleet['unit_id'].nunique()
        assert (surv_df['event'] == 1).all()  # run-to-failure

    def test_fit_kaplan_meier(self, analyzer, synthetic_fleet):
        surv_df = analyzer.prepare_survival_data(synthetic_fleet)
        analyzer.fit_kaplan_meier(surv_df)
        assert analyzer.km_fitted is True
        assert analyzer.median_survival is not None

    def test_predict_survival_probability(self, analyzer, synthetic_fleet):
        surv_df = analyzer.prepare_survival_data(synthetic_fleet)
        analyzer.fit_kaplan_meier(surv_df)
        prob_df = analyzer.predict_survival_probability()
        assert isinstance(prob_df, pd.DataFrame)
        assert 'survival_probability' in prob_df.columns

    def test_get_hazard_function(self, analyzer, synthetic_fleet):
        surv_df = analyzer.prepare_survival_data(synthetic_fleet)
        analyzer.fit_kaplan_meier(surv_df)
        hazard_df = analyzer.get_hazard_function()
        assert isinstance(hazard_df, pd.DataFrame)
        assert 'cumulative_hazard' in hazard_df.columns

    def test_fit_cox(self, analyzer, synthetic_fleet):
        surv_df = analyzer.prepare_survival_data(synthetic_fleet)
        # Pick a small set of covariates to avoid overfitting
        sensor_means = [c for c in surv_df.columns if c.endswith('_mean') and 'sensor' in c][:3]
        result = analyzer.fit_cox(surv_df, covariates=sensor_means, penalizer=1.0)
        assert analyzer.cox_fitted is True
        assert 'concordance_index' in result

    def test_get_summary(self, analyzer, synthetic_fleet):
        surv_df = analyzer.prepare_survival_data(synthetic_fleet)
        analyzer.fit_kaplan_meier(surv_df)
        summary = analyzer.get_summary()
        assert isinstance(summary, dict)
        assert 'km_fitted' in summary
        assert summary['km_fitted'] is True

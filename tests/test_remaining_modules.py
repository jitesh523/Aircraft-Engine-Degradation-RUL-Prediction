"""
Unit tests for remaining untested modules:
- SensorNetwork
- DigitalTwin
- WhatIfSimulator
- ReportEngine
- MaintenanceAssistant (LLM fallback only — no API key needed)
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sensor_network import SensorNetwork
from digital_twin import DigitalTwin, HPC_DEGRADATION
from whatif_simulator import WhatIfSimulator
from report_engine import ReportEngine
from llm_assistant import MaintenanceAssistant

# synthetic_fleet, single_engine_df, fleet_rul_df from conftest.py


# ============================================================
# SensorNetwork tests
# ============================================================

class TestSensorNetwork:
    """Test suite for SensorNetwork."""

    def test_initialization(self):
        net = SensorNetwork(corr_threshold=0.5)
        assert net.corr_threshold == 0.5

    def test_build_network(self, synthetic_fleet):
        net = SensorNetwork(corr_threshold=0.3)
        stats = net.build_network(synthetic_fleet)
        assert isinstance(stats, dict)
        assert 'n_nodes' in stats
        assert 'n_edges' in stats
        assert stats['n_nodes'] > 0

    def test_communities_detected(self, synthetic_fleet):
        net = SensorNetwork(corr_threshold=0.3)
        net.build_network(synthetic_fleet)
        assert len(net.communities) > 0

    def test_propagation_paths(self, synthetic_fleet):
        net = SensorNetwork(corr_threshold=0.3)
        net.build_network(synthetic_fleet)
        # Pick a sensor that exists in the network
        if net.adjacency:
            source = list(net.adjacency.keys())[0]
            paths = net.get_propagation_paths(source, max_hops=2)
            assert isinstance(paths, list)

    def test_degradation_correlation(self, synthetic_fleet):
        net = SensorNetwork(corr_threshold=0.3)
        net.build_network(synthetic_fleet)
        shift_df = net.analyze_degradation_correlation(synthetic_fleet)
        assert isinstance(shift_df, pd.DataFrame)
        assert len(shift_df) > 0


# ============================================================
# DigitalTwin tests
# ============================================================

class TestDigitalTwin:
    """Test suite for DigitalTwin."""

    def test_initialization(self):
        twin = DigitalTwin(engine_id='TEST-001', total_life=150)
        assert twin.engine_id == 'TEST-001'
        assert twin.total_life == 150

    def test_simulate(self):
        twin = DigitalTwin(total_life=100, degradation=HPC_DEGRADATION)
        sim_df = twin.simulate(n_cycles=100)
        assert isinstance(sim_df, pd.DataFrame)
        assert len(sim_df) == 100
        assert 'time_cycles' in sim_df.columns
        # Should have sensor columns
        sensor_cols = [c for c in sim_df.columns if c.startswith('sensor_')]
        assert len(sensor_cols) > 0

    def test_simulate_generates_rul(self):
        twin = DigitalTwin(total_life=80)
        sim_df = twin.simulate()
        assert 'RUL' in sim_df.columns
        # RUL should decrease over time
        assert sim_df['RUL'].iloc[0] > sim_df['RUL'].iloc[-1]

    def test_project_remaining_life(self):
        twin = DigitalTwin(total_life=150, degradation=HPC_DEGRADATION)
        twin.simulate()
        projection = twin.project_remaining_life(current_cycle=50, n_simulations=100)
        assert isinstance(projection, dict)
        assert 'mean_rul' in projection
        assert 'median_rul' in projection
        assert 'ci_lower' in projection
        assert 'ci_upper' in projection
        assert projection['mean_rul'] > 0

    def test_generate_fleet(self):
        twin = DigitalTwin(degradation=HPC_DEGRADATION)
        fleet_df = twin.generate_fleet(n_engines=5, life_range=(100, 150))
        assert isinstance(fleet_df, pd.DataFrame)
        # Should have 5 engines
        assert fleet_df['unit_id'].nunique() == 5

    def test_compare_with_real(self, single_engine_df):
        twin = DigitalTwin(total_life=len(single_engine_df), degradation=HPC_DEGRADATION)
        twin.simulate()
        comparison = twin.compare_with_real(single_engine_df)
        assert isinstance(comparison, dict)
        assert 'per_sensor' in comparison


# ============================================================
# WhatIfSimulator tests
# ============================================================

class TestWhatIfSimulator:
    """Test suite for WhatIfSimulator."""

    def test_initialization(self):
        sim = WhatIfSimulator()
        assert sim is not None

    def test_learn_degradation_rates(self, synthetic_fleet):
        sim = WhatIfSimulator()
        sim.learn_degradation_rates(synthetic_fleet)
        assert sim.degradation_models is not None
        assert len(sim.degradation_models) > 0

    def test_delayed_maintenance(self):
        sim = WhatIfSimulator()
        scenario = sim.simulate_delayed_maintenance(
            current_rul=80.0,
            delay_cycles=20,
            degradation_rate=-1.0
        )
        assert isinstance(scenario, dict)
        assert 'final_rul' in scenario
        assert scenario['final_rul'] < 80.0

    def test_sensor_drift(self, single_engine_df):
        sim = WhatIfSimulator()
        sim.learn_degradation_rates(
            pd.concat([single_engine_df] * 2)  # needs multiple engines
        )
        scenario = sim.simulate_sensor_drift(
            engine_data=single_engine_df,
            sensor='sensor_1',
            drift_multiplier=2.0,
            n_cycles=30
        )
        assert isinstance(scenario, dict)

    def test_fleet_scenario(self):
        sim = WhatIfSimulator()
        fleet_ruls = np.array([10, 30, 50, 80, 120, 150], dtype=float)
        for strategy in ['proactive', 'reactive', 'minimal']:
            scenario = sim.simulate_fleet_scenario(
                fleet_ruls=fleet_ruls,
                strategy=strategy,
                horizon=50
            )
            assert isinstance(scenario, dict)
            assert 'total_cost' in scenario
            assert 'total_failures' in scenario

    def test_compare_scenarios(self):
        sim = WhatIfSimulator()
        fleet_ruls = np.array([15, 40, 60, 100], dtype=float)
        for strategy in ['proactive', 'reactive']:
            sim.simulate_fleet_scenario(fleet_ruls, strategy=strategy, horizon=50)
        comparison = sim.compare_scenarios()
        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) >= 2


# ============================================================
# ReportEngine tests
# ============================================================

class TestReportEngine:
    """Test suite for ReportEngine."""

    def test_initialization(self):
        engine = ReportEngine()
        assert engine is not None

    def test_analyze_fleet(self, fleet_rul_df):
        engine = ReportEngine()
        analysis = engine.analyze_fleet(fleet_rul_df, rul_col='rul_pred')
        assert isinstance(analysis, dict)
        assert 'n_engines' in analysis
        assert analysis['n_engines'] == len(fleet_rul_df)

    def test_generate_recommendations(self, fleet_rul_df):
        engine = ReportEngine()
        engine.analyze_fleet(fleet_rul_df, rul_col='rul_pred')
        recs = engine._generate_recommendations()
        assert isinstance(recs, list)
        assert len(recs) > 0

    def test_summary_text(self, fleet_rul_df):
        engine = ReportEngine()
        engine.analyze_fleet(fleet_rul_df, rul_col='rul_pred')
        summary = engine.generate_summary_text()
        assert isinstance(summary, str)
        assert len(summary) > 50

    def test_html_report(self, fleet_rul_df, tmp_path):
        engine = ReportEngine()
        engine.analyze_fleet(fleet_rul_df, rul_col='rul_pred')
        report_path = engine.generate_html_report(
            title="Test Report",
            filename=str(tmp_path / "test_report.html")
        )
        assert report_path is not None
        assert Path(report_path).exists()
        content = Path(report_path).read_text()
        assert 'Test Report' in content


# ============================================================
# MaintenanceAssistant tests (no API key — fallback mode)
# ============================================================

class TestMaintenanceAssistant:
    """Test suite for MaintenanceAssistant (offline/fallback mode)."""

    def test_initialization_no_key(self):
        """Should initialize without API key (fallback mode)."""
        assistant = MaintenanceAssistant(api_key=None)
        assert assistant is not None

    def test_build_fleet_context(self):
        """Should build context string from fleet data."""
        assistant = MaintenanceAssistant(api_key=None)
        sample_df = pd.DataFrame({
            'unit_id': [1, 2, 3, 4, 5],
            'RUL_pred': [10, 25, 55, 90, 150]
        })
        context = assistant._build_fleet_context(sample_df)
        assert isinstance(context, str)
        assert 'FLEET HEALTH DATA' in context
        assert 'CRITICAL' in context

    def test_fallback_response(self):
        """Fallback should return helpful message when no API key."""
        assistant = MaintenanceAssistant(api_key=None)
        response = assistant._fallback_response("test prompt")
        assert isinstance(response, str)
        assert 'Offline' in response or 'offline' in response.lower() or 'GEMINI_API_KEY' in response

    def test_generate_fleet_summary_fallback(self):
        """Fleet summary should work in fallback mode."""
        assistant = MaintenanceAssistant(api_key=None)
        sample_df = pd.DataFrame({
            'unit_id': [1, 2, 3],
            'RUL_pred': [15, 50, 120]
        })
        summary = assistant.generate_fleet_summary(sample_df)
        assert isinstance(summary, str)
        assert len(summary) > 20


if __name__ == "__main__":
    pytest.main([__file__, '-v'])

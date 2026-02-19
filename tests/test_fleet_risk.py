"""
Tests for FleetRiskSimulator module
"""

import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestFleetRiskSimulator:
    """Test suite for fleet_risk_simulator.FleetRiskSimulator."""

    @pytest.fixture
    def simulator(self):
        from fleet_risk_simulator import FleetRiskSimulator
        return FleetRiskSimulator(n_simulations=500)  # small for speed

    @pytest.fixture
    def fleet_df(self):
        np.random.seed(42)
        return pd.DataFrame({
            'engine_id': [f'E{i}' for i in range(10)],
            'rul_pred': np.random.uniform(10, 150, 10),
            'rul_std': np.random.uniform(5, 20, 10)
        })

    def test_initialization(self, simulator):
        assert simulator.n_simulations == 500

    def test_setup_fleet(self, simulator, fleet_df):
        stats = simulator.setup_fleet(fleet_df)
        assert isinstance(stats, dict)

    def test_run_simulation(self, simulator, fleet_df):
        simulator.setup_fleet(fleet_df)
        results = simulator.run_simulation(horizon=30)
        assert isinstance(results, dict)
        assert 'expected_failures' in results or 'mean_failures' in results

    def test_sensitivity_analysis(self, simulator, fleet_df):
        simulator.setup_fleet(fleet_df)
        simulator.run_simulation(horizon=30)
        df_sens = simulator.sensitivity_analysis(horizons=[10, 20, 30])
        assert isinstance(df_sens, pd.DataFrame)
        assert len(df_sens) == 3

    def test_spare_parts_analysis(self, simulator, fleet_df):
        simulator.setup_fleet(fleet_df)
        simulator.run_simulation(horizon=30)
        spares_df = simulator.spare_parts_analysis(max_spares=5)
        assert isinstance(spares_df, pd.DataFrame)

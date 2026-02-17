"""
Unit tests for Phase 11 modules:
- DegradationClusterer
- SimilarityFinder
- CostOptimizer
- EnvelopeAnalyzer
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from degradation_clusterer import DegradationClusterer
from similarity_finder import SimilarityFinder, fast_dtw_distance
from cost_optimizer import CostOptimizer
from envelope_analyzer import EnvelopeAnalyzer

# synthetic_fleet fixture is provided by conftest.py


# ============================================================
# DegradationClusterer tests
# ============================================================

class TestDegradationClusterer:
    """Test suite for DegradationClusterer."""

    def test_initialization(self):
        clusterer = DegradationClusterer(n_clusters=3)
        assert clusterer is not None

    def test_extract_trajectory_features(self, synthetic_fleet):
        clusterer = DegradationClusterer(n_clusters=3)
        features = clusterer.extract_trajectory_features(synthetic_fleet)
        assert features is not None
        # Should have one row per engine
        assert features.shape[0] == synthetic_fleet['unit_id'].nunique()

    def test_cluster(self, synthetic_fleet):
        clusterer = DegradationClusterer(n_clusters=3)
        clusterer.extract_trajectory_features(synthetic_fleet)
        labels = clusterer.cluster()
        assert labels is not None
        assert len(labels) == synthetic_fleet['unit_id'].nunique()
        # Labels should be in range [0, n_clusters)
        assert set(labels).issubset({0, 1, 2})

    def test_cluster_profiles(self, synthetic_fleet):
        clusterer = DegradationClusterer(n_clusters=3)
        clusterer.extract_trajectory_features(synthetic_fleet)
        clusterer.cluster()
        profiles = clusterer.compute_cluster_profiles(synthetic_fleet)
        assert isinstance(profiles, dict)

    def test_find_optimal_k(self, synthetic_fleet):
        clusterer = DegradationClusterer(n_clusters=3)
        clusterer.extract_trajectory_features(synthetic_fleet)
        k_results = clusterer.find_optimal_k(k_range=range(2, 5))
        assert 'optimal_k' in k_results
        assert 'scores' in k_results
        assert 2 <= k_results['optimal_k'] <= 4


# ============================================================
# SimilarityFinder tests
# ============================================================

class TestSimilarityFinder:
    """Test suite for SimilarityFinder."""

    def test_fast_dtw_distance(self):
        """DTW distance of identical sequences should be ~0."""
        s1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        s2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        dist = fast_dtw_distance(s1, s2)
        assert dist == pytest.approx(0.0, abs=1e-6)

    def test_fast_dtw_different(self):
        """DTW distance of different sequences should be > 0."""
        s1 = np.array([1.0, 2.0, 3.0])
        s2 = np.array([5.0, 6.0, 7.0])
        dist = fast_dtw_distance(s1, s2)
        assert dist > 0

    def test_initialization(self):
        finder = SimilarityFinder(n_sensors=4, max_engines=20)
        assert finder is not None

    def test_build_fleet_profiles(self, synthetic_fleet):
        finder = SimilarityFinder(n_sensors=4, max_engines=10)
        profiles = finder.build_fleet_profiles(synthetic_fleet)
        assert isinstance(profiles, dict)
        assert len(profiles) == synthetic_fleet['unit_id'].nunique()

    def test_find_similar(self, synthetic_fleet):
        finder = SimilarityFinder(n_sensors=4, max_engines=10)
        finder.build_fleet_profiles(synthetic_fleet)
        similar = finder.find_similar(query_engine_id=1, k=3)
        assert isinstance(similar, list)
        assert len(similar) <= 3
        # Ensure query engine is not in results
        for s in similar:
            assert s['engine_id'] != 1

    def test_transfer_prognosis(self, synthetic_fleet):
        finder = SimilarityFinder(n_sensors=4, max_engines=10)
        finder.build_fleet_profiles(synthetic_fleet)
        prognosis = finder.transfer_prognosis(query_engine_id=1, k=3)
        assert isinstance(prognosis, dict)
        assert 'predicted_rul' in prognosis


# ============================================================
# CostOptimizer tests
# ============================================================

class TestCostOptimizer:
    """Test suite for CostOptimizer."""

    @pytest.fixture
    def fleet_df(self):
        np.random.seed(42)
        return pd.DataFrame({
            'engine_id': [f'E{i}' for i in range(1, 21)],
            'rul_pred': np.concatenate([
                np.random.randint(5, 25, 5),
                np.random.randint(30, 80, 7),
                np.random.randint(80, 180, 8),
            ]).astype(float)
        })

    def test_initialization(self):
        opt = CostOptimizer(budget_cap=200000, hangar_capacity=3, safety_rul=15)
        assert opt.budget_cap == 200000
        assert opt.hangar_capacity == 3

    def test_generate_solutions(self, fleet_df):
        opt = CostOptimizer()
        solutions = opt.generate_solutions(fleet_df, n_solutions=50)
        assert isinstance(solutions, pd.DataFrame)
        assert len(solutions) == 50
        assert 'total_cost' in solutions.columns
        assert 'risk_cost' in solutions.columns

    def test_find_pareto_front(self, fleet_df):
        opt = CostOptimizer()
        opt.generate_solutions(fleet_df, n_solutions=100)
        pareto = opt.find_pareto_front()
        assert isinstance(pareto, pd.DataFrame)
        assert len(pareto) > 0
        assert len(pareto) <= 100

    def test_recommend_solution(self, fleet_df):
        opt = CostOptimizer()
        opt.generate_solutions(fleet_df, n_solutions=100)
        opt.find_pareto_front()

        for pref in ['cost', 'safety', 'balanced', 'availability']:
            rec = opt.recommend_solution(pref)
            assert isinstance(rec, dict)
            assert 'total_cost' in rec
            assert 'preference' in rec
            assert rec['preference'] == pref

    def test_budget_constraint(self, fleet_df):
        """Solutions exceeding budget should be marked infeasible."""
        opt = CostOptimizer(budget_cap=10000)
        solutions = opt.generate_solutions(fleet_df, n_solutions=50)
        feasible = solutions[solutions['within_budget']]
        for _, row in feasible.iterrows():
            assert row['total_cost'] <= 10000


# ============================================================
# EnvelopeAnalyzer tests
# ============================================================

class TestEnvelopeAnalyzer:
    """Test suite for EnvelopeAnalyzer."""

    def test_initialization(self):
        analyzer = EnvelopeAnalyzer(method='percentile', margin=0.1)
        assert analyzer.method == 'percentile'

    def test_learn_envelope(self, synthetic_fleet):
        analyzer = EnvelopeAnalyzer()
        envelopes = analyzer.learn_envelope(synthetic_fleet, rul_threshold=50)
        assert isinstance(envelopes, dict)
        assert len(envelopes) > 0
        # Check envelope structure
        first_env = list(envelopes.values())[0]
        assert first_env.lower_bound < first_env.upper_bound
        assert first_env.percentile_5 < first_env.percentile_95

    def test_learn_envelope_iqr(self, synthetic_fleet):
        analyzer = EnvelopeAnalyzer(method='iqr')
        envelopes = analyzer.learn_envelope(synthetic_fleet, rul_threshold=50)
        assert len(envelopes) > 0

    def test_score_violations(self, synthetic_fleet):
        analyzer = EnvelopeAnalyzer()
        analyzer.learn_envelope(synthetic_fleet, rul_threshold=50)
        scored = analyzer.score_violations(synthetic_fleet)
        assert 'violation_score' in scored.columns
        assert 'n_violations' in scored.columns
        assert len(scored) == len(synthetic_fleet)

    def test_degradation_onset(self, synthetic_fleet):
        analyzer = EnvelopeAnalyzer()
        analyzer.learn_envelope(synthetic_fleet, rul_threshold=50)
        engine_df = synthetic_fleet[synthetic_fleet['unit_id'] == 1].sort_values('time_cycles')
        onset = analyzer.detect_degradation_onset(engine_df)
        assert isinstance(onset, dict)
        assert 'onset_cycle' in onset
        assert 'total_cycles' in onset
        assert 'max_violation' in onset

    def test_score_without_learn_raises(self, synthetic_fleet):
        """Scoring before learning should raise RuntimeError."""
        analyzer = EnvelopeAnalyzer()
        with pytest.raises(RuntimeError):
            analyzer.score_violations(synthetic_fleet)


if __name__ == "__main__":
    pytest.main([__file__, '-v'])

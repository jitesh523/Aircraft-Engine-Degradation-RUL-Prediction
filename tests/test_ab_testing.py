"""
Tests for A/B Testing Framework
"""

import pytest
import numpy as np
from ab_testing import ABTesting, ChampionChallengerFramework
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge


@pytest.fixture
def sample_data():
    """Generate sample data for testing"""
    np.random.seed(42)
    n_samples = 200
    n_features = 10
    
    X_test = np.random.randn(n_samples, n_features)
    y_test = np.random.rand(n_samples) * 100 + 50
    
    return X_test, y_test


@pytest.fixture
def trained_models(sample_data):
    """Create two trained models"""
    X_test, y_test = sample_data
    
    model_a = RandomForestRegressor(n_estimators=20, random_state=42)
    model_b = RandomForestRegressor(n_estimators=30, random_state=43)
    
    model_a.fit(X_test[:150], y_test[:150])
    model_b.fit(X_test[:150], y_test[:150])
    
    return model_a, model_b


class TestABTesting:
    """Test A/B testing framework"""
    
    def test_initialization(self, trained_models):
        """Test A/B test initialization"""
        model_a, model_b = trained_models
        ab_test = ABTesting(model_a, model_b, "Model_A", "Model_B")
        assert ab_test.model_a is not None
        assert ab_test.model_b is not None
        assert ab_test.name_a == "Model_A"
        assert ab_test.name_b == "Model_B"
    
    def test_run_comparison(self, sample_data, trained_models):
        """Test running comparison"""
        X_test, y_test = sample_data
        model_a, model_b = trained_models
        
        ab_test = ABTesting(model_a, model_b)
        results = ab_test.run_comparison(X_test[150:], y_test[150:])
        
        assert 'Model_A' in results
        assert 'Model_B' in results
        assert 'comparison' in results
        assert 'statistical_tests' in results
        
        assert 'rmse' in results['Model_A']
        assert 'mae' in results['Model_A']
        assert 'r2' in results['Model_A']
    
    def test_statistical_tests(self, sample_data, trained_models):
        """Test statistical significance tests"""
        X_test, y_test = sample_data
        model_a, model_b = trained_models
        
        ab_test = ABTesting(model_a, model_b)
        results = ab_test.run_comparison(X_test[150:], y_test[150:])
        
        st = results['statistical_tests']
        assert 'paired_t_test' in st
        assert 'wilcoxon_test' in st
        assert 'cohens_d' in st
        assert 'ci_95_difference' in st
        
        assert 'p_value' in st['paired_t_test']
        assert 'significant' in st['paired_t_test']
    
    def test_winner_determination(self, sample_data, trained_models):
        """Test winner determination"""
        X_test, y_test = sample_data
        model_a, model_b = trained_models
        
        ab_test = ABTesting(model_a, model_b)
        results = ab_test.run_comparison(X_test[150:], y_test[150:])
        
        assert 'winner' in results['comparison']
        assert 'improvement_percent' in results['comparison']
        assert 'recommendation' in results['comparison']
    
    def test_power_analysis(self, trained_models):
        """Test power analysis for sample size"""
        model_a, model_b = trained_models
        ab_test = ABTesting(model_a, model_b)
        
        n = ab_test.power_analysis(effect_size=0.5, alpha=0.05, power=0.8)
        assert isinstance(n, int)
        assert n > 0
    
    def test_bootstrap_comparison(self, sample_data, trained_models):
        """Test bootstrap comparison"""
        X_test, y_test = sample_data
        model_a, model_b = trained_models
        
        ab_test = ABTesting(model_a, model_b)
        bootstrap_results = ab_test.bootstrap_comparison(
            X_test[150:], y_test[150:],
            n_bootstrap=100,
            metric='rmse'
        )
        
        assert 'Model_A' in bootstrap_results
        assert 'Model_B' in bootstrap_results
        assert 'difference' in bootstrap_results
        assert 'ci_95' in bootstrap_results['Model_A']
    
    def test_print_summary(self, sample_data, trained_models):
        """Test summary printing"""
        X_test, y_test = sample_data
        model_a, model_b = trained_models
        
        ab_test = ABTesting(model_a, model_b)
        ab_test.run_comparison(X_test[150:], y_test[150:])
        
        # Should not raise exception
        ab_test.print_summary()


class TestChampionChallengerFramework:
    """Test champion/challenger framework"""
    
    def test_initialization(self, trained_models):
        """Test framework initialization"""
        model_a, _ = trained_models
        framework = ChampionChallengerFramework(model_a, "Champion")
        assert framework.champion is not None
        assert framework.champion_name == "Champion"
        assert len(framework.challengers) == 0
    
    def test_add_challenger(self, trained_models):
        """Test adding challenger"""
        model_a, model_b = trained_models
        framework = ChampionChallengerFramework(model_a)
        framework.add_challenger(model_b, "Challenger_1")
        assert "Challenger_1" in framework.challengers
    
    def test_test_challenger(self, sample_data, trained_models):
        """Test challenger testing"""
        X_test, y_test = sample_data
        model_a, model_b = trained_models
        
        framework = ChampionChallengerFramework(model_a, "Champion")
        framework.add_challenger(model_b, "Challenger_1")
        
        results = framework.test_challenger("Challenger_1", X_test[150:], y_test[150:])
        assert results is not None
        assert len(framework.test_history) == 1
    
    def test_promote_challenger(self, trained_models):
        """Test promoting challenger"""
        model_a, model_b = trained_models
        framework = ChampionChallengerFramework(model_a, "Champion")
        framework.add_challenger(model_b, "Challenger_1")
        
        framework.promote_challenger("Challenger_1")
        assert framework.champion_name == "Challenger_1"
        assert "Challenger_1" not in framework.challengers
    
    def test_get_test_history(self, sample_data, trained_models):
        """Test getting test history"""
        X_test, y_test = sample_data
        model_a, model_b = trained_models
        
        framework = ChampionChallengerFramework(model_a)
        framework.add_challenger(model_b, "Challenger_1")
        framework.test_challenger("Challenger_1", X_test[150:], y_test[150:])
        
        history = framework.get_test_history()
        assert len(history) == 1
        assert 'challenger' in history.columns
        assert 'winner' in history.columns

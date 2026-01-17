"""
Tests for Stacking Ensemble
"""

import pytest
import numpy as np
from models.stacking_ensemble import StackingEnsemble, WeightedAverageEnsemble
from models.gradient_boosting_models import XGBoostRULModel, LightGBMRULModel
from sklearn.ensemble import RandomForestRegressor


@pytest.fixture
def sample_data():
    """Generate sample data for testing"""
    np.random.seed(42)
    n_samples = 500
    n_features = 20
    
    X_train = np.random.randn(n_samples, n_features)
    y_train = np.random.rand(n_samples) * 100 + 50
    
    X_test = np.random.randn(100, n_features)
    y_test = np.random.rand(100) * 100 + 50
    
    return X_train, y_train, X_test, y_test


@pytest.fixture
def base_models():
    """Create base models for ensemble"""
    models = [
        ('xgboost', XGBoostRULModel(n_estimators=10)),
        ('lightgbm', LightGBMRULModel(n_estimators=10)),
        ('random_forest', RandomForestRegressor(n_estimators=10, random_state=42))
    ]
    return models


class TestStackingEnsemble:
    """Test stacking ensemble"""
    
    def test_initialization(self, base_models):
        """Test ensemble initialization"""
        ensemble = StackingEnsemble(base_models, n_folds=3)
        assert len(ensemble.base_models) == 3
        assert ensemble.n_folds == 3
    
    def test_training(self, sample_data, base_models):
        """Test ensemble training"""
        X_train, y_train, X_test, y_test = sample_data
        ensemble = StackingEnsemble(base_models, n_folds=2)
        
        ensemble.fit(X_train, y_train, verbose=False)
        assert ensemble.meta_features_shape_ is not None
    
    def test_prediction(self, sample_data, base_models):
        """Test ensemble prediction"""
        X_train, y_train, X_test, y_test = sample_data
        ensemble = StackingEnsemble(base_models, n_folds=2)
        ensemble.fit(X_train, y_train, verbose=False)
        
        predictions = ensemble.predict(X_test)
        assert len(predictions) == len(y_test)
        assert np.all(predictions >= 0)
    
    def test_base_model_predictions(self, sample_data, base_models):
        """Test getting predictions from all models"""
        X_train, y_train, X_test, y_test = sample_data
        ensemble = StackingEnsemble(base_models, n_folds=2)
        ensemble.fit(X_train, y_train, verbose=False)
        
        all_predictions = ensemble.predict_with_base_models(X_test)
        assert len(all_predictions) == len(base_models) + 1  # +1 for ensemble
        assert 'stacking_ensemble' in all_predictions
    
    def test_base_model_scores(self, sample_data, base_models):
        """Test evaluation of base models"""
        X_train, y_train, X_test, y_test = sample_data
        ensemble = StackingEnsemble(base_models, n_folds=2)
        ensemble.fit(X_train, y_train, verbose=False)
        
        scores_df = ensemble.get_base_model_scores(X_test, y_test)
        assert len(scores_df) == len(base_models) + 1
        assert all(col in scores_df.columns for col in ['model', 'rmse', 'mae', 'r2'])


class TestWeightedAverageEnsemble:
    """Test weighted average ensemble"""
    
    def test_initialization(self, base_models):
        """Test ensemble initialization"""
        ensemble = WeightedAverageEnsemble(base_models)
        assert len(ensemble.models) == 3
        assert np.allclose(ensemble.weights.sum(), 1.0)
    
    def test_custom_weights(self, base_models):
        """Test custom weights"""
        weights = [0.5, 0.3, 0.2]
        ensemble = WeightedAverageEnsemble(base_models, weights=weights)
        assert np.allclose(ensemble.weights.sum(), 1.0)
    
    def test_prediction(self, sample_data, base_models):
        """Test ensemble prediction"""
        X_train, y_train, X_test, y_test = sample_data
        ensemble = WeightedAverageEnsemble(base_models)
        ensemble.fit(X_train, y_train)
        
        predictions = ensemble.predict(X_test)
        assert len(predictions) == len(y_test)
        assert np.all(predictions >= 0)
    
    def test_weight_optimization(self, sample_data, base_models):
        """Test weight optimization"""
        X_train, y_train, X_test, y_test = sample_data
        ensemble = WeightedAverageEnsemble(base_models)
        ensemble.fit(X_train, y_train)
        
        ensemble.optimize_weights(X_test, y_test)
        assert np.allclose(ensemble.weights.sum(), 1.0)
        assert all(w >= 0 for w in ensemble.weights)

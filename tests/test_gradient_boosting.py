"""
Tests for Gradient Boosting Models
"""

import pytest
import numpy as np
import pandas as pd
from models.gradient_boosting_models import (
    XGBoostRULModel,
    LightGBMRULModel,
    CatBoostRULModel,
    create_gradient_boosting_model,
    evaluate_model
)


@pytest.fixture
def sample_data():
    """Generate sample data for testing"""
    np.random.seed(42)
    n_samples = 500
    n_features = 20
    
    X_train = np.random.randn(n_samples, n_features)
    y_train = np.random.rand(n_samples) * 100 + 50  # RUL between 50-150
    
    X_test = np.random.randn(100, n_features)
    y_test = np.random.rand(100) * 100 + 50
    
    return X_train, y_train, X_test, y_test


class TestXGBoostRULModel:
    """Test XGBoost model"""
    
    def test_initialization(self):
        """Test model initialization"""
        model = XGBoostRULModel()
        assert model.model is not None
        assert model.params['objective'] == 'reg:squarederror'
    
    def test_training(self, sample_data):
        """Test model training"""
        X_train, y_train, X_test, y_test = sample_data
        model = XGBoostRULModel(n_estimators=10)
        
        model.fit(X_train, y_train, verbose=False)
        assert model.feature_importance_ is not None
        assert len(model.feature_importance_) == X_train.shape[1]
    
    def test_prediction(self, sample_data):
        """Test model prediction"""
        X_train, y_train, X_test, y_test = sample_data
        model = XGBoostRULModel(n_estimators=10)
        model.fit(X_train, y_train, verbose=False)
        
        predictions = model.predict(X_test)
        assert len(predictions) == len(y_test)
        assert np.all(predictions >= 0)  # Non-negative RUL
    
    def test_feature_importance(self, sample_data):
        """Test feature importance extraction"""
        X_train, y_train, X_test, y_test = sample_data
        model = XGBoostRULModel(n_estimators=10)
        model.fit(X_train, y_train, verbose=False)
        
        importance_df = model.get_feature_importance()
        assert isinstance(importance_df, pd.DataFrame)
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns


class TestLightGBMRULModel:
    """Test LightGBM model"""
    
    def test_initialization(self):
        """Test model initialization"""
        model = LightGBMRULModel()
        assert model.model is not None
        assert model.params['objective'] == 'regression'
    
    def test_training(self, sample_data):
        """Test model training"""
        X_train, y_train, X_test, y_test = sample_data
        model = LightGBMRULModel(n_estimators=10)
        
        model.fit(X_train, y_train, verbose=False)
        assert model.feature_importance_ is not None
    
    def test_prediction(self, sample_data):
        """Test model prediction"""
        X_train, y_train, X_test, y_test = sample_data
        model = LightGBMRULModel(n_estimators=10)
        model.fit(X_train, y_train, verbose=False)
        
        predictions = model.predict(X_test)
        assert len(predictions) == len(y_test)
        assert np.all(predictions >= 0)


class TestCatBoostRULModel:
    """Test CatBoost model"""
    
    def test_initialization(self):
        """Test model initialization"""
        model = CatBoostRULModel()
        assert model.model is not None
        assert model.params['loss_function'] == 'RMSE'
    
    def test_training(self, sample_data):
        """Test model training"""
        X_train, y_train, X_test, y_test = sample_data
        model = CatBoostRULModel(iterations=10)
        
        model.fit(X_train, y_train, verbose=False)
        assert model.feature_importance_ is not None
    
    def test_prediction(self, sample_data):
        """Test model prediction"""
        X_train, y_train, X_test, y_test = sample_data
        model = CatBoostRULModel(iterations=10)
        model.fit(X_train, y_train, verbose=False)
        
        predictions = model.predict(X_test)
        assert len(predictions) == len(y_test)
        assert np.all(predictions >= 0)


class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_create_gradient_boosting_model(self):
        """Test model factory function"""
        xgb_model = create_gradient_boosting_model('xgboost', n_estimators=10)
        assert isinstance(xgb_model, XGBoostRULModel)
        
        lgb_model = create_gradient_boosting_model('lightgbm', n_estimators=10)
        assert isinstance(lgb_model, LightGBMRULModel)
        
        cb_model = create_gradient_boosting_model('catboost', iterations=10)
        assert isinstance(cb_model, CatBoostRULModel)
        
        with pytest.raises(ValueError):
            create_gradient_boosting_model('invalid_model')
    
    def test_evaluate_model(self, sample_data):
        """Test model evaluation"""
        X_train, y_train, X_test, y_test = sample_data
        model = XGBoostRULModel(n_estimators=10)
        model.fit(X_train, y_train, verbose=False)
        
        metrics = evaluate_model(model, X_test, y_test)
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        assert all(isinstance(v, (int, float)) for v in metrics.values())

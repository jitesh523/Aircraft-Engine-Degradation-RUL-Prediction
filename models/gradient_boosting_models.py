"""
Gradient Boosting Models for RUL Prediction
Implements XGBoost, LightGBM, and CatBoost regressors
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle


class XGBoostRULModel:
    """XGBoost model for RUL prediction"""
    
    def __init__(self, **params):
        """
        Initialize XGBoost regressor
        
        Args:
            **params: XGBoost hyperparameters
        """
        default_params = {
            'objective': 'reg:squarederror',
            'n_estimators': 500,
            'max_depth': 7,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        }
        default_params.update(params)
        self.params = default_params
        self.model = xgb.XGBRegressor(**default_params)
        self.feature_importance_ = None
    
    def fit(self, X, y, eval_set=None, early_stopping_rounds=50, verbose=False):
        """
        Train the XGBoost model
        
        Args:
            X: Training features
            y: Training targets
            eval_set: Optional validation set for early stopping
            early_stopping_rounds: Rounds for early stopping
            verbose: Print training progress
            
        Returns:
            self
        """
        fit_params = {
            'verbose': verbose
        }
        
        if eval_set is not None:
            fit_params['eval_set'] = eval_set
            fit_params['early_stopping_rounds'] = early_stopping_rounds
        
        self.model.fit(X, y, **fit_params)
        self.feature_importance_ = self.model.feature_importances_
        return self
    
    def predict(self, X):
        """Make predictions"""
        predictions = self.model.predict(X)
        # Ensure non-negative RUL
        return np.maximum(predictions, 0)
    
    def get_feature_importance(self, feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """Get feature importance scores"""
        if self.feature_importance_ is None:
            raise RuntimeError("Model not trained yet")
        
        importance_df = pd.DataFrame({
            'feature': feature_names if feature_names else range(len(self.feature_importance_)),
            'importance': self.feature_importance_
        })
        return importance_df.sort_values('importance', ascending=False)
    
    def save(self, filepath: str):
        """Save model to file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(filepath: str):
        """Load model from file"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


class LightGBMRULModel:
    """LightGBM model for RUL prediction"""
    
    def __init__(self, **params):
        """
        Initialize LightGBM regressor
        
        Args:
            **params: LightGBM hyperparameters
        """
        default_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'n_estimators': 500,
            'num_leaves': 63,
            'max_depth': 8,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        default_params.update(params)
        self.params = default_params
        self.model = lgb.LGBMRegressor(**default_params)
        self.feature_importance_ = None
    
    def fit(self, X, y, eval_set=None, early_stopping_rounds=50, verbose=False):
        """
        Train the LightGBM model
        
        Args:
            X: Training features
            y: Training targets
            eval_set: Optional validation set for early stopping
            early_stopping_rounds: Rounds for early stopping
            verbose: Print training progress
            
        Returns:
            self
        """
        callbacks = []
        if not verbose:
            callbacks.append(lgb.log_evaluation(period=0))
        
        fit_params = {
            'callbacks': callbacks
        }
        
        if eval_set is not None:
            fit_params['eval_set'] = eval_set
            if early_stopping_rounds:
                callbacks.append(lgb.early_stopping(early_stopping_rounds))
        
        self.model.fit(X, y, **fit_params)
        self.feature_importance_ = self.model.feature_importances_
        return self
    
    def predict(self, X):
        """Make predictions"""
        predictions = self.model.predict(X)
        # Ensure non-negative RUL
        return np.maximum(predictions, 0)
    
    def get_feature_importance(self, feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """Get feature importance scores"""
        if self.feature_importance_ is None:
            raise RuntimeError("Model not trained yet")
        
        importance_df = pd.DataFrame({
            'feature': feature_names if feature_names else range(len(self.feature_importance_)),
            'importance': self.feature_importance_
        })
        return importance_df.sort_values('importance', ascending=False)
    
    def save(self, filepath: str):
        """Save model to file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(filepath: str):
        """Load model from file"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


class CatBoostRULModel:
    """CatBoost model for RUL prediction"""
    
    def __init__(self, **params):
        """
        Initialize CatBoost regressor
        
        Args:
            **params: CatBoost hyperparameters
        """
        default_params = {
            'loss_function': 'RMSE',
            'iterations': 500,
            'depth': 7,
            'learning_rate': 0.05,
            'l2_leaf_reg': 3.0,
            'random_seed': 42,
            'verbose': False,
            'thread_count': -1
        }
        default_params.update(params)
        self.params = default_params
        self.model = cb.CatBoostRegressor(**default_params)
        self.feature_importance_ = None
    
    def fit(self, X, y, eval_set=None, early_stopping_rounds=50, verbose=False):
        """
        Train the CatBoost model
        
        Args:
            X: Training features
            y: Training targets
            eval_set: Optional validation set for early stopping
            early_stopping_rounds: Rounds for early stopping
            verbose: Print training progress
            
        Returns:
            self
        """
        fit_params = {
            'verbose': verbose
        }
        
        if eval_set is not None:
            fit_params['eval_set'] = eval_set
            if early_stopping_rounds:
                fit_params['early_stopping_rounds'] = early_stopping_rounds
        
        self.model.fit(X, y, **fit_params)
        self.feature_importance_ = self.model.feature_importances_
        return self
    
    def predict(self, X):
        """Make predictions"""
        predictions = self.model.predict(X)
        # Ensure non-negative RUL
        return np.maximum(predictions, 0)
    
    def get_feature_importance(self, feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """Get feature importance scores"""
        if self.feature_importance_ is None:
            raise RuntimeError("Model not trained yet")
        
        importance_df = pd.DataFrame({
            'feature': feature_names if feature_names else range(len(self.feature_importance_)),
            'importance': self.feature_importance_
        })
        return importance_df.sort_values('importance', ascending=False)
    
    def save(self, filepath: str):
        """Save model to file"""
        self.model.save_model(filepath)
    
    @staticmethod
    def load(filepath: str):
        """Load model from file"""
        model_instance = CatBoostRULModel()
        model_instance.model = cb.CatBoostRegressor()
        model_instance.model.load_model(filepath)
        model_instance.feature_importance_ = model_instance.model.feature_importances_
        return model_instance


def create_gradient_boosting_model(model_type: str, **params):
    """
    Factory function to create gradient boosting models
    
    Args:
        model_type: Type of model ('xgboost', 'lightgbm', 'catboost')
        **params: Model hyperparameters
        
    Returns:
        Model instance
    """
    model_map = {
        'xgboost': XGBoostRULModel,
        'lightgbm': LightGBMRULModel,
        'catboost': CatBoostRULModel
    }
    
    if model_type.lower() not in model_map:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(model_map.keys())}")
    
    return model_map[model_type.lower()](**params)


def evaluate_model(model, X_test, y_test) -> Dict[str, float]:
    """
    Evaluate a trained model
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        
    Returns:
        Dictionary of metrics
    """
    predictions = model.predict(X_test)
    
    return {
        'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
        'mae': mean_absolute_error(y_test, predictions),
        'r2': r2_score(y_test, predictions)
    }

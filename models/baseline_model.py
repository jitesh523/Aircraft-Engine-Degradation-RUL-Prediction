"""
Baseline Models for RUL Prediction
Implements Random Forest and Linear Regression baselines
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Tuple
import pickle
import config
from utils import setup_logging

logger = setup_logging(__name__)


class BaselineModel:
    """
    Baseline regression models for RUL prediction
    """
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize baseline model
        
        Args:
            model_type: 'random_forest' or 'linear_regression'
        """
        self.model_type = model_type
        self.model = None
        self.feature_importance = None
        
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(**config.BASELINE_CONFIG['random_forest'])
            logger.info("Initialized Random Forest baseline model")
        elif model_type == 'linear_regression':
            self.model = LinearRegression(**config.BASELINE_CONFIG['linear_regression'])
            logger.info("Initialized Linear Regression baseline model")
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the baseline model
        
        Args:
            X_train: Training features
            y_train: Training targets (RUL values)
        """
        logger.info(f"Training {self.model_type} on {len(X_train)} samples...")
        
        self.model.fit(X_train, y_train)
        
        # Extract feature importance for Random Forest
        if self.model_type == 'random_forest':
            self.feature_importance = self.model.feature_importances_
        
        logger.info("Training complete")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make RUL predictions
        
        Args:
            X: Features
            
        Returns:
            Predicted RUL values
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        predictions = self.model.predict(X)
        
        # Clip negative predictions to 0
        predictions = np.maximum(predictions, 0)
        
        return predictions
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: True RUL values
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating model...")
        
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        metrics = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'model_type': self.model_type
        }
        
        logger.info(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.4f}")
        
        return metrics
    
    def get_feature_importance(self, feature_names: list = None, top_k: int = 20) -> pd.DataFrame:
        """
        Get feature importance (Random Forest only)
        
        Args:
            feature_names: List of feature names
            top_k: Number of top features to return
            
        Returns:
            DataFrame with feature importance scores
        """
        if self.model_type != 'random_forest':
            logger.warning("Feature importance only available for Random Forest")
            return None
        
        if self.feature_importance is None:
            logger.warning("Model not trained yet")
            return None
        
        # Create DataFrame
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(self.feature_importance))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importance
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_k)
    
    def save(self, filepath: str) -> None:
        """
        Save the trained model
        
        Args:
            filepath: Path to save the model
        """
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        Load a trained model
        
        Args:
            filepath: Path to the saved model
        """
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        
        # Extract feature importance if Random Forest
        if self.model_type == 'random_forest' and hasattr(self.model, 'feature_importances_'):
            self.feature_importance = self.model.feature_importances_
        
        logger.info(f"Model loaded from {filepath}")


if __name__ == "__main__":
    # Test baseline models
    print("="*60)
    print("Testing Baseline Models")
    print("="*60)
    
    # Generate synthetic data for testing
    np.random.seed(42)
    X_train = np.random.randn(1000, 20)
    y_train = np.random.randint(0, 150, size=1000)
    X_test = np.random.randn(200, 20)
    y_test = np.random.randint(0, 150, size=200)
    
    # Test Random Forest
    print("\n1. Random Forest Baseline:")
    rf_model = BaselineModel('random_forest')
    rf_model.train(X_train, y_train)
    rf_metrics = rf_model.evaluate(X_test, y_test)
    print(f"   RMSE: {rf_metrics['rmse']:.2f}")
    print(f"   MAE: {rf_metrics['mae']:.2f}")
    print(f"   R2: {rf_metrics['r2']:.4f}")
    
    # Test Linear Regression
    print("\n2. Linear Regression Baseline:")
    lr_model = BaselineModel('linear_regression')
    lr_model.train(X_train, y_train)
    lr_metrics = lr_model.evaluate(X_test, y_test)
    print(f"   RMSE: {lr_metrics['rmse']:.2f}")
    print(f"   MAE: {lr_metrics['mae']:.2f}")
    print(f"   R2: {lr_metrics['r2']:.4f}")
    
    # Feature importance
    print("\n3. Feature Importance (Random Forest):")
    importance = rf_model.get_feature_importance(top_k=5)
    if importance is not None:
        print(importance)
    
    print("\n" + "="*60)
    print("Baseline models test complete!")
    print("="*60)

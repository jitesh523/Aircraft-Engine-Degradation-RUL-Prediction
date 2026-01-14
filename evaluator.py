"""
Model Evaluator for RUL Prediction
Evaluates models and generates performance metrics
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Tuple
import config
from utils import setup_logging, asymmetric_score

logger = setup_logging(__name__)


class RULEvaluator:
    """
    Evaluator for RUL prediction models
    """
    
    def __init__(self):
        """Initialize evaluator"""
        self.results = {}
        logger.info("Initialized RUL Evaluator")
    
    def evaluate_predictions(self, 
                            y_true: np.ndarray, 
                            y_pred: np.ndarray,
                            model_name: str = 'model') -> Dict:
        """
        Evaluate RUL predictions
        
        Args:
            y_true: True RUL values
            y_pred: Predicted RUL values
            model_name: Name of the model
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating {model_name}...")
        
        # Basic metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Asymmetric scoring (NASA metric)
        asym_score = asymmetric_score(y_true, y_pred)
        
        # Error statistics
        errors = y_pred - y_true
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        max_error = np.max(np.abs(errors))
        
        # Under-prediction and over-prediction
        under_predictions = np.sum(errors < 0)  # Predicting less RUL than actual (dangerous)
        over_predictions = np.sum(errors > 0)   # Predicting more RUL than actual (conservative)
        
        metrics = {
            'model_name': model_name,
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'asymmetric_score': float(asym_score),
            'mean_error': float(mean_error),
            'std_error': float(std_error),
            'max_error': float(max_error),
            'under_predictions': int(under_predictions),
            'over_predictions': int(over_predictions),
            'total_predictions': len(y_true)
        }
        
        logger.info(f"  RMSE: {rmse:.2f} cycles")
        logger.info(f"  MAE: {mae:.2f} cycles")
        logger.info(f"  R²: {r2:.4f}")
        logger.info(f"  Asymmetric Score: {asym_score:.2f}")
        logger.info(f"  Mean Error: {mean_error:.2f} cycles")
        logger.info(f"  Under-predictions: {under_predictions}/{len(y_true)} ({under_predictions/len(y_true)*100:.1f}%)")
        
        self.results[model_name] = metrics
        return metrics
    
    def compare_models(self, predictions_dict: Dict[str, np.ndarray], y_true: np.ndarray) -> pd.DataFrame:
        """
        Compare multiple models
        
        Args:
            predictions_dict: Dictionary of {model_name: predictions}
            y_true: True RUL values
            
        Returns:
            DataFrame with comparison metrics
        """
        logger.info("Comparing models...")
        
        comparison_results = []
        
        for model_name, y_pred in predictions_dict.items():
            metrics = self.evaluate_predictions(y_true, y_pred, model_name)
            comparison_results.append(metrics)
        
        comparison_df = pd.DataFrame(comparison_results)
        comparison_df = comparison_df.sort_values('rmse')
        
        logger.info("\nModel Comparison (sorted by RMSE):")
        logger.info(comparison_df[['model_name', 'rmse', 'mae', 'r2']].to_string(index=False))
        
        return comparison_df
    
    def check_performance_targets(self, metrics: Dict) -> Dict[str, bool]:
        """
        Check if metrics meet target performance
        
        Args:
            metrics: Dictionary of metrics
            
        Returns:
            Dictionary of checks {metric: passed}
        """
        checks = {
            'rmse_target': metrics['rmse'] <= config.TARGET_METRICS['RMSE'],
            'mae_target': metrics['mae'] <= config.TARGET_METRICS['MAE'],
            'r2_target': metrics['r2'] >= config.TARGET_METRICS['R2']
        }
        
        logger.info("\nPerformance Target Check:")
        logger.info(f"  RMSE ≤ {config.TARGET_METRICS['RMSE']}: {'✓' if checks['rmse_target'] else '✗'} ({metrics['rmse']:.2f})")
        logger.info(f"  MAE ≤ {config.TARGET_METRICS['MAE']}: {'✓' if checks['mae_target'] else '✗'} ({metrics['mae']:.2f})")
        logger.info(f"  R² ≥ {config.TARGET_METRICS['R2']}: {'✓' if checks['r2_target'] else '✗'} ({metrics['r2']:.4f})")
        
        return checks
    
    def get_engine_level_errors(self, 
                                df: pd.DataFrame,
                                pred_col: str = 'RUL_pred',
                                true_col: str = 'RUL_true') -> pd.DataFrame:
        """
        Calculate per-engine prediction errors
        
        Args:
            df: DataFrame with unit_id, predictions, and true values
            pred_col: Name of prediction column
            true_col: Name of true RUL column
            
        Returns:
            DataFrame with per-engine errors
        """
        logger.info("Calculating per-engine errors...")
        
        # Calculate error for each engine
        engine_errors = df.groupby('unit_id').apply(
            lambda x: pd.Series({
                'true_rul': x[true_col].iloc[-1] if len(x) > 0 else np.nan,
                'pred_rul': x[pred_col].iloc[-1] if len(x) > 0 else np.nan,
                'error': x[pred_col].iloc[-1] - x[true_col].iloc[-1] if len(x) > 0 else np.nan,
                'abs_error': abs(x[pred_col].iloc[-1] - x[true_col].iloc[-1]) if len(x) > 0 else np.nan
            })
        ).reset_index()
        
        logger.info(f"Computed errors for {len(engine_errors)} engines")
        
        return engine_errors


if __name__ == "__main__":
    # Test evaluator
    print("="*60)
    print("Testing RUL Evaluator")
    print("="*60)
    
    # Generate synthetic data
    np.random.seed(42)
    y_true = np.random.randint(0, 150, size=100).astype(float)
    y_pred1 = y_true + np.random.randn(100) * 15  # Model 1
    y_pred2 = y_true + np.random.randn(100) * 20  # Model 2 (worse)
    
    # Clip negative predictions
    y_pred1 = np.maximum(y_pred1, 0)
    y_pred2 = np.maximum(y_pred2, 0)
    
    # Create evaluator
    evaluator = RULEvaluator()
    
    # Evaluate single model
    print("\n1. Single Model Evaluation:")
    metrics1 = evaluator.evaluate_predictions(y_true, y_pred1, 'LSTM')
    
    # Check targets
    print("\n2. Performance Target Check:")
    checks = evaluator.check_performance_targets(metrics1)
    
    # Compare models
    print("\n3. Model Comparison:")
    predictions_dict = {
        'LSTM': y_pred1,
        'Random Forest': y_pred2
    }
    comparison_df = evaluator.compare_models(predictions_dict, y_true)
    
    print("\n" + "="*60)
    print("Evaluator test complete!")
    print("="*60)

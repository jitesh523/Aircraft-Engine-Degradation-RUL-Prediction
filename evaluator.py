"""
Model Evaluator for RUL Prediction
Evaluates models and generates performance metrics
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from typing import Dict, Tuple, List, Callable
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
    
    def cross_validate_time_series(self,
                                   X: np.ndarray,
                                   y: np.ndarray,
                                   model_factory: Callable,
                                   n_splits: int = 5) -> Dict:
        """
        Perform time-series aware cross-validation
        
        Uses TimeSeriesSplit to respect temporal ordering and prevent data leakage.
        
        Args:
            X: Feature matrix
            y: Target values
            model_factory: Function that returns a fresh model instance
            n_splits: Number of cross-validation folds
            
        Returns:
            Dictionary with cross-validation results and confidence intervals
        """
        logger.info(f"Performing time-series cross-validation with {n_splits} folds...")
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Get fresh model and train
            model = model_factory()
            model.fit(X_train, y_train)
            
            # Predict and evaluate
            y_pred = model.predict(X_val)
            
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            
            fold_metrics.append({
                'fold': fold + 1,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'n_train': len(train_idx),
                'n_val': len(val_idx)
            })
            
            logger.info(f"  Fold {fold + 1}: RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.4f}")
        
        # Calculate statistics with confidence intervals
        rmse_values = [m['rmse'] for m in fold_metrics]
        mae_values = [m['mae'] for m in fold_metrics]
        r2_values = [m['r2'] for m in fold_metrics]
        
        cv_results = {
            'n_folds': n_splits,
            'fold_metrics': fold_metrics,
            'rmse_mean': float(np.mean(rmse_values)),
            'rmse_std': float(np.std(rmse_values)),
            'rmse_ci_95': (float(np.mean(rmse_values) - 1.96 * np.std(rmse_values) / np.sqrt(n_splits)),
                          float(np.mean(rmse_values) + 1.96 * np.std(rmse_values) / np.sqrt(n_splits))),
            'mae_mean': float(np.mean(mae_values)),
            'mae_std': float(np.std(mae_values)),
            'mae_ci_95': (float(np.mean(mae_values) - 1.96 * np.std(mae_values) / np.sqrt(n_splits)),
                         float(np.mean(mae_values) + 1.96 * np.std(mae_values) / np.sqrt(n_splits))),
            'r2_mean': float(np.mean(r2_values)),
            'r2_std': float(np.std(r2_values)),
            'r2_ci_95': (float(np.mean(r2_values) - 1.96 * np.std(r2_values) / np.sqrt(n_splits)),
                        float(np.mean(r2_values) + 1.96 * np.std(r2_values) / np.sqrt(n_splits)))
        }
        
        logger.info(f"\nCross-Validation Summary:")
        logger.info(f"  RMSE: {cv_results['rmse_mean']:.2f} ± {cv_results['rmse_std']:.2f} (95% CI: {cv_results['rmse_ci_95'][0]:.2f}-{cv_results['rmse_ci_95'][1]:.2f})")
        logger.info(f"  MAE:  {cv_results['mae_mean']:.2f} ± {cv_results['mae_std']:.2f} (95% CI: {cv_results['mae_ci_95'][0]:.2f}-{cv_results['mae_ci_95'][1]:.2f})")
        logger.info(f"  R²:   {cv_results['r2_mean']:.4f} ± {cv_results['r2_std']:.4f} (95% CI: {cv_results['r2_ci_95'][0]:.4f}-{cv_results['r2_ci_95'][1]:.4f})")
        
        return cv_results
    
    def bootstrap_confidence_interval(self,
                                      y_true: np.ndarray,
                                      y_pred: np.ndarray,
                                      metric_func: Callable,
                                      n_iterations: int = 1000,
                                      confidence_level: float = 0.95) -> Tuple[float, float, float]:
        """
        Calculate bootstrap confidence interval for a metric
        
        Args:
            y_true: True values
            y_pred: Predicted values
            metric_func: Function that computes metric(y_true, y_pred)
            n_iterations: Number of bootstrap iterations
            confidence_level: Confidence level (default 95%)
            
        Returns:
            Tuple of (mean, lower_bound, upper_bound)
        """
        n_samples = len(y_true)
        bootstrap_metrics = []
        
        for _ in range(n_iterations):
            # Sample with replacement
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            y_true_sample = y_true[indices]
            y_pred_sample = y_pred[indices]
            
            # Calculate metric
            metric_value = metric_func(y_true_sample, y_pred_sample)
            bootstrap_metrics.append(metric_value)
        
        bootstrap_metrics = np.array(bootstrap_metrics)
        mean_value = np.mean(bootstrap_metrics)
        
        # Calculate percentile confidence intervals
        alpha = (1 - confidence_level) / 2
        lower_bound = np.percentile(bootstrap_metrics, alpha * 100)
        upper_bound = np.percentile(bootstrap_metrics, (1 - alpha) * 100)
        
        return float(mean_value), float(lower_bound), float(upper_bound)
    
    def evaluate_with_confidence(self,
                                 y_true: np.ndarray,
                                 y_pred: np.ndarray,
                                 model_name: str = 'model',
                                 n_bootstrap: int = 1000) -> Dict:
        """
        Evaluate predictions with bootstrap confidence intervals
        
        Args:
            y_true: True RUL values
            y_pred: Predicted RUL values
            model_name: Name of the model
            n_bootstrap: Number of bootstrap iterations
            
        Returns:
            Dictionary of metrics with confidence intervals
        """
        logger.info(f"Evaluating {model_name} with bootstrap confidence intervals...")
        
        # Define metric functions
        def rmse_func(yt, yp):
            return np.sqrt(mean_squared_error(yt, yp))
        
        def mae_func(yt, yp):
            return mean_absolute_error(yt, yp)
        
        def r2_func(yt, yp):
            return r2_score(yt, yp)
        
        # Calculate bootstrap confidence intervals
        rmse_mean, rmse_low, rmse_high = self.bootstrap_confidence_interval(
            y_true, y_pred, rmse_func, n_bootstrap
        )
        mae_mean, mae_low, mae_high = self.bootstrap_confidence_interval(
            y_true, y_pred, mae_func, n_bootstrap
        )
        r2_mean, r2_low, r2_high = self.bootstrap_confidence_interval(
            y_true, y_pred, r2_func, n_bootstrap
        )
        
        metrics = {
            'model_name': model_name,
            'rmse': rmse_mean,
            'rmse_ci_95': (rmse_low, rmse_high),
            'mae': mae_mean,
            'mae_ci_95': (mae_low, mae_high),
            'r2': r2_mean,
            'r2_ci_95': (r2_low, r2_high),
            'n_bootstrap': n_bootstrap
        }
        
        logger.info(f"  RMSE: {rmse_mean:.2f} (95% CI: {rmse_low:.2f}-{rmse_high:.2f})")
        logger.info(f"  MAE:  {mae_mean:.2f} (95% CI: {mae_low:.2f}-{mae_high:.2f})")
        logger.info(f"  R²:   {r2_mean:.4f} (95% CI: {r2_low:.4f}-{r2_high:.4f})")
        
        self.results[f"{model_name}_with_ci"] = metrics
        return metrics


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

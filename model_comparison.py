"""
Model Comparison Utility for RUL Prediction
Provides comprehensive comparison and analysis of different models
"""

import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import config
from utils import setup_logging, asymmetric_score

logger = setup_logging(__name__)


class ModelComparisonReport:
    """
    Generate comprehensive model comparison reports
    """
    
    def __init__(self):
        """Initialize comparison report generator"""
        self.comparison_results = {}
        self.model_predictions = {}
        logger.info("Initialized ModelComparisonReport")
    
    def add_model_results(self,
                          model_name: str,
                          y_true: np.ndarray,
                          y_pred: np.ndarray,
                          training_time: float = None,
                          inference_time: float = None,
                          model_size_mb: float = None) -> Dict:
        """
        Add model results for comparison
        
        Args:
            model_name: Name of the model
            y_true: True RUL values
            y_pred: Predicted RUL values
            training_time: Training time in seconds (optional)
            inference_time: Inference time per sample in ms (optional)
            model_size_mb: Model size in MB (optional)
            
        Returns:
            Dictionary with model metrics
        """
        logger.info(f"Adding results for {model_name}...")
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        asym_score = asymmetric_score(y_true, y_pred)
        
        # Error analysis
        errors = y_pred - y_true
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        
        # Under/over prediction counts
        under_preds = np.sum(errors < 0)
        over_preds = np.sum(errors > 0)
        
        # Percentile errors
        abs_errors = np.abs(errors)
        p50_error = np.percentile(abs_errors, 50)
        p90_error = np.percentile(abs_errors, 90)
        p99_error = np.percentile(abs_errors, 99)
        
        metrics = {
            'model_name': model_name,
            'n_samples': len(y_true),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'asymmetric_score': float(asym_score),
            'mean_error': float(mean_error),
            'std_error': float(std_error),
            'under_predictions': int(under_preds),
            'over_predictions': int(over_preds),
            'p50_error': float(p50_error),
            'p90_error': float(p90_error),
            'p99_error': float(p99_error),
            'training_time_s': training_time,
            'inference_time_ms': inference_time,
            'model_size_mb': model_size_mb
        }
        
        self.comparison_results[model_name] = metrics
        self.model_predictions[model_name] = {
            'y_true': y_true.copy(),
            'y_pred': y_pred.copy()
        }
        
        logger.info(f"  RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}")
        
        return metrics
    
    def get_comparison_dataframe(self) -> pd.DataFrame:
        """
        Get comparison results as DataFrame
        
        Returns:
            DataFrame with all model metrics
        """
        if not self.comparison_results:
            return pd.DataFrame()
        
        df = pd.DataFrame(list(self.comparison_results.values()))
        
        # Sort by RMSE (best first)
        df = df.sort_values('rmse')
        
        # Add rank
        df['rank'] = range(1, len(df) + 1)
        
        return df
    
    def statistical_comparison(self, 
                               model_a: str, 
                               model_b: str,
                               n_bootstrap: int = 1000) -> Dict:
        """
        Perform statistical comparison between two models
        
        Uses bootstrap to determine if difference is significant
        
        Args:
            model_a: First model name
            model_b: Second model name
            n_bootstrap: Number of bootstrap iterations
            
        Returns:
            Dictionary with statistical comparison results
        """
        if model_a not in self.model_predictions or model_b not in self.model_predictions:
            raise ValueError(f"Model predictions not found for {model_a} or {model_b}")
        
        y_true = self.model_predictions[model_a]['y_true']
        y_pred_a = self.model_predictions[model_a]['y_pred']
        y_pred_b = self.model_predictions[model_b]['y_pred']
        
        n_samples = len(y_true)
        
        # Bootstrap RMSE differences
        rmse_diffs = []
        for _ in range(n_bootstrap):
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            
            rmse_a = np.sqrt(mean_squared_error(y_true[indices], y_pred_a[indices]))
            rmse_b = np.sqrt(mean_squared_error(y_true[indices], y_pred_b[indices]))
            rmse_diffs.append(rmse_a - rmse_b)
        
        rmse_diffs = np.array(rmse_diffs)
        
        # Calculate statistics
        mean_diff = np.mean(rmse_diffs)
        std_diff = np.std(rmse_diffs)
        ci_low = np.percentile(rmse_diffs, 2.5)
        ci_high = np.percentile(rmse_diffs, 97.5)
        
        # Significance: CI doesn't include 0
        significant = (ci_low > 0) or (ci_high < 0)
        
        # Determine winner
        if mean_diff > 0:
            winner = model_b
            improvement = abs(mean_diff)
        elif mean_diff < 0:
            winner = model_a
            improvement = abs(mean_diff)
        else:
            winner = 'tie'
            improvement = 0
        
        result = {
            'model_a': model_a,
            'model_b': model_b,
            'rmse_diff_mean': float(mean_diff),
            'rmse_diff_std': float(std_diff),
            'ci_95_lower': float(ci_low),
            'ci_95_upper': float(ci_high),
            'significant_at_95': significant,
            'winner': winner,
            'rmse_improvement': float(improvement)
        }
        
        logger.info(f"Statistical comparison: {model_a} vs {model_b}")
        logger.info(f"  Winner: {winner} (Δ RMSE = {improvement:.2f}, Significant: {significant})")
        
        return result
    
    def find_best_model(self, 
                        metric: str = 'rmse',
                        constraints: Dict = None) -> Dict:
        """
        Find best model based on specified metric and constraints
        
        Args:
            metric: Metric to optimize ('rmse', 'mae', 'r2', 'asymmetric_score')
            constraints: Optional constraints e.g., {'inference_time_ms': 10, 'model_size_mb': 100}
            
        Returns:
            Dictionary with best model info
        """
        df = self.get_comparison_dataframe()
        
        if df.empty:
            return {}
        
        # Apply constraints
        if constraints:
            for constraint_name, constraint_value in constraints.items():
                if constraint_name in df.columns:
                    df = df[df[constraint_name] <= constraint_value]
        
        if df.empty:
            logger.warning("No models satisfy constraints")
            return {}
        
        # Find best based on metric
        if metric == 'r2':
            best_idx = df[metric].idxmax()  # Maximize R²
        else:
            best_idx = df[metric].idxmin()  # Minimize others
        
        best_model = df.loc[best_idx].to_dict()
        
        logger.info(f"Best model by {metric}: {best_model['model_name']}")
        
        return best_model
    
    def generate_error_analysis(self, model_name: str) -> Dict:
        """
        Generate detailed error analysis for a model
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with detailed error analysis
        """
        if model_name not in self.model_predictions:
            raise ValueError(f"Model predictions not found for {model_name}")
        
        y_true = self.model_predictions[model_name]['y_true']
        y_pred = self.model_predictions[model_name]['y_pred']
        errors = y_pred - y_true
        
        # Bin analysis: performance at different RUL ranges
        rul_bins = [0, 30, 50, 80, 125, float('inf')]
        bin_labels = ['0-30', '30-50', '50-80', '80-125', '125+']
        
        bin_analysis = []
        for i in range(len(rul_bins) - 1):
            mask = (y_true >= rul_bins[i]) & (y_true < rul_bins[i+1])
            if mask.sum() > 0:
                bin_errors = errors[mask]
                bin_y_true = y_true[mask]
                bin_y_pred = y_pred[mask]
                
                bin_analysis.append({
                    'rul_range': bin_labels[i],
                    'n_samples': int(mask.sum()),
                    'rmse': float(np.sqrt(mean_squared_error(bin_y_true, bin_y_pred))),
                    'mae': float(mean_absolute_error(bin_y_true, bin_y_pred)),
                    'mean_error': float(np.mean(bin_errors)),
                    'under_prediction_pct': float(np.sum(bin_errors < 0) / mask.sum() * 100)
                })
        
        # Extreme error analysis
        abs_errors = np.abs(errors)
        extreme_threshold = np.percentile(abs_errors, 95)
        extreme_indices = abs_errors >= extreme_threshold
        
        return {
            'model_name': model_name,
            'bin_analysis': bin_analysis,
            'n_extreme_errors': int(extreme_indices.sum()),
            'extreme_error_threshold': float(extreme_threshold),
            'worst_underprediction': float(np.min(errors)),
            'worst_overprediction': float(np.max(errors))
        }
    
    def save_report(self, filepath: str = None):
        """
        Save comparison report to JSON file
        
        Args:
            filepath: Path to save report (default: results/model_comparison.json)
        """
        if filepath is None:
            filepath = os.path.join(config.RESULTS_DIR, 'model_comparison.json')
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'n_models': len(self.comparison_results),
            'models': self.comparison_results,
            'ranking': self.get_comparison_dataframe()['model_name'].tolist()
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report saved to {filepath}")
    
    def print_summary(self):
        """Print summary of model comparison"""
        df = self.get_comparison_dataframe()
        
        if df.empty:
            print("No models to compare")
            return
        
        print("\n" + "="*80)
        print("MODEL COMPARISON SUMMARY")
        print("="*80)
        
        # Header
        print(f"\n{'Rank':<6}{'Model':<25}{'RMSE':<10}{'MAE':<10}{'R²':<10}{'Asym Score':<12}")
        print("-"*80)
        
        for _, row in df.iterrows():
            print(f"{row['rank']:<6}{row['model_name']:<25}{row['rmse']:<10.2f}"
                  f"{row['mae']:<10.2f}{row['r2']:<10.4f}{row['asymmetric_score']:<12.2f}")
        
        # Best model
        best = df.iloc[0]
        print(f"\n{'='*80}")
        print(f"🏆 BEST MODEL: {best['model_name']}")
        print(f"   RMSE: {best['rmse']:.2f} | MAE: {best['mae']:.2f} | R²: {best['r2']:.4f}")
        print("="*80 + "\n")


def compare_saved_models(model_paths: Dict[str, str],
                         X_test: np.ndarray,
                         y_test: np.ndarray) -> ModelComparisonReport:
    """
    Compare multiple saved models
    
    Args:
        model_paths: Dictionary of {model_name: model_path}
        X_test: Test features
        y_test: Test targets
        
    Returns:
        ModelComparisonReport with all models compared
    """
    import pickle
    import tensorflow as tf
    
    report = ModelComparisonReport()
    
    for model_name, model_path in model_paths.items():
        try:
            # Determine model type and load
            if model_path.endswith('.h5'):
                model = tf.keras.models.load_model(model_path)
                y_pred = model.predict(X_test).flatten()
            elif model_path.endswith('.pkl'):
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                y_pred = model.predict(X_test)
            else:
                logger.warning(f"Unknown model format: {model_path}")
                continue
            
            # Get model size
            model_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
            
            report.add_model_results(
                model_name=model_name,
                y_true=y_test,
                y_pred=y_pred,
                model_size_mb=model_size
            )
            
        except Exception as e:
            logger.error(f"Error loading {model_name}: {e}")
    
    return report


if __name__ == "__main__":
    print("="*60)
    print("Testing Model Comparison Utility")
    print("="*60)
    
    # Generate synthetic data
    np.random.seed(42)
    y_true = np.random.randint(0, 150, size=100).astype(float)
    
    # Simulated model predictions with different accuracy levels
    y_pred_lstm = y_true + np.random.randn(100) * 15
    y_pred_xgb = y_true + np.random.randn(100) * 12
    y_pred_ensemble = y_true + np.random.randn(100) * 10
    y_pred_rf = y_true + np.random.randn(100) * 20
    
    # Clip negative predictions
    y_pred_lstm = np.maximum(y_pred_lstm, 0)
    y_pred_xgb = np.maximum(y_pred_xgb, 0)
    y_pred_ensemble = np.maximum(y_pred_ensemble, 0)
    y_pred_rf = np.maximum(y_pred_rf, 0)
    
    # Create comparison report
    report = ModelComparisonReport()
    
    report.add_model_results('LSTM', y_true, y_pred_lstm, 
                             training_time=300, inference_time=5, model_size_mb=50)
    report.add_model_results('XGBoost', y_true, y_pred_xgb,
                             training_time=60, inference_time=1, model_size_mb=10)
    report.add_model_results('Stacking Ensemble', y_true, y_pred_ensemble,
                             training_time=180, inference_time=3, model_size_mb=30)
    report.add_model_results('Random Forest', y_true, y_pred_rf,
                             training_time=45, inference_time=2, model_size_mb=15)
    
    # Print summary
    report.print_summary()
    
    # Statistical comparison
    print("\n--- Statistical Comparison ---")
    comparison = report.statistical_comparison('Stacking Ensemble', 'LSTM')
    print(f"Ensemble vs LSTM: Winner={comparison['winner']}, "
          f"Improvement={comparison['rmse_improvement']:.2f}, "
          f"Significant={comparison['significant_at_95']}")
    
    # Find best model with constraints
    print("\n--- Best Model Selection ---")
    best = report.find_best_model(metric='rmse')
    print(f"Best by RMSE: {best['model_name']}")
    
    best_constrained = report.find_best_model(
        metric='rmse', 
        constraints={'inference_time_ms': 3}
    )
    print(f"Best by RMSE (inference < 3ms): {best_constrained.get('model_name', 'None')}")
    
    # Error analysis
    print("\n--- Error Analysis ---")
    error_analysis = report.generate_error_analysis('Stacking Ensemble')
    print(f"Error analysis for {error_analysis['model_name']}:")
    for bin_info in error_analysis['bin_analysis']:
        print(f"  RUL {bin_info['rul_range']}: RMSE={bin_info['rmse']:.2f}, "
              f"Under-pred: {bin_info['under_prediction_pct']:.1f}%")
    
    # Save report
    report.save_report()
    
    print("\n" + "="*60)
    print("Model comparison test complete!")
    print("="*60)

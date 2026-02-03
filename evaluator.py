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


class PerformanceBenchmark:
    """
    Comprehensive benchmarking for model inference performance
    Measures latency, throughput, and memory usage
    """
    
    def __init__(self):
        """Initialize performance benchmark"""
        self.benchmark_results = {}
        logger.info("Initialized PerformanceBenchmark")
    
    def benchmark_inference(self,
                           model,
                           X_test: np.ndarray,
                           n_runs: int = 100,
                           warmup_runs: int = 10) -> Dict:
        """
        Benchmark model inference latency
        
        Args:
            model: Trained model with predict() method
            X_test: Test data for inference
            n_runs: Number of inference runs for timing
            warmup_runs: Number of warmup runs before timing
            
        Returns:
            Dictionary with latency statistics
        """
        import time
        
        logger.info(f"Benchmarking inference latency ({n_runs} runs)...")
        
        # Warmup runs
        for _ in range(warmup_runs):
            _ = model.predict(X_test[:1])
        
        # Single sample timing
        single_times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = model.predict(X_test[:1])
            single_times.append((time.perf_counter() - start) * 1000)  # ms
        
        # Batch timing
        batch_times = []
        batch_size = min(32, len(X_test))
        for _ in range(n_runs // 5):
            start = time.perf_counter()
            _ = model.predict(X_test[:batch_size])
            batch_times.append((time.perf_counter() - start) * 1000)  # ms
        
        # Full dataset timing
        full_start = time.perf_counter()
        _ = model.predict(X_test)
        full_time = (time.perf_counter() - full_start) * 1000
        
        results = {
            'single_sample': {
                'mean_ms': float(np.mean(single_times)),
                'std_ms': float(np.std(single_times)),
                'min_ms': float(np.min(single_times)),
                'max_ms': float(np.max(single_times)),
                'p50_ms': float(np.percentile(single_times, 50)),
                'p95_ms': float(np.percentile(single_times, 95)),
                'p99_ms': float(np.percentile(single_times, 99))
            },
            'batch_32': {
                'mean_ms': float(np.mean(batch_times)),
                'per_sample_ms': float(np.mean(batch_times) / batch_size)
            },
            'full_dataset': {
                'total_ms': float(full_time),
                'n_samples': len(X_test),
                'per_sample_ms': float(full_time / len(X_test))
            }
        }
        
        logger.info(f"  Single sample: {results['single_sample']['mean_ms']:.3f}ms (p95: {results['single_sample']['p95_ms']:.3f}ms)")
        logger.info(f"  Batch (32): {results['batch_32']['per_sample_ms']:.3f}ms per sample")
        
        return results
    
    def benchmark_throughput(self,
                            model,
                            X_test: np.ndarray,
                            duration_seconds: float = 5.0) -> Dict:
        """
        Benchmark model throughput (predictions per second)
        
        Args:
            model: Trained model
            X_test: Test data
            duration_seconds: How long to run the benchmark
            
        Returns:
            Dictionary with throughput statistics
        """
        import time
        
        logger.info(f"Benchmarking throughput for {duration_seconds}s...")
        
        # Single sample throughput
        single_count = 0
        single_start = time.perf_counter()
        while (time.perf_counter() - single_start) < duration_seconds:
            _ = model.predict(X_test[:1])
            single_count += 1
        single_elapsed = time.perf_counter() - single_start
        single_throughput = single_count / single_elapsed
        
        # Batch throughput
        batch_size = min(32, len(X_test))
        batch_count = 0
        batch_start = time.perf_counter()
        while (time.perf_counter() - batch_start) < duration_seconds:
            _ = model.predict(X_test[:batch_size])
            batch_count += 1
        batch_elapsed = time.perf_counter() - batch_start
        batch_throughput = (batch_count * batch_size) / batch_elapsed
        
        results = {
            'single_sample_throughput': float(single_throughput),
            'batch_throughput': float(batch_throughput),
            'batch_size': batch_size,
            'duration_seconds': duration_seconds
        }
        
        logger.info(f"  Single sample: {single_throughput:.1f} pred/sec")
        logger.info(f"  Batch ({batch_size}): {batch_throughput:.1f} pred/sec")
        
        return results
    
    def memory_profiling(self, model, X_test: np.ndarray) -> Dict:
        """
        Profile memory usage during inference
        
        Args:
            model: Trained model
            X_test: Test data
            
        Returns:
            Dictionary with memory statistics
        """
        import sys
        
        logger.info("Profiling memory usage...")
        
        # Model size estimate
        try:
            import pickle
            import io
            buffer = io.BytesIO()
            pickle.dump(model, buffer)
            model_size_bytes = buffer.tell()
        except Exception:
            model_size_bytes = sys.getsizeof(model)
        
        # Input data size
        input_size_bytes = X_test.nbytes
        
        # Output size
        output = model.predict(X_test[:100])
        output_size_per_sample = sys.getsizeof(output) / 100
        
        results = {
            'model_size_mb': float(model_size_bytes / (1024 * 1024)),
            'input_size_mb': float(input_size_bytes / (1024 * 1024)),
            'output_size_per_sample_kb': float(output_size_per_sample / 1024),
            'estimated_batch_memory_mb': float((input_size_bytes + model_size_bytes) / (1024 * 1024))
        }
        
        logger.info(f"  Model size: {results['model_size_mb']:.2f} MB")
        logger.info(f"  Input size: {results['input_size_mb']:.2f} MB")
        
        return results
    
    def generate_benchmark_report(self,
                                  model,
                                  X_test: np.ndarray,
                                  model_name: str = 'Model') -> Dict:
        """
        Generate comprehensive benchmark report
        
        Args:
            model: Trained model
            X_test: Test data
            model_name: Name of the model
            
        Returns:
            Complete benchmark report dictionary
        """
        logger.info(f"Generating benchmark report for {model_name}...")
        
        latency = self.benchmark_inference(model, X_test)
        throughput = self.benchmark_throughput(model, X_test, duration_seconds=3.0)
        memory = self.memory_profiling(model, X_test)
        
        report = {
            'model_name': model_name,
            'test_samples': len(X_test),
            'latency': latency,
            'throughput': throughput,
            'memory': memory,
            'summary': {
                'avg_latency_ms': latency['single_sample']['mean_ms'],
                'p95_latency_ms': latency['single_sample']['p95_ms'],
                'throughput_per_sec': throughput['batch_throughput'],
                'model_size_mb': memory['model_size_mb']
            }
        }
        
        self.benchmark_results[model_name] = report
        
        return report
    
    def compare_benchmarks(self, 
                          models: Dict[str, Any],
                          X_test: np.ndarray) -> pd.DataFrame:
        """
        Compare benchmarks across multiple models
        
        Args:
            models: Dictionary of {model_name: model}
            X_test: Test data
            
        Returns:
            DataFrame with comparison
        """
        results = []
        
        for name, model in models.items():
            report = self.generate_benchmark_report(model, X_test, name)
            results.append({
                'model': name,
                'latency_ms': report['summary']['avg_latency_ms'],
                'p95_latency_ms': report['summary']['p95_latency_ms'],
                'throughput_per_sec': report['summary']['throughput_per_sec'],
                'model_size_mb': report['summary']['model_size_mb']
            })
        
        return pd.DataFrame(results).sort_values('latency_ms')
    
    def print_report(self, model_name: str = None):
        """Print formatted benchmark report"""
        if model_name:
            reports = {model_name: self.benchmark_results.get(model_name, {})}
        else:
            reports = self.benchmark_results
        
        for name, report in reports.items():
            if not report:
                continue
            
            print("\n" + "="*60)
            print(f"PERFORMANCE BENCHMARK: {name}")
            print("="*60)
            print(f"\n📊 LATENCY:")
            print(f"  Single sample:  {report['latency']['single_sample']['mean_ms']:.3f}ms avg")
            print(f"  P95 latency:    {report['latency']['single_sample']['p95_ms']:.3f}ms")
            print(f"  P99 latency:    {report['latency']['single_sample']['p99_ms']:.3f}ms")
            print(f"\n⚡ THROUGHPUT:")
            print(f"  {report['throughput']['batch_throughput']:.1f} predictions/sec (batch)")
            print(f"\n💾 MEMORY:")
            print(f"  Model size: {report['memory']['model_size_mb']:.2f} MB")
            print("="*60)


class CrossValidationPipeline:
    """
    Cross-validation pipeline for RUL prediction models
    Supports engine-aware and time-series validation strategies
    """
    
    def __init__(self, n_splits: int = 5):
        """
        Initialize cross-validation pipeline
        
        Args:
            n_splits: Number of CV splits
        """
        self.n_splits = n_splits
        self.cv_results = []
        logger.info(f"Initialized CrossValidationPipeline (n_splits={n_splits})")
    
    def stratified_engine_cv(self,
                             df: pd.DataFrame,
                             feature_cols: List[str],
                             target_col: str = 'RUL',
                             model_fn: Callable = None) -> Dict:
        """
        Perform engine-aware cross-validation
        
        Ensures engines are not split across train/val sets
        
        Args:
            df: DataFrame with unit_id column
            feature_cols: Feature columns
            target_col: Target column
            model_fn: Function that returns trained model
            
        Returns:
            Cross-validation results
        """
        from sklearn.model_selection import GroupKFold
        from sklearn.ensemble import RandomForestRegressor
        
        logger.info("Performing stratified engine cross-validation...")
        
        # Get unique engines
        engine_ids = df['unit_id'].unique()
        groups = df['unit_id'].values
        
        # Prepare data
        X = df[feature_cols].values
        y = df[target_col].values
        
        # Create group-based CV
        gkf = GroupKFold(n_splits=min(self.n_splits, len(engine_ids)))
        
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Get engine counts
            train_engines = df.iloc[train_idx]['unit_id'].nunique()
            val_engines = df.iloc[val_idx]['unit_id'].nunique()
            
            # Train model
            if model_fn:
                model = model_fn()
            else:
                model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
            
            model.fit(X_train, y_train)
            
            # Predict and evaluate
            y_pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            
            fold_results.append({
                'fold': fold + 1,
                'train_samples': len(train_idx),
                'val_samples': len(val_idx),
                'train_engines': train_engines,
                'val_engines': val_engines,
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            })
            
            logger.info(f"  Fold {fold+1}: RMSE={rmse:.2f}, Engines: train={train_engines}, val={val_engines}")
        
        # Aggregate results
        cv_results = {
            'strategy': 'stratified_engine',
            'n_splits': len(fold_results),
            'folds': fold_results,
            'mean_rmse': float(np.mean([f['rmse'] for f in fold_results])),
            'std_rmse': float(np.std([f['rmse'] for f in fold_results])),
            'mean_mae': float(np.mean([f['mae'] for f in fold_results])),
            'mean_r2': float(np.mean([f['r2'] for f in fold_results]))
        }
        
        self.cv_results.append(cv_results)
        
        logger.info(f"Engine CV: RMSE = {cv_results['mean_rmse']:.2f} ± {cv_results['std_rmse']:.2f}")
        
        return cv_results
    
    def time_series_cv(self,
                       df: pd.DataFrame,
                       feature_cols: List[str],
                       target_col: str = 'RUL',
                       model_fn: Callable = None) -> Dict:
        """
        Time-respecting cross-validation
        
        Ensures training data always precedes validation data
        
        Args:
            df: DataFrame sorted by time
            feature_cols: Feature columns
            target_col: Target column
            model_fn: Function that returns trained model
            
        Returns:
            Cross-validation results
        """
        from sklearn.ensemble import RandomForestRegressor
        
        logger.info("Performing time-series cross-validation...")
        
        X = df[feature_cols].values
        y = df[target_col].values
        
        # Create time series splits
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model
            if model_fn:
                model = model_fn()
            else:
                model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
            
            model.fit(X_train, y_train)
            
            # Predict and evaluate
            y_pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            
            fold_results.append({
                'fold': fold + 1,
                'train_samples': len(train_idx),
                'val_samples': len(val_idx),
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            })
            
            logger.info(f"  Fold {fold+1}: RMSE={rmse:.2f}, Train={len(train_idx)}, Val={len(val_idx)}")
        
        # Aggregate results
        cv_results = {
            'strategy': 'time_series',
            'n_splits': len(fold_results),
            'folds': fold_results,
            'mean_rmse': float(np.mean([f['rmse'] for f in fold_results])),
            'std_rmse': float(np.std([f['rmse'] for f in fold_results])),
            'mean_mae': float(np.mean([f['mae'] for f in fold_results])),
            'mean_r2': float(np.mean([f['r2'] for f in fold_results]))
        }
        
        self.cv_results.append(cv_results)
        
        logger.info(f"Time-series CV: RMSE = {cv_results['mean_rmse']:.2f} ± {cv_results['std_rmse']:.2f}")
        
        return cv_results
    
    def aggregate_cv_results(self) -> Dict:
        """
        Aggregate and compare all CV results
        
        Returns:
            Aggregated CV comparison
        """
        if not self.cv_results:
            return {'status': 'no_results'}
        
        comparison = []
        for result in self.cv_results:
            comparison.append({
                'strategy': result['strategy'],
                'mean_rmse': result['mean_rmse'],
                'std_rmse': result['std_rmse'],
                'mean_r2': result['mean_r2']
            })
        
        return {
            'total_evaluations': len(self.cv_results),
            'comparison': comparison,
            'best_strategy': min(comparison, key=lambda x: x['mean_rmse'])['strategy']
        }
    
    def get_cv_summary(self) -> str:
        """Generate formatted CV summary report"""
        lines = [
            "=" * 60,
            "CROSS-VALIDATION SUMMARY",
            "=" * 60,
            ""
        ]
        
        for i, result in enumerate(self.cv_results):
            lines.extend([
                f"Strategy: {result['strategy']}",
                f"  Folds: {result['n_splits']}",
                f"  RMSE: {result['mean_rmse']:.2f} ± {result['std_rmse']:.2f}",
                f"  MAE:  {result['mean_mae']:.2f}",
                f"  R²:   {result['mean_r2']:.4f}",
                ""
            ])
        
        if self.cv_results:
            best = min(self.cv_results, key=lambda x: x['mean_rmse'])
            lines.append(f"Best Strategy: {best['strategy']} (RMSE: {best['mean_rmse']:.2f})")
        
        lines.append("=" * 60)
        
        return '\n'.join(lines)


class PerformanceProfiler:
    """
    Performance profiling for ML inference
    Tracks memory, CPU, and timing metrics
    """
    
    def __init__(self):
        """Initialize performance profiler"""
        self.profiles = []
        self.current_profile = None
        self.recommendations = []
        logger.info("Initialized PerformanceProfiler")
    
    def start_profile(self, name: str):
        """Start a new profile session"""
        import time
        
        self.current_profile = {
            'name': name,
            'start_time': time.time(),
            'start_memory': self._get_memory_usage(),
            'checkpoints': [],
            'metrics': {}
        }
    
    def checkpoint(self, label: str):
        """Add a checkpoint to current profile"""
        import time
        
        if not self.current_profile:
            return
        
        checkpoint = {
            'label': label,
            'time': time.time(),
            'elapsed': time.time() - self.current_profile['start_time'],
            'memory': self._get_memory_usage()
        }
        
        self.current_profile['checkpoints'].append(checkpoint)
    
    def end_profile(self) -> Dict:
        """End current profile and analyze results"""
        import time
        
        if not self.current_profile:
            return {}
        
        end_time = time.time()
        end_memory = self._get_memory_usage()
        
        profile = self.current_profile.copy()
        profile['end_time'] = end_time
        profile['end_memory'] = end_memory
        profile['total_time'] = end_time - profile['start_time']
        profile['memory_delta'] = end_memory - profile['start_memory']
        profile['bottlenecks'] = self._identify_bottlenecks(profile)
        
        self.profiles.append(profile)
        self.current_profile = None
        
        return profile
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0
    
    def _identify_bottlenecks(self, profile: Dict) -> List[Dict]:
        """Identify performance bottlenecks"""
        bottlenecks = []
        checkpoints = profile.get('checkpoints', [])
        
        if len(checkpoints) < 2:
            return bottlenecks
        
        step_times = []
        for i in range(1, len(checkpoints)):
            step_time = checkpoints[i]['elapsed'] - checkpoints[i-1]['elapsed']
            step_times.append({
                'step': f"{checkpoints[i-1]['label']} → {checkpoints[i]['label']}",
                'time': step_time
            })
        
        total = sum(s['time'] for s in step_times)
        
        for step in step_times:
            pct = (step['time'] / total * 100) if total > 0 else 0
            if pct > 50:
                bottlenecks.append({
                    'type': 'time',
                    'description': f"{step['step']} takes {pct:.1f}% of total time",
                    'severity': 'high' if pct > 70 else 'medium'
                })
        
        if profile.get('memory_delta', 0) > 100:
            bottlenecks.append({
                'type': 'memory',
                'description': f"Memory growth: {profile['memory_delta']:.1f} MB",
                'severity': 'high' if profile['memory_delta'] > 500 else 'medium'
            })
        
        return bottlenecks
    
    def profile_inference(self, model, data: np.ndarray) -> Dict:
        """Profile model inference"""
        import time
        
        n_samples = len(data)
        
        self.start_profile('inference')
        self.checkpoint('start')
        
        predictions = model.predict(data)
        self.checkpoint('predict')
        
        profile = self.end_profile()
        
        profile['metrics'] = {
            'samples': n_samples,
            'throughput': n_samples / profile['total_time'] if profile['total_time'] > 0 else 0,
            'latency_ms': profile['total_time'] * 1000 / n_samples if n_samples > 0 else 0
        }
        
        return profile
    
    def generate_recommendations(self) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        if not self.profiles:
            return ["Run some profiles to generate recommendations"]
        
        avg_time = np.mean([p['total_time'] for p in self.profiles])
        avg_memory = np.mean([p.get('memory_delta', 0) for p in self.profiles])
        
        if avg_time > 1.0:
            recommendations.append("Consider batch processing or model optimization")
        
        if avg_memory > 200:
            recommendations.append("High memory usage - consider smaller batch sizes")
        
        self.recommendations = recommendations
        return recommendations
    
    def get_profile_summary(self) -> str:
        """Generate profile summary"""
        lines = [
            "=" * 60,
            "PERFORMANCE PROFILE SUMMARY",
            "=" * 60,
            f"Total Profiles: {len(self.profiles)}",
            ""
        ]
        
        if self.profiles:
            avg_time = np.mean([p['total_time'] for p in self.profiles])
            lines.append(f"Average Time: {avg_time*1000:.1f} ms")
        
        recommendations = self.generate_recommendations()
        if recommendations:
            lines.extend(["", "Recommendations:"])
            for rec in recommendations:
                lines.append(f"  • {rec}")
        
        lines.append("=" * 60)
        
        return '\n'.join(lines)


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

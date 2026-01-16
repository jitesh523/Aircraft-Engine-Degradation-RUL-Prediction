"""
Model Monitoring Module
Detects data drift and model performance degradation over time
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
import json
import os
from datetime import datetime
import config
from utils import setup_logging

logger = setup_logging(__name__)


class ModelMonitor:
    """
    Monitors model performance and detects data/concept drift
    """
    
    def __init__(self, baseline_metrics: Optional[Dict] = None):
        """
        Initialize model monitor
        
        Args:
            baseline_metrics: Baseline performance metrics to compare against
        """
        self.baseline_metrics = baseline_metrics or {}
        self.drift_history = []
    
    def calculate_psi(self, expected: np.ndarray, actual: np.ndarray, 
                      bins: int = 10) -> float:
        """
        Calculate Population Stability Index (PSI) for data drift detection
        
        PSI < 0.1: No significant drift
        0.1 <= PSI < 0.2: Moderate drift
        PSI >= 0.2: Significant drift
        
        Args:
            expected: Baseline distribution (training data)
            actual: Current distribution (new data)
            bins: Number of bins for discretization
            
        Returns:
            PSI value
        """
        # Remove NaN and infinite values
        expected = expected[np.isfinite(expected)]
        actual = actual[np.isfinite(actual)]
        
        if len(expected) == 0 or len(actual) == 0:
            return np.nan
        
        # Create bins based on expected distribution
        breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf
        
        # Calculate distributions
        expected_percents = np.histogram(expected, bins=breakpoints)[0] / len(expected)
        actual_percents = np.histogram(actual, bins=breakpoints)[0] / len(actual)
        
        # Avoid division by zero
        expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
        actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)
        
        # Calculate PSI
        psi = np.sum((actual_percents - expected_percents) * 
                     np.log(actual_percents / expected_percents))
        
        return psi
    
    def kolmogorov_smirnov_test(self, baseline: np.ndarray, 
                               current: np.ndarray,
                               alpha: float = 0.05) -> Tuple[bool, float, float]:
        """
        Perform Kolmogorov-Smirnov test to detect distribution shift
        
        Args:
            baseline: Baseline distribution
            current: Current distribution
            alpha: Significance level
            
        Returns:
            (drift_detected, statistic, p_value)
        """
        # Remove NaN values
        baseline = baseline[np.isfinite(baseline)]
        current = current[np.isfinite(current)]
        
        if len(baseline) == 0 or len(current) == 0:
            return False, np.nan, np.nan
        
        # Perform KS test
        statistic, p_value = stats.ks_2samp(baseline, current)
        
        # Drift detected if p-value < alpha
        drift_detected = p_value < alpha
        
        return drift_detected, statistic, p_value
    
    def detect_feature_drift(self, baseline_data: pd.DataFrame,
                            current_data: pd.DataFrame,
                            feature_cols: List[str],
                            method: str = 'psi') -> Dict[str, any]:
        """
        Detect drift in feature distributions
        
        Args:
            baseline_data: Training/baseline data
            current_data: New/current data
            feature_cols: List of feature columns to check
            method: Drift detection method ('psi' or 'ks')
            
        Returns:
            Dictionary with drift results per feature
        """
        logger.info(f"Detecting feature drift using {method} method...")
        
        drift_results = {}
        
        for feature in feature_cols:
            if feature not in baseline_data.columns or feature not in current_data.columns:
                continue
            
            baseline_values = baseline_data[feature].values
            current_values = current_data[feature].values
            
            if method == 'psi':
                psi = self.calculate_psi(baseline_values, current_values)
                drift_level = 'none' if psi < 0.1 else ('moderate' if psi < 0.2 else 'significant')
                
                drift_results[feature] = {
                    'method': 'PSI',
                    'score': psi,
                    'drift_level': drift_level,
                    'drift_detected': psi >= 0.1
                }
            
            elif method == 'ks':
                drift_detected, statistic, p_value = self.kolmogorov_smirnov_test(
                    baseline_values, current_values
                )
                
                drift_results[feature] = {
                    'method': 'KS Test',
                    'statistic': statistic,
                    'p_value': p_value,
                    'drift_detected': drift_detected
                }
        
        # Summary
        total_features = len(drift_results)
        drifted_features = sum(1 for r in drift_results.values() if r['drift_detected'])
        
        logger.info(f"Drift detected in {drifted_features}/{total_features} features")
        
        return {
            'individual_features': drift_results,
            'summary': {
                'total_features': total_features,
                'drifted_features': drifted_features,
                'drift_percentage': (drifted_features / total_features * 100) if total_features > 0 else 0
            }
        }
    
    def monitor_performance(self, y_true: np.ndarray, 
                           y_pred: np.ndarray,
                           metric_name: str = 'RMSE') -> Dict[str, any]:
        """
        Monitor model performance and compare with baseline
        
        Args:
            y_true: True values
            y_pred: Predicted values
            metric_name: Name of metric to track
            
        Returns:
            Performance monitoring results
        """
        from evaluator import Evaluator
        
        evaluator = Evaluator()
        
        # Calculate current metrics
        current_rmse = evaluator.calculate_rmse(y_true, y_pred)
        current_mae = evaluator.calculate_mae(y_true, y_pred)
        current_r2 = evaluator.calculate_r2(y_true, y_pred)
        
        current_metrics = {
            'RMSE': current_rmse,
            'MAE': current_mae,
            'R2': current_r2
        }
        
        # Compare with baseline if available
        comparison = {}
        if self.baseline_metrics:
            for metric, current_value in current_metrics.items():
                if metric in self.baseline_metrics:
                    baseline_value = self.baseline_metrics[metric]
                    
                    # Calculate degradation
                    if metric == 'R2':
                        # For R2, lower is worse
                        degradation = ((baseline_value - current_value) / baseline_value) * 100
                    else:
                        # For RMSE/MAE, higher is worse
                        degradation = ((current_value - baseline_value) / baseline_value) * 100
                    
                    comparison[metric] = {
                        'baseline': baseline_value,
                        'current': current_value,
                        'degradation_pct': degradation,
                        'alert': degradation > 10  # Alert if >10% degradation
                    }
        
        return {
            'current_metrics': current_metrics,
            'baseline_comparison': comparison,
            'timestamp': datetime.now().isoformat()
        }
    
    def generate_drift_report(self, baseline_data: pd.DataFrame,
                             current_data: pd.DataFrame,
                             y_true: Optional[np.ndarray] = None,
                             y_pred: Optional[np.ndarray] = None,
                             feature_cols: Optional[List[str]] = None) -> Dict[str, any]:
        """
        Generate comprehensive drift monitoring report
        
        Args:
            baseline_data: Training data
            current_data: Current data
            y_true: True labels (if available)
            y_pred: Predictions (if available)
            feature_cols: Feature columns to monitor
            
        Returns:
            Comprehensive monitoring report
        """
        logger.info("Generating drift monitoring report...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'baseline_size': len(baseline_data),
            'current_size': len(current_data)
        }
        
        # Feature drift detection
        if feature_cols:
            drift_results = self.detect_feature_drift(
                baseline_data, current_data, feature_cols, method='psi'
            )
            report['feature_drift'] = drift_results
        
        # Performance monitoring
        if y_true is not None and y_pred is not None:
            perf_results = self.monitor_performance(y_true, y_pred)
            report['performance'] = perf_results
        
        # Generate alerts
        alerts = []
        
        if 'feature_drift' in report:
            drift_pct = report['feature_drift']['summary']['drift_percentage']
            if drift_pct > 30:
                alerts.append({
                    'severity': 'high',
                    'message': f"Significant drift detected in {drift_pct:.1f}% of features"
                })
            elif drift_pct > 15:
                alerts.append({
                    'severity': 'medium',
                    'message': f"Moderate drift detected in {drift_pct:.1f}% of features"
                })
        
        if 'performance' in report and 'baseline_comparison' in report['performance']:
            for metric, comp in report['performance']['baseline_comparison'].items():
                if comp.get('alert', False):
                    alerts.append({
                        'severity': 'high',
                        'message': f"{metric} degraded by {comp['degradation_pct']:.1f}%"
                    })
        
        report['alerts'] = alerts
        report['overall_status'] = 'degraded' if len(alerts) > 0 else 'healthy'
        
        # Save report
        self.drift_history.append(report)
        
        logger.info(f"Report generated. Status: {report['overall_status']}")
        logger.info(f"Alerts: {len(alerts)}")
        
        return report
    
    def save_report(self, report: Dict[str, any], filepath: str):
        """Save monitoring report to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report saved to {filepath}")
    
    def load_baseline_metrics(self, filepath: str):
        """Load baseline metrics from file"""
        with open(filepath, 'r') as f:
            self.baseline_metrics = json.load(f)
        
        logger.info(f"Loaded baseline metrics from {filepath}")


if __name__ == "__main__":
    # Test model monitor
    print("="*60)
    print("Testing Model Monitor")
    print("="*60)
    
    # Create sample data
    np.random.seed(42)
    baseline = np.random.randn(1000)
    current_no_drift = np.random.randn(1000)
    current_with_drift = np.random.randn(1000) + 0.5  # Shifted distribution
    
    monitor = ModelMonitor()
    
    # Test PSI
    psi_no_drift = monitor.calculate_psi(baseline, current_no_drift)
    psi_with_drift = monitor.calculate_psi(baseline, current_with_drift)
    
    print(f"\nPSI (no drift): {psi_no_drift:.4f}")
    print(f"PSI (with drift): {psi_with_drift:.4f}")
    
    # Test KS test
    drift_detected, stat, p_val = monitor.kolmogorov_smirnov_test(baseline, current_with_drift)
    print(f"\nKS Test: drift_detected={drift_detected}, statistic={stat:.4f}, p_value={p_val:.4f}")
    
    print("="*60)
    print("Model monitor tests complete!")
    print("="*60)

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


class ConceptDriftDetector:
    """
    Advanced concept drift detection for RUL prediction models
    Detects target drift, covariate shift, and provides action recommendations
    """
    
    def __init__(self, sensitivity: str = 'medium'):
        """
        Initialize concept drift detector
        
        Args:
            sensitivity: Detection sensitivity ('low', 'medium', 'high')
        """
        self.sensitivity = sensitivity
        self.thresholds = {
            'low': {'psi': 0.25, 'ks_alpha': 0.01},
            'medium': {'psi': 0.15, 'ks_alpha': 0.05},
            'high': {'psi': 0.1, 'ks_alpha': 0.1}
        }
        self.drift_history = []
        logger.info(f"Initialized ConceptDriftDetector (sensitivity: {sensitivity})")
    
    def detect_target_drift(self,
                           baseline_rul: np.ndarray,
                           current_rul: np.ndarray) -> Dict:
        """
        Detect drift in target (RUL) distribution
        
        Args:
            baseline_rul: Baseline RUL values
            current_rul: Current RUL values
            
        Returns:
            Target drift analysis
        """
        logger.info("Detecting target drift...")
        
        baseline_rul = baseline_rul[np.isfinite(baseline_rul)]
        current_rul = current_rul[np.isfinite(current_rul)]
        
        # Statistical tests
        ks_stat, ks_pval = stats.ks_2samp(baseline_rul, current_rul)
        
        # Distribution comparison
        baseline_stats = {
            'mean': float(np.mean(baseline_rul)),
            'std': float(np.std(baseline_rul)),
            'median': float(np.median(baseline_rul)),
            'min': float(np.min(baseline_rul)),
            'max': float(np.max(baseline_rul))
        }
        
        current_stats = {
            'mean': float(np.mean(current_rul)),
            'std': float(np.std(current_rul)),
            'median': float(np.median(current_rul)),
            'min': float(np.min(current_rul)),
            'max': float(np.max(current_rul))
        }
        
        # Calculate shift magnitude
        mean_shift = abs(current_stats['mean'] - baseline_stats['mean'])
        std_shift = abs(current_stats['std'] - baseline_stats['std'])
        
        threshold = self.thresholds[self.sensitivity]
        drift_detected = ks_pval < threshold['ks_alpha']
        
        result = {
            'drift_detected': drift_detected,
            'ks_statistic': float(ks_stat),
            'ks_pvalue': float(ks_pval),
            'baseline_stats': baseline_stats,
            'current_stats': current_stats,
            'mean_shift': float(mean_shift),
            'std_shift': float(std_shift),
            'significance_level': threshold['ks_alpha']
        }
        
        if drift_detected:
            logger.warning(f"Target drift detected (p={ks_pval:.4f})")
        else:
            logger.info("No significant target drift detected")
        
        return result
    
    def detect_covariate_shift(self,
                               baseline_features: pd.DataFrame,
                               current_features: pd.DataFrame,
                               feature_cols: List[str]) -> Dict:
        """
        Detect covariate (input feature) shift
        
        Args:
            baseline_features: Baseline feature DataFrame
            current_features: Current feature DataFrame
            feature_cols: Feature columns to analyze
            
        Returns:
            Covariate shift analysis
        """
        logger.info(f"Detecting covariate shift in {len(feature_cols)} features...")
        
        threshold = self.thresholds[self.sensitivity]
        
        feature_drifts = {}
        drifted_count = 0
        
        for col in feature_cols:
            if col not in baseline_features.columns or col not in current_features.columns:
                continue
            
            baseline_vals = baseline_features[col].dropna().values
            current_vals = current_features[col].dropna().values
            
            if len(baseline_vals) == 0 or len(current_vals) == 0:
                continue
            
            # KS test
            ks_stat, ks_pval = stats.ks_2samp(baseline_vals, current_vals)
            drift_detected = ks_pval < threshold['ks_alpha']
            
            if drift_detected:
                drifted_count += 1
            
            feature_drifts[col] = {
                'drift_detected': drift_detected,
                'ks_statistic': float(ks_stat),
                'ks_pvalue': float(ks_pval),
                'baseline_mean': float(np.mean(baseline_vals)),
                'current_mean': float(np.mean(current_vals))
            }
        
        drift_percentage = (drifted_count / len(feature_drifts) * 100) if feature_drifts else 0
        
        result = {
            'drift_detected': drifted_count > 0,
            'drifted_feature_count': drifted_count,
            'total_features': len(feature_drifts),
            'drift_percentage': float(drift_percentage),
            'feature_details': feature_drifts,
            'top_drifted': sorted(
                [(k, v['ks_statistic']) for k, v in feature_drifts.items() if v['drift_detected']],
                key=lambda x: x[1], reverse=True
            )[:10]
        }
        
        logger.info(f"Covariate shift: {drifted_count}/{len(feature_drifts)} features drifted")
        
        return result
    
    def calculate_drift_severity(self,
                                 target_drift: Dict,
                                 covariate_shift: Dict) -> Dict:
        """
        Calculate overall drift severity score
        
        Args:
            target_drift: Target drift results
            covariate_shift: Covariate shift results
            
        Returns:
            Drift severity assessment
        """
        logger.info("Calculating drift severity...")
        
        severity_score = 0.0
        factors = []
        
        # Target drift contribution (0-50 points)
        if target_drift.get('drift_detected'):
            mean_shift_pct = (target_drift['mean_shift'] / 
                            (target_drift['baseline_stats']['mean'] + 1e-10)) * 100
            target_score = min(50, mean_shift_pct * 2)
            severity_score += target_score
            factors.append(f"Target mean shifted by {mean_shift_pct:.1f}%")
        
        # Covariate shift contribution (0-50 points)
        drift_pct = covariate_shift.get('drift_percentage', 0)
        covariate_score = min(50, drift_pct)
        severity_score += covariate_score
        
        if drift_pct > 0:
            factors.append(f"{drift_pct:.1f}% of features drifted")
        
        # Severity classification
        if severity_score >= 70:
            severity_level = 'critical'
        elif severity_score >= 40:
            severity_level = 'high'
        elif severity_score >= 20:
            severity_level = 'moderate'
        elif severity_score > 0:
            severity_level = 'low'
        else:
            severity_level = 'none'
        
        result = {
            'severity_score': float(severity_score),
            'severity_level': severity_level,
            'contributing_factors': factors,
            'requires_action': severity_level in ['critical', 'high']
        }
        
        logger.info(f"Drift severity: {severity_level} (score: {severity_score:.1f})")
        
        return result
    
    def recommend_actions(self, severity: Dict) -> List[Dict]:
        """
        Recommend actions based on drift severity
        
        Args:
            severity: Drift severity results
            
        Returns:
            List of recommended actions
        """
        logger.info("Generating action recommendations...")
        
        actions = []
        level = severity['severity_level']
        
        if level == 'critical':
            actions.extend([
                {
                    'priority': 'immediate',
                    'action': 'retrain_model',
                    'description': 'Immediately retrain model with recent data'
                },
                {
                    'priority': 'high',
                    'action': 'investigate_root_cause',
                    'description': 'Investigate data pipeline for anomalies'
                },
                {
                    'priority': 'high',
                    'action': 'recalibrate_sensors',
                    'description': 'Check sensor calibration status'
                }
            ])
        elif level == 'high':
            actions.extend([
                {
                    'priority': 'high',
                    'action': 'schedule_retraining',
                    'description': 'Schedule model retraining within 24 hours'
                },
                {
                    'priority': 'medium',
                    'action': 'increase_monitoring',
                    'description': 'Increase monitoring frequency'
                }
            ])
        elif level == 'moderate':
            actions.extend([
                {
                    'priority': 'medium',
                    'action': 'plan_retraining',
                    'description': 'Plan model retraining within 1 week'
                },
                {
                    'priority': 'low',
                    'action': 'collect_more_data',
                    'description': 'Collect additional labeled data'
                }
            ])
        elif level == 'low':
            actions.append({
                'priority': 'low',
                'action': 'continue_monitoring',
                'description': 'Continue routine monitoring'
            })
        else:
            actions.append({
                'priority': 'info',
                'action': 'no_action_required',
                'description': 'No drift detected, system healthy'
            })
        
        return actions
    
    def run_full_analysis(self,
                          baseline_features: pd.DataFrame,
                          current_features: pd.DataFrame,
                          baseline_rul: np.ndarray,
                          current_rul: np.ndarray,
                          feature_cols: List[str]) -> Dict:
        """
        Run complete drift analysis
        
        Args:
            baseline_features: Baseline feature DataFrame
            current_features: Current feature DataFrame
            baseline_rul: Baseline RUL values
            current_rul: Current RUL values
            feature_cols: Feature columns
            
        Returns:
            Complete drift analysis report
        """
        logger.info("Running full drift analysis...")
        
        # Detect target drift
        target_drift = self.detect_target_drift(baseline_rul, current_rul)
        
        # Detect covariate shift
        covariate_shift = self.detect_covariate_shift(
            baseline_features, current_features, feature_cols
        )
        
        # Calculate severity
        severity = self.calculate_drift_severity(target_drift, covariate_shift)
        
        # Get recommendations
        recommendations = self.recommend_actions(severity)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'sensitivity': self.sensitivity,
            'target_drift': target_drift,
            'covariate_shift': covariate_shift,
            'severity': severity,
            'recommendations': recommendations
        }
        
        self.drift_history.append(report)
        
        return report


class AlertManager:
    """
    Alert management for RUL predictions and drift detection
    Handles thresholds, notifications, and alert history
    """
    
    def __init__(self):
        """Initialize alert manager"""
        self.thresholds = {
            'critical_rul': 15,     # Immediate attention
            'warning_rul': 30,      # Schedule maintenance
            'caution_rul': 50,      # Monitor closely
            'drift_psi': 0.2,       # Significant drift
            'error_rate': 0.1       # High error rate
        }
        
        self.alert_history = []
        self.acknowledged_alerts = set()
        self.notification_channels = ['log']
        
        logger.info("Initialized AlertManager")
    
    def configure_thresholds(self, thresholds: Dict[str, float]):
        """
        Configure alert thresholds
        
        Args:
            thresholds: Dictionary of threshold values
        """
        self.thresholds.update(thresholds)
        logger.info(f"Updated thresholds: {thresholds}")
    
    def add_notification_channel(self, channel: str):
        """Add a notification channel"""
        if channel not in self.notification_channels:
            self.notification_channels.append(channel)
    
    def check_rul_alerts(self,
                         unit_id: int,
                         rul: float) -> Dict:
        """
        Check if RUL prediction triggers alerts
        
        Args:
            unit_id: Engine unit ID
            rul: Predicted RUL value
            
        Returns:
            Alert information if triggered
        """
        alert = None
        
        if rul <= self.thresholds['critical_rul']:
            alert = self._create_alert(
                alert_type='CRITICAL',
                source=f'Engine {unit_id}',
                message=f'RUL = {rul:.1f} cycles - IMMEDIATE ACTION REQUIRED',
                severity='critical',
                rul=rul
            )
        elif rul <= self.thresholds['warning_rul']:
            alert = self._create_alert(
                alert_type='WARNING',
                source=f'Engine {unit_id}',
                message=f'RUL = {rul:.1f} cycles - Schedule maintenance',
                severity='high',
                rul=rul
            )
        elif rul <= self.thresholds['caution_rul']:
            alert = self._create_alert(
                alert_type='CAUTION',
                source=f'Engine {unit_id}',
                message=f'RUL = {rul:.1f} cycles - Monitor closely',
                severity='medium',
                rul=rul
            )
        
        if alert:
            self._dispatch_alert(alert)
        
        return alert
    
    def check_drift_alert(self,
                          psi_score: float,
                          feature_name: str = 'prediction') -> Dict:
        """
        Check if drift triggers alert
        
        Args:
            psi_score: PSI drift score
            feature_name: Name of feature with drift
            
        Returns:
            Alert if triggered
        """
        if psi_score >= self.thresholds['drift_psi']:
            alert = self._create_alert(
                alert_type='DRIFT',
                source=f'Feature: {feature_name}',
                message=f'PSI = {psi_score:.4f} - Model drift detected',
                severity='high',
                psi=psi_score
            )
            self._dispatch_alert(alert)
            return alert
        
        return None
    
    def check_batch_alerts(self,
                           predictions: List[Tuple[int, float]]) -> List[Dict]:
        """
        Check batch of predictions for alerts
        
        Args:
            predictions: List of (unit_id, rul) tuples
            
        Returns:
            List of triggered alerts
        """
        alerts = []
        
        for unit_id, rul in predictions:
            alert = self.check_rul_alerts(unit_id, rul)
            if alert:
                alerts.append(alert)
        
        return alerts
    
    def _create_alert(self,
                      alert_type: str,
                      source: str,
                      message: str,
                      severity: str,
                      **kwargs) -> Dict:
        """Create alert record"""
        alert_id = f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.alert_history)}"
        
        alert = {
            'alert_id': alert_id,
            'type': alert_type,
            'source': source,
            'message': message,
            'severity': severity,
            'timestamp': datetime.now().isoformat(),
            'acknowledged': False,
            **kwargs
        }
        
        self.alert_history.append(alert)
        
        return alert
    
    def _dispatch_alert(self, alert: Dict):
        """Dispatch alert to notification channels"""
        for channel in self.notification_channels:
            if channel == 'log':
                if alert['severity'] == 'critical':
                    logger.critical(f"[{alert['type']}] {alert['message']}")
                elif alert['severity'] == 'high':
                    logger.warning(f"[{alert['type']}] {alert['message']}")
                else:
                    logger.info(f"[{alert['type']}] {alert['message']}")
    
    def acknowledge_alert(self, alert_id: str):
        """
        Acknowledge an alert
        
        Args:
            alert_id: Alert ID to acknowledge
        """
        self.acknowledged_alerts.add(alert_id)
        
        for alert in self.alert_history:
            if alert['alert_id'] == alert_id:
                alert['acknowledged'] = True
                alert['acknowledged_at'] = datetime.now().isoformat()
                break
        
        logger.info(f"Alert acknowledged: {alert_id}")
    
    def get_active_alerts(self, severity: str = None) -> List[Dict]:
        """
        Get active (unacknowledged) alerts
        
        Args:
            severity: Filter by severity level
            
        Returns:
            List of active alerts
        """
        active = [a for a in self.alert_history if not a['acknowledged']]
        
        if severity:
            active = [a for a in active if a['severity'] == severity]
        
        return active
    
    def get_alert_statistics(self) -> Dict:
        """Get alert statistics"""
        total = len(self.alert_history)
        acknowledged = len(self.acknowledged_alerts)
        
        by_severity = {}
        by_type = {}
        
        for alert in self.alert_history:
            sev = alert['severity']
            by_severity[sev] = by_severity.get(sev, 0) + 1
            
            atype = alert['type']
            by_type[atype] = by_type.get(atype, 0) + 1
        
        return {
            'total_alerts': total,
            'acknowledged': acknowledged,
            'active': total - acknowledged,
            'by_severity': by_severity,
            'by_type': by_type
        }
    
    def get_alert_summary(self) -> str:
        """Generate alert summary"""
        stats = self.get_alert_statistics()
        
        lines = [
            "=" * 60,
            "ALERT SUMMARY",
            "=" * 60,
            f"Total Alerts: {stats['total_alerts']}",
            f"Acknowledged: {stats['acknowledged']}",
            f"Active: {stats['active']}",
            "",
            "By Severity:"
        ]
        
        for sev, count in stats['by_severity'].items():
            lines.append(f"  {sev}: {count}")
        
        lines.extend(["", "By Type:"])
        for atype, count in stats['by_type'].items():
            lines.append(f"  {atype}: {count}")
        
        active = self.get_active_alerts()
        if active:
            lines.extend(["", "Active Alerts:"])
            for alert in active[:5]:
                lines.append(f"  [{alert['severity'].upper()}] {alert['message']}")
        
        lines.append("=" * 60)
        
        return '\n'.join(lines)


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

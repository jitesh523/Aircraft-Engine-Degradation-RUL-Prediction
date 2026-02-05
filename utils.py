"""
Utility functions for Aircraft Engine RUL Prediction System
Helper functions for data manipulation, sequence generation, and model I/O
"""

import numpy as np
import pandas as pd
import pickle
import json
import logging
import os
from datetime import datetime
from typing import Tuple, List, Dict, Any
import config

# ==================== Logging Setup ====================
def setup_logging(name: str = __name__) -> logging.Logger:
    """
    Set up logging configuration
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=getattr(logging, config.LOGGING_CONFIG['level']),
        format=config.LOGGING_CONFIG['format'],
        datefmt=config.LOGGING_CONFIG['datefmt']
    )
    return logging.getLogger(name)

logger = setup_logging()

# ==================== Sequence Generation ====================
def generate_sequences(data: pd.DataFrame, 
                       sequence_length: int,
                       feature_cols: List[str],
                       target_col: str = 'RUL') -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate time-series sequences for LSTM input
    
    For each engine unit, creates overlapping sequences of length 'sequence_length'
    from the available time cycles. Each sequence has shape (sequence_length, num_features).
    
    Args:
        data: DataFrame with columns [unit_id, time_cycles, features..., RUL]
        sequence_length: Number of time steps in each sequence
        feature_cols: List of feature column names to include
        target_col: Name of target column (RUL)
        
    Returns:
        X: Array of sequences, shape (num_sequences, sequence_length, num_features)
        y: Array of targets, shape (num_sequences,)
    """
    sequences = []
    targets = []
    
    # Group by unit_id to process each engine separately
    for unit_id, unit_data in data.groupby('unit_id'):
        # Sort by time to ensure chronological order
        unit_data = unit_data.sort_values('time_cycles')
        
        # Extract features and target
        features = unit_data[feature_cols].values
        rul_values = unit_data[target_col].values
        
        # Generate sequences
        num_cycles = len(unit_data)
        for i in range(sequence_length, num_cycles + 1):
            # Sequence: from (i - sequence_length) to i
            seq = features[i - sequence_length:i, :]
            target = rul_values[i - 1]  # RUL at the end of sequence
            
            sequences.append(seq)
            targets.append(target)
    
    X = np.array(sequences)
    y = np.array(targets)
    
    logger.info(f"Generated {len(sequences)} sequences with shape {X.shape}")
    return X, y


def generate_sequences_for_prediction(data: pd.DataFrame,
                                      sequence_length: int,
                                      feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate sequences for test data prediction
    
    For each test engine, creates ONE sequence from the last 'sequence_length' cycles
    to predict the final RUL.
    
    Args:
        data: Test DataFrame with columns [unit_id, time_cycles, features...]
        sequence_length: Number of time steps in each sequence
        feature_cols: List of feature column names
        
    Returns:
        X: Array of sequences, shape (num_engines, sequence_length, num_features)
        unit_ids: Array of corresponding unit IDs
    """
    sequences = []
    unit_ids = []
    
    for unit_id, unit_data in data.groupby('unit_id'):
        unit_data = unit_data.sort_values('time_cycles')
        features = unit_data[feature_cols].values
        
        # Take only the last 'sequence_length' cycles
        if len(unit_data) >= sequence_length:
            seq = features[-sequence_length:, :]
        else:
            # Pad with zeros if not enough cycles
            padding = np.zeros((sequence_length - len(features), len(feature_cols)))
            seq = np.vstack([padding, features])
        
        sequences.append(seq)
        unit_ids.append(unit_id)
    
    X = np.array(sequences)
    unit_ids = np.array(unit_ids)
    
    logger.info(f"Generated {len(sequences)} test sequences with shape {X.shape}")
    return X, unit_ids


# ==================== Model I/O ====================
def save_model(model: Any, filepath: str) -> None:
    """
    Save trained model to disk
    
    Args:
        model: Trained model object
        filepath: Path to save the model
    """
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # For Keras/TensorFlow models
    if hasattr(model, 'save'):
        model.save(filepath)
        logger.info(f"Keras model saved to {filepath}")
    # For sklearn models
    else:
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Sklearn model saved to {filepath}")


def load_model(filepath: str, is_keras: bool = True) -> Any:
    """
    Load trained model from disk
    
    Args:
        filepath: Path to the saved model
        is_keras: Whether the model is a Keras model
        
    Returns:
        Loaded model object
    """
    if is_keras:
        from tensorflow import keras
        model = keras.models.load_model(filepath)
        logger.info(f"Keras model loaded from {filepath}")
    else:
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Sklearn model loaded from {filepath}")
    
    return model


def save_scaler(scaler: Any, filepath: str) -> None:
    """
    Save trained scaler to disk
    
    Args:
        scaler: Fitted scaler object
        filepath: Path to save the scaler
    """
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f"Scaler saved to {filepath}")


def load_scaler(filepath: str) -> Any:
    """
    Load trained scaler from disk
    
    Args:
        filepath: Path to the saved scaler
        
    Returns:
        Loaded scaler object
    """
    with open(filepath, 'rb') as f:
        scaler = pickle.load(f)
    logger.info(f"Scaler loaded from {filepath}")
    return scaler


# ==================== Results I/O ====================
def save_results(results: Dict[str, Any], filepath: str) -> None:
    """
    Save evaluation results to JSON file
    
    Args:
        results: Dictionary of results
        filepath: Path to save the results
    """
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Convert numpy types to Python native types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    results_serializable = {k: convert_numpy(v) for k, v in results.items()}
    
    with open(filepath, 'w') as f:
        json.dump(results_serializable, f, indent=4)
    logger.info(f"Results saved to {filepath}")


def load_results(filepath: str) -> Dict[str, Any]:
    """
    Load evaluation results from JSON file
    
    Args:
        filepath: Path to the results file
        
    Returns:
        Dictionary of results
    """
    with open(filepath, 'r') as f:
        results = json.load(f)
    logger.info(f"Results loaded from {filepath}")
    return results


# ==================== Data Manipulation ====================
def add_remaining_useful_life(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate and add RUL column to training data
    
    RUL = max_cycle - current_cycle for each engine unit
    
    Args:
        df: DataFrame with unit_id and time_cycles columns
        
    Returns:
        DataFrame with added RUL column
    """
    # Get max cycle for each engine
    max_cycles = df.groupby('unit_id')['time_cycles'].max().reset_index()
    max_cycles.columns = ['unit_id', 'max_cycle']
    
    # Merge and calculate RUL
    df = df.merge(max_cycles, on='unit_id', how='left')
    df['RUL'] = df['max_cycle'] - df['time_cycles']
    df = df.drop('max_cycle', axis=1)
    
    # Apply RUL clipping if configured
    if config.USE_RUL_CLIPPING:
        df['RUL'] = df['RUL'].clip(upper=config.MAX_RUL)
    
    logger.info(f"Added RUL column. Range: {df['RUL'].min()} to {df['RUL'].max()}")
    return df


def calculate_rate_of_change(df: pd.DataFrame, columns: List[str], periods: int = 1) -> pd.DataFrame:
    """
    Calculate rate of change for specified columns
    
    Args:
        df: DataFrame with unit_id and time_cycles
        columns: List of column names to calculate rate of change
        periods: Number of periods to shift for rate calculation
        
    Returns:
        DataFrame with additional rate-of-change columns
    """
    for col in columns:
        rate_col = f'{col}_roc_{periods}'
        df[rate_col] = df.groupby('unit_id')[col].diff(periods)
        # Fill NaN values with 0 (first few cycles)
        df[rate_col] = df[rate_col].fillna(0)
    
    logger.info(f"Calculated rate of change for {len(columns)} columns")
    return df


def add_rolling_statistics(df: pd.DataFrame, 
                          columns: List[str], 
                          window_sizes: List[int],
                          stats: List[str] = ['mean', 'std']) -> pd.DataFrame:
    """
    Add rolling window statistics for specified columns
    
    Args:
        df: DataFrame with unit_id and time_cycles
        columns: List of column names to calculate statistics
        window_sizes: List of window sizes for rolling statistics
        stats: List of statistics to calculate ('mean', 'std', 'min', 'max')
        
    Returns:
        DataFrame with additional rolling statistic columns
    """
    for col in columns:
        for window in window_sizes:
            for stat in stats:
                new_col = f'{col}_rolling_{stat}_{window}'
                if stat == 'mean':
                    df[new_col] = df.groupby('unit_id')[col].rolling(window=window, min_periods=1).mean().reset_index(0, drop=True)
                elif stat == 'std':
                    df[new_col] = df.groupby('unit_id')[col].rolling(window=window, min_periods=1).std().reset_index(0, drop=True)
                elif stat == 'min':
                    df[new_col] = df.groupby('unit_id')[col].rolling(window=window, min_periods=1).min().reset_index(0, drop=True)
                elif stat == 'max':
                    df[new_col] = df.groupby('unit_id')[col].rolling(window=window, min_periods=1).max().reset_index(0, drop=True)
                # Fill NaN with original value
                df[new_col] = df[new_col].fillna(df[col])
    
    logger.info(f"Added rolling statistics for {len(columns)} columns with {len(window_sizes)} window sizes")
    return df


# ==================== Evaluation Helpers ====================
def asymmetric_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate asymmetric scoring function (NASA scoring metric)
    
    Penalizes late predictions (under-predicting RUL) more heavily than early predictions
    
    Args:
        y_true: True RUL values
        y_pred: Predicted RUL values
        
    Returns:
        Asymmetric score (lower is better)
    """
    diff = y_pred - y_true
    score = np.sum(
        np.where(diff < 0, 
                np.exp(-diff / config.ASYMMETRIC_LOSS_WEIGHTS['late_prediction_penalty']) - 1,
                np.exp(diff / config.ASYMMETRIC_LOSS_WEIGHTS['early_prediction_penalty']) - 1)
    )
    return score


# ==================== Prediction Explainability ====================
class PredictionExplainer:
    """
    Generates human-readable explanations for RUL predictions
    Helps operators understand why a particular RUL was predicted
    """
    
    def __init__(self, feature_names: List[str] = None):
        """
        Initialize prediction explainer
        
        Args:
            feature_names: List of feature names used in the model
        """
        self.feature_names = feature_names or []
        self.sensor_descriptions = {
            'sensor_2': 'LPC outlet temperature',
            'sensor_3': 'HPC outlet temperature',
            'sensor_4': 'LPT outlet temperature',
            'sensor_7': 'HPC outlet pressure',
            'sensor_8': 'Physical fan speed',
            'sensor_9': 'Physical core speed',
            'sensor_11': 'Static pressure at HPC outlet',
            'sensor_12': 'Fuel flow ratio',
            'sensor_13': 'Corrected fan speed',
            'sensor_14': 'Corrected core speed',
            'sensor_15': 'Bypass ratio',
            'sensor_17': 'Bleed enthalpy',
            'sensor_20': 'HPT coolant bleed',
            'sensor_21': 'LPT coolant bleed'
        }
        
        # Thresholds for concern levels
        self.rul_thresholds = {
            'critical': 30,
            'warning': 50,
            'caution': 80
        }
        
        logger.info("Initialized PredictionExplainer")
    
    def explain_prediction(self,
                          rul_predicted: float,
                          sensor_values: Dict[str, float] = None,
                          sensor_trends: Dict[str, str] = None) -> Dict:
        """
        Generate natural language explanation for a single RUL prediction
        
        Args:
            rul_predicted: Predicted RUL value
            sensor_values: Dictionary of current sensor values (optional)
            sensor_trends: Dictionary of sensor trends ('increasing', 'decreasing', 'stable')
            
        Returns:
            Dictionary with explanation components
        """
        # Determine health status
        if rul_predicted < self.rul_thresholds['critical']:
            status = 'CRITICAL'
            urgency = 'immediate'
            status_emoji = '🔴'
            summary = f"Engine requires immediate attention. Predicted remaining life is only {rul_predicted:.0f} cycles."
        elif rul_predicted < self.rul_thresholds['warning']:
            status = 'WARNING'
            urgency = 'urgent'
            status_emoji = '🟡'
            summary = f"Engine shows signs of degradation. Predicted remaining life is {rul_predicted:.0f} cycles."
        elif rul_predicted < self.rul_thresholds['caution']:
            status = 'CAUTION'
            urgency = 'moderate'
            status_emoji = '🟠'
            summary = f"Engine is showing early degradation signs. Predicted remaining life is {rul_predicted:.0f} cycles."
        else:
            status = 'HEALTHY'
            urgency = 'low'
            status_emoji = '🟢'
            summary = f"Engine is operating within normal parameters. Predicted remaining life is {rul_predicted:.0f} cycles."
        
        # Build detailed explanation
        explanation_parts = [summary]
        
        if sensor_trends:
            concerning_sensors = [s for s, t in sensor_trends.items() 
                                if t in ['increasing', 'decreasing'] and 'temp' in s.lower()]
            if concerning_sensors:
                explanation_parts.append(
                    f"Temperature sensors showing concerning trends: {', '.join(concerning_sensors)}"
                )
        
        # Recommendations based on status
        recommendations = self._get_recommendations(status, rul_predicted)
        
        return {
            'status': status,
            'status_emoji': status_emoji,
            'urgency_level': urgency,
            'rul_predicted': rul_predicted,
            'summary': summary,
            'detailed_explanation': ' '.join(explanation_parts),
            'recommendations': recommendations,
            'confidence_note': self._get_confidence_note(rul_predicted)
        }
    
    def _get_recommendations(self, status: str, rul: float) -> List[str]:
        """Generate action recommendations based on status"""
        if status == 'CRITICAL':
            return [
                "Ground aircraft immediately for inspection",
                "Initiate emergency maintenance protocol",
                "Review recent flight data for anomalies",
                "Notify flight operations and maintenance crew"
            ]
        elif status == 'WARNING':
            return [
                "Schedule maintenance within 48 hours",
                "Increase monitoring frequency",
                "Prepare replacement parts",
                "Consider reducing operational load"
            ]
        elif status == 'CAUTION':
            return [
                f"Plan maintenance within next {int(rul * 0.5)} cycles",
                "Include in next scheduled maintenance window",
                "Monitor sensor trends closely"
            ]
        else:
            return [
                "Continue normal operations",
                "Maintain regular monitoring schedule",
                "No immediate action required"
            ]
    
    def _get_confidence_note(self, rul: float) -> str:
        """Generate confidence note based on RUL value"""
        if rul < 20:
            return "High confidence - extensive degradation patterns detected"
        elif rul < 50:
            return "Moderate-high confidence - clear degradation trajectory"
        elif rul < 100:
            return "Moderate confidence - some uncertainty in exact RUL"
        else:
            return "Lower confidence - early stage prediction with larger margin"
    
    def identify_critical_sensors(self,
                                  sensor_data: pd.DataFrame,
                                  rul: float,
                                  n_top: int = 5) -> List[Dict]:
        """
        Identify sensors most likely responsible for degradation
        
        Args:
            sensor_data: DataFrame with sensor readings over time
            rul: Current RUL prediction
            n_top: Number of top sensors to return
            
        Returns:
            List of dictionaries with sensor analysis
        """
        critical_sensors = []
        
        sensor_cols = [c for c in sensor_data.columns if c.startswith('sensor_')]
        
        for col in sensor_cols:
            if col not in sensor_data.columns:
                continue
                
            values = sensor_data[col].values
            if len(values) < 2:
                continue
            
            # Calculate metrics
            recent_trend = (values[-1] - values[-10]) if len(values) >= 10 else (values[-1] - values[0])
            volatility = np.std(values[-20:]) if len(values) >= 20 else np.std(values)
            current_vs_mean = abs(values[-1] - np.mean(values)) / (np.std(values) + 1e-6)
            
            # Score the sensor (higher = more concerning)
            concern_score = abs(recent_trend) + volatility + current_vs_mean
            
            critical_sensors.append({
                'sensor': col,
                'description': self.sensor_descriptions.get(col, 'Unknown sensor'),
                'concern_score': float(concern_score),
                'trend': 'increasing' if recent_trend > 0 else 'decreasing',
                'trend_magnitude': abs(float(recent_trend)),
                'current_deviation': float(current_vs_mean)
            })
        
        # Sort by concern score and return top N
        critical_sensors.sort(key=lambda x: x['concern_score'], reverse=True)
        return critical_sensors[:n_top]
    
    def generate_report(self,
                       unit_id: int,
                       rul_predicted: float,
                       sensor_data: pd.DataFrame = None,
                       include_sensors: bool = True) -> str:
        """
        Generate comprehensive prediction report
        
        Args:
            unit_id: Engine unit ID
            rul_predicted: Predicted RUL value
            sensor_data: Historical sensor data for this engine
            include_sensors: Whether to include sensor analysis
            
        Returns:
            Formatted report string
        """
        explanation = self.explain_prediction(rul_predicted)
        
        report_lines = [
            "=" * 60,
            f"RUL PREDICTION REPORT - Engine Unit {unit_id}",
            "=" * 60,
            "",
            f"{explanation['status_emoji']} Status: {explanation['status']}",
            f"Predicted RUL: {rul_predicted:.0f} cycles",
            f"Urgency: {explanation['urgency_level'].upper()}",
            "",
            "SUMMARY:",
            explanation['summary'],
            "",
            f"Confidence: {explanation['confidence_note']}",
            "",
            "RECOMMENDATIONS:"
        ]
        
        for i, rec in enumerate(explanation['recommendations'], 1):
            report_lines.append(f"  {i}. {rec}")
        
        if include_sensors and sensor_data is not None and len(sensor_data) > 0:
            report_lines.extend([
                "",
                "CRITICAL SENSOR ANALYSIS:"
            ])
            
            critical = self.identify_critical_sensors(sensor_data, rul_predicted)
            for sensor in critical:
                report_lines.append(
                    f"  • {sensor['sensor']}: {sensor['description']} "
                    f"(trend: {sensor['trend']}, concern: {sensor['concern_score']:.2f})"
                )
        
        report_lines.extend([
            "",
            "=" * 60
        ])
        
        return '\n'.join(report_lines)


class ComprehensiveReportGenerator:
    """
    Generates comprehensive analysis reports for RUL predictions
    Supports multiple output formats (Markdown, JSON, HTML)
    """
    
    def __init__(self, output_dir: str = None):
        """
        Initialize report generator
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = output_dir or os.path.join(config.RESULTS_DIR, 'reports')
        os.makedirs(self.output_dir, exist_ok=True)
        self.report_data = {}
        logger.info(f"Initialized ComprehensiveReportGenerator (output: {self.output_dir})")
    
    def generate_fleet_summary(self,
                               predictions_df: pd.DataFrame,
                               rul_col: str = 'RUL_pred') -> Dict:
        """
        Generate fleet-wide summary statistics
        
        Args:
            predictions_df: DataFrame with predictions for all engines
            rul_col: Name of RUL prediction column
            
        Returns:
            Dictionary with fleet summary
        """
        logger.info("Generating fleet summary...")
        
        # Count by health status
        def get_status(rul):
            if rul < 30:
                return 'Critical'
            elif rul < 50:
                return 'Warning'
            elif rul < 80:
                return 'Caution'
            else:
                return 'Healthy'
        
        engine_ruls = predictions_df.groupby('unit_id')[rul_col].last()
        statuses = engine_ruls.apply(get_status)
        
        summary = {
            'total_engines': len(engine_ruls),
            'fleet_status': statuses.value_counts().to_dict(),
            'rul_statistics': {
                'mean': float(engine_ruls.mean()),
                'median': float(engine_ruls.median()),
                'min': float(engine_ruls.min()),
                'max': float(engine_ruls.max()),
                'std': float(engine_ruls.std())
            },
            'critical_engines': engine_ruls[engine_ruls < 30].index.tolist(),
            'warning_engines': engine_ruls[(engine_ruls >= 30) & (engine_ruls < 50)].index.tolist()
        }
        
        self.report_data['fleet_summary'] = summary
        return summary
    
    def generate_engine_details(self,
                                predictions_df: pd.DataFrame,
                                engine_ids: List[int] = None,
                                rul_col: str = 'RUL_pred') -> Dict:
        """
        Generate detailed report for specific engines
        
        Args:
            predictions_df: DataFrame with predictions
            engine_ids: List of engine IDs (default: critical engines)
            rul_col: RUL prediction column name
            
        Returns:
            Dictionary with engine details
        """
        if engine_ids is None:
            # Default to engines with lowest RUL
            engine_ruls = predictions_df.groupby('unit_id')[rul_col].last()
            engine_ids = engine_ruls.nsmallest(5).index.tolist()
        
        details = {}
        
        for engine_id in engine_ids:
            engine_data = predictions_df[predictions_df['unit_id'] == engine_id]
            
            if len(engine_data) == 0:
                continue
            
            # Calculate metrics
            current_rul = engine_data[rul_col].iloc[-1]
            rul_history = engine_data[rul_col].values
            
            # Estimate degradation rate
            if len(rul_history) >= 10:
                recent_rate = (rul_history[-10] - rul_history[-1]) / 10
            else:
                recent_rate = rul_history[0] - rul_history[-1] if len(rul_history) > 1 else 0
            
            details[engine_id] = {
                'current_rul': float(current_rul),
                'degradation_rate': float(recent_rate),
                'total_cycles': int(len(engine_data)),
                'status': 'Critical' if current_rul < 30 else 
                         'Warning' if current_rul < 50 else
                         'Caution' if current_rul < 80 else 'Healthy'
            }
        
        self.report_data['engine_details'] = details
        return details
    
    def generate_model_performance(self,
                                   y_true: np.ndarray,
                                   y_pred: np.ndarray,
                                   model_name: str = 'Model') -> Dict:
        """
        Generate model performance summary
        
        Args:
            y_true: True RUL values
            y_pred: Predicted RUL values
            model_name: Name of the model
            
        Returns:
            Dictionary with performance metrics
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        errors = y_pred - y_true
        
        performance = {
            'model_name': model_name,
            'metrics': {
                'rmse': float(rmse),
                'mae': float(mae),
                'r2': float(r2)
            },
            'error_distribution': {
                'mean_error': float(np.mean(errors)),
                'std_error': float(np.std(errors)),
                'max_overestimate': float(np.max(errors)),
                'max_underestimate': float(np.min(errors))
            },
            'prediction_quality': {
                'within_10_cycles': float((np.abs(errors) <= 10).mean() * 100),
                'within_20_cycles': float((np.abs(errors) <= 20).mean() * 100),
                'within_30_cycles': float((np.abs(errors) <= 30).mean() * 100)
            }
        }
        
        self.report_data['model_performance'] = performance
        return performance
    
    def export_markdown(self, filename: str = None) -> str:
        """
        Export report to Markdown format
        
        Args:
            filename: Output filename (optional)
            
        Returns:
            Markdown string
        """
        lines = [
            "# RUL Prediction Analysis Report",
            f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
            ""
        ]
        
        # Fleet Summary
        if 'fleet_summary' in self.report_data:
            summary = self.report_data['fleet_summary']
            lines.extend([
                "## Fleet Summary",
                "",
                f"**Total Engines:** {summary['total_engines']}",
                "",
                "### Health Status Distribution",
                ""
            ])
            for status, count in summary['fleet_status'].items():
                lines.append(f"- {status}: {count} engines")
            
            lines.extend([
                "",
                "### RUL Statistics",
                "",
                f"- Mean: {summary['rul_statistics']['mean']:.1f} cycles",
                f"- Median: {summary['rul_statistics']['median']:.1f} cycles",
                f"- Min: {summary['rul_statistics']['min']:.1f} cycles",
                f"- Max: {summary['rul_statistics']['max']:.1f} cycles",
                ""
            ])
            
            if summary['critical_engines']:
                lines.extend([
                    "### ⚠️ Critical Engines",
                    "",
                    f"Engines requiring immediate attention: {summary['critical_engines']}",
                    ""
                ])
        
        # Model Performance
        if 'model_performance' in self.report_data:
            perf = self.report_data['model_performance']
            lines.extend([
                "## Model Performance",
                "",
                f"**Model:** {perf['model_name']}",
                "",
                "| Metric | Value |",
                "|--------|-------|",
                f"| RMSE | {perf['metrics']['rmse']:.2f} cycles |",
                f"| MAE | {perf['metrics']['mae']:.2f} cycles |",
                f"| R² | {perf['metrics']['r2']:.4f} |",
                "",
                "### Prediction Accuracy",
                "",
                f"- Within 10 cycles: {perf['prediction_quality']['within_10_cycles']:.1f}%",
                f"- Within 20 cycles: {perf['prediction_quality']['within_20_cycles']:.1f}%",
                f"- Within 30 cycles: {perf['prediction_quality']['within_30_cycles']:.1f}%",
                ""
            ])
        
        # Engine Details
        if 'engine_details' in self.report_data:
            lines.extend([
                "## Critical Engine Details",
                "",
                "| Engine | RUL | Status | Degradation Rate |",
                "|--------|-----|--------|------------------|"
            ])
            for engine_id, details in self.report_data['engine_details'].items():
                lines.append(
                    f"| {engine_id} | {details['current_rul']:.0f} | "
                    f"{details['status']} | {details['degradation_rate']:.2f}/cycle |"
                )
            lines.append("")
        
        markdown = '\n'.join(lines)
        
        if filename:
            filepath = os.path.join(self.output_dir, filename)
            with open(filepath, 'w') as f:
                f.write(markdown)
            logger.info(f"Report saved to {filepath}")
        
        return markdown
    
    def export_json(self, filename: str = None) -> Dict:
        """
        Export report to JSON format
        
        Args:
            filename: Output filename (optional)
            
        Returns:
            Report dictionary
        """
        report = {
            'generated_at': datetime.now().isoformat(),
            **self.report_data
        }
        
        if filename:
            import json
            filepath = os.path.join(self.output_dir, filename)
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"JSON report saved to {filepath}")
        
        return report
    
    def generate_full_report(self,
                            predictions_df: pd.DataFrame,
                            y_true: np.ndarray = None,
                            y_pred: np.ndarray = None,
                            model_name: str = 'Model',
                            export_formats: List[str] = None) -> Dict:
        """
        Generate comprehensive report with all sections
        
        Args:
            predictions_df: DataFrame with predictions
            y_true: True RUL values (optional)
            y_pred: Predicted RUL values (optional)
            model_name: Model name
            export_formats: List of formats to export ('markdown', 'json')
            
        Returns:
            Complete report dictionary
        """
        logger.info("Generating comprehensive report...")
        
        # Generate all sections
        self.generate_fleet_summary(predictions_df)
        self.generate_engine_details(predictions_df)
        
        if y_true is not None and y_pred is not None:
            self.generate_model_performance(y_true, y_pred, model_name)
        
        # Export if requested
        if export_formats:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            if 'markdown' in export_formats:
                self.export_markdown(f'report_{timestamp}.md')
            if 'json' in export_formats:
                self.export_json(f'report_{timestamp}.json')
        
        logger.info("Comprehensive report generated")
        return self.report_data


class ExperimentTracker:
    """
    Track and compare ML experiments
    Stores parameters, metrics, and artifacts for each experiment
    """
    
    def __init__(self, experiments_dir: str = None):
        """
        Initialize experiment tracker
        
        Args:
            experiments_dir: Directory to store experiments
        """
        self.experiments_dir = experiments_dir or os.path.join(config.RESULTS_DIR, 'experiments')
        os.makedirs(self.experiments_dir, exist_ok=True)
        
        self.experiments = {}
        self.current_experiment = None
        self._load_existing_experiments()
        logger.info(f"Initialized ExperimentTracker (dir: {self.experiments_dir})")
    
    def _load_existing_experiments(self):
        """Load existing experiments from directory"""
        for f in os.listdir(self.experiments_dir):
            if f.endswith('.json'):
                try:
                    filepath = os.path.join(self.experiments_dir, f)
                    with open(filepath, 'r') as file:
                        exp = json.load(file)
                        self.experiments[exp['experiment_id']] = exp
                except Exception:
                    continue
        
        logger.info(f"Loaded {len(self.experiments)} existing experiments")
    
    def create_experiment(self,
                          name: str,
                          description: str = '',
                          tags: List[str] = None) -> str:
        """
        Create a new experiment
        
        Args:
            name: Experiment name
            description: Experiment description
            tags: Optional tags for categorization
            
        Returns:
            Experiment ID
        """
        experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        experiment = {
            'experiment_id': experiment_id,
            'name': name,
            'description': description,
            'tags': tags or [],
            'created_at': datetime.now().isoformat(),
            'status': 'running',
            'parameters': {},
            'metrics': {},
            'artifacts': []
        }
        
        self.experiments[experiment_id] = experiment
        self.current_experiment = experiment_id
        
        logger.info(f"Created experiment: {name} ({experiment_id})")
        
        return experiment_id
    
    def log_parameters(self, params: Dict[str, Any]):
        """
        Log parameters for current experiment
        
        Args:
            params: Dictionary of parameter names and values
        """
        if not self.current_experiment:
            raise ValueError("No active experiment. Call create_experiment() first.")
        
        self.experiments[self.current_experiment]['parameters'].update(params)
        logger.debug(f"Logged {len(params)} parameters")
    
    def log_metrics(self, metrics: Dict[str, float]):
        """
        Log metrics for current experiment
        
        Args:
            metrics: Dictionary of metric names and values
        """
        if not self.current_experiment:
            raise ValueError("No active experiment. Call create_experiment() first.")
        
        self.experiments[self.current_experiment]['metrics'].update(metrics)
        logger.debug(f"Logged {len(metrics)} metrics")
    
    def log_artifact(self, name: str, filepath: str):
        """
        Log an artifact for current experiment
        
        Args:
            name: Artifact name
            filepath: Path to artifact file
        """
        if not self.current_experiment:
            raise ValueError("No active experiment. Call create_experiment() first.")
        
        artifact = {
            'name': name,
            'path': filepath,
            'logged_at': datetime.now().isoformat()
        }
        
        self.experiments[self.current_experiment]['artifacts'].append(artifact)
        logger.debug(f"Logged artifact: {name}")
    
    def end_experiment(self, status: str = 'completed'):
        """
        End current experiment
        
        Args:
            status: Final status ('completed', 'failed', 'aborted')
        """
        if not self.current_experiment:
            return
        
        exp = self.experiments[self.current_experiment]
        exp['status'] = status
        exp['ended_at'] = datetime.now().isoformat()
        
        # Save to file
        filepath = os.path.join(self.experiments_dir, f"{self.current_experiment}.json")
        with open(filepath, 'w') as f:
            json.dump(exp, f, indent=2)
        
        logger.info(f"Experiment {self.current_experiment} ended with status: {status}")
        self.current_experiment = None
    
    def compare_experiments(self,
                            experiment_ids: List[str] = None,
                            metric: str = 'rmse') -> pd.DataFrame:
        """
        Compare multiple experiments
        
        Args:
            experiment_ids: List of experiment IDs to compare (None = all)
            metric: Primary metric for sorting
            
        Returns:
            Comparison DataFrame
        """
        if experiment_ids is None:
            experiment_ids = list(self.experiments.keys())
        
        comparisons = []
        
        for exp_id in experiment_ids:
            if exp_id not in self.experiments:
                continue
            
            exp = self.experiments[exp_id]
            row = {
                'experiment_id': exp_id,
                'name': exp['name'],
                'status': exp['status'],
                'created_at': exp['created_at']
            }
            row.update(exp.get('metrics', {}))
            comparisons.append(row)
        
        df = pd.DataFrame(comparisons)
        
        if metric in df.columns:
            df = df.sort_values(metric)
        
        return df
    
    def get_best_experiment(self,
                            metric: str = 'rmse',
                            minimize: bool = True) -> Dict:
        """
        Get best experiment by metric
        
        Args:
            metric: Metric to optimize
            minimize: True if lower is better
            
        Returns:
            Best experiment details
        """
        best_exp = None
        best_value = float('inf') if minimize else float('-inf')
        
        for exp_id, exp in self.experiments.items():
            if exp['status'] != 'completed':
                continue
            
            value = exp.get('metrics', {}).get(metric)
            if value is None:
                continue
            
            if minimize and value < best_value:
                best_value = value
                best_exp = exp
            elif not minimize and value > best_value:
                best_value = value
                best_exp = exp
        
        return best_exp
    
    def get_experiment_summary(self, experiment_id: str = None) -> str:
        """Generate summary for an experiment"""
        if experiment_id is None:
            experiment_id = self.current_experiment
        
        if experiment_id not in self.experiments:
            return "Experiment not found"
        
        exp = self.experiments[experiment_id]
        
        lines = [
            "=" * 60,
            f"EXPERIMENT: {exp['name']}",
            "=" * 60,
            f"ID: {exp['experiment_id']}",
            f"Status: {exp['status']}",
            f"Created: {exp['created_at']}",
            "",
            "Parameters:",
        ]
        
        for k, v in exp.get('parameters', {}).items():
            lines.append(f"  {k}: {v}")
        
        lines.extend(["", "Metrics:"])
        for k, v in exp.get('metrics', {}).items():
            lines.append(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        
        lines.append("=" * 60)
        
        return '\n'.join(lines)


class ModelSerializer:
    """
    Versioned model serialization with registry and rollback support
    Manages model versions with metadata and changelog
    """
    
    def __init__(self, registry_dir: str = None):
        """
        Initialize model serializer
        
        Args:
            registry_dir: Directory for model registry
        """
        self.registry_dir = registry_dir or os.path.join(config.MODELS_DIR, 'registry')
        os.makedirs(self.registry_dir, exist_ok=True)
        
        self.registry = {}
        self.changelog = []
        self._load_registry()
        logger.info(f"Initialized ModelSerializer (registry: {self.registry_dir})")
    
    def _load_registry(self):
        """Load model registry from disk"""
        registry_file = os.path.join(self.registry_dir, 'registry.json')
        if os.path.exists(registry_file):
            with open(registry_file, 'r') as f:
                self.registry = json.load(f)
        
        changelog_file = os.path.join(self.registry_dir, 'changelog.json')
        if os.path.exists(changelog_file):
            with open(changelog_file, 'r') as f:
                self.changelog = json.load(f)
    
    def _save_registry(self):
        """Save registry to disk"""
        registry_file = os.path.join(self.registry_dir, 'registry.json')
        with open(registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
        
        changelog_file = os.path.join(self.registry_dir, 'changelog.json')
        with open(changelog_file, 'w') as f:
            json.dump(self.changelog, f, indent=2)
    
    def save_model(self,
                   model,
                   model_name: str,
                   version: str = None,
                   metrics: Dict[str, float] = None,
                   description: str = '') -> Dict:
        """
        Save model with version and metadata
        
        Args:
            model: Model object to save
            model_name: Name of the model
            version: Version string (auto-generated if None)
            metrics: Performance metrics
            description: Version description
            
        Returns:
            Model version metadata
        """
        # Generate version
        if version is None:
            existing = self.registry.get(model_name, {}).get('versions', [])
            version = f"v{len(existing) + 1}.0.0"
        
        # Create version directory
        version_dir = os.path.join(self.registry_dir, model_name, version)
        os.makedirs(version_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(version_dir, 'model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Create metadata
        metadata = {
            'model_name': model_name,
            'version': version,
            'created_at': datetime.now().isoformat(),
            'model_path': model_path,
            'metrics': metrics or {},
            'description': description,
            'model_type': type(model).__name__
        }
        
        # Save metadata
        metadata_path = os.path.join(version_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Update registry
        if model_name not in self.registry:
            self.registry[model_name] = {
                'versions': [],
                'latest': None,
                'production': None
            }
        
        self.registry[model_name]['versions'].append(version)
        self.registry[model_name]['latest'] = version
        
        # Add to changelog
        self.changelog.append({
            'timestamp': datetime.now().isoformat(),
            'action': 'save',
            'model_name': model_name,
            'version': version,
            'description': description
        })
        
        self._save_registry()
        
        logger.info(f"Saved model {model_name} version {version}")
        
        return metadata
    
    def load_model(self,
                   model_name: str,
                   version: str = None) -> Any:
        """
        Load model from registry
        
        Args:
            model_name: Name of the model
            version: Version to load (None = latest)
            
        Returns:
            Loaded model object
        """
        if model_name not in self.registry:
            raise ValueError(f"Model {model_name} not found in registry")
        
        if version is None:
            version = self.registry[model_name]['latest']
        
        if version not in self.registry[model_name]['versions']:
            raise ValueError(f"Version {version} not found for {model_name}")
        
        model_path = os.path.join(self.registry_dir, model_name, version, 'model.pkl')
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        logger.info(f"Loaded model {model_name} version {version}")
        
        return model
    
    def promote_to_production(self, model_name: str, version: str):
        """
        Promote a model version to production
        
        Args:
            model_name: Model name
            version: Version to promote
        """
        if model_name not in self.registry:
            raise ValueError(f"Model {model_name} not found")
        
        if version not in self.registry[model_name]['versions']:
            raise ValueError(f"Version {version} not found")
        
        old_prod = self.registry[model_name]['production']
        self.registry[model_name]['production'] = version
        
        self.changelog.append({
            'timestamp': datetime.now().isoformat(),
            'action': 'promote',
            'model_name': model_name,
            'version': version,
            'previous_production': old_prod
        })
        
        self._save_registry()
        
        logger.info(f"Promoted {model_name} {version} to production")
    
    def rollback(self, model_name: str, to_version: str = None):
        """
        Rollback to a previous version
        
        Args:
            model_name: Model name
            to_version: Version to rollback to (None = previous)
        """
        if model_name not in self.registry:
            raise ValueError(f"Model {model_name} not found")
        
        versions = self.registry[model_name]['versions']
        current = self.registry[model_name]['production'] or self.registry[model_name]['latest']
        
        if to_version is None:
            current_idx = versions.index(current)
            if current_idx == 0:
                raise ValueError("Already at earliest version")
            to_version = versions[current_idx - 1]
        
        self.registry[model_name]['production'] = to_version
        
        self.changelog.append({
            'timestamp': datetime.now().isoformat(),
            'action': 'rollback',
            'model_name': model_name,
            'from_version': current,
            'to_version': to_version
        })
        
        self._save_registry()
        
        logger.info(f"Rolled back {model_name} from {current} to {to_version}")
    
    def list_versions(self, model_name: str) -> List[Dict]:
        """List all versions of a model"""
        if model_name not in self.registry:
            return []
        
        versions = []
        for ver in self.registry[model_name]['versions']:
            metadata_path = os.path.join(self.registry_dir, model_name, ver, 'metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    versions.append(json.load(f))
        
        return versions
    
    def get_registry_summary(self) -> str:
        """Generate registry summary"""
        lines = [
            "=" * 60,
            "MODEL REGISTRY SUMMARY",
            "=" * 60,
            ""
        ]
        
        for model_name, info in self.registry.items():
            lines.extend([
                f"Model: {model_name}",
                f"  Versions: {len(info['versions'])}",
                f"  Latest: {info['latest']}",
                f"  Production: {info['production'] or 'None'}",
                ""
            ])
        
        lines.append("=" * 60)
        
        return '\n'.join(lines)


        return '\n'.join(lines)


class ModelSignature:
    """
    Model integrity verification
    Generates and verifies SHA256 checksums for model artifacts
    """
    
    def __init__(self, key: str = None):
        """
        Initialize model signature
        
        Args:
            key: Optional secret key for signing (HMAC)
        """
        self.key = key.encode() if key else None
        logger.info("Initialized ModelSignature")
    
    def generate_checksum(self, filepath: str) -> str:
        """
        Generate SHA256 checksum for a file
        
        Args:
            filepath: Path to file
            
        Returns:
            Hex digest of checksum
        """
        import hashlib
        
        sha256_hash = hashlib.sha256()
        
        with open(filepath, "rb") as f:
            # Read in chunks to handle large files
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        
        return sha256_hash.hexdigest()
    
    def sign_model(self, model_path: str, metadata: Dict = None) -> Dict:
        """
        Sign a model artifact
        
        Args:
            model_path: Path to model file
            metadata: Additional metadata to include
            
        Returns:
            Signature dictionary
        """
        import hashlib
        import hmac
        import json
        
        checksum = self.generate_checksum(model_path)
        timestamp = datetime.now().isoformat()
        
        signature_data = {
            'checksum': checksum,
            'timestamp': timestamp,
            'metadata': metadata or {}
        }
        
        # Add HMAC signature if key is provided
        if self.key:
            message = json.dumps(signature_data, sort_keys=True).encode()
            signature = hmac.new(self.key, message, hashlib.sha256).hexdigest()
            signature_data['signature'] = signature
        
        logger.info(f"Signed model at {model_path} (checksum: {checksum[:8]}...)")
        
        return signature_data
    
    def verify_integrity(self, 
                        model_path: str, 
                        signature_data: Dict) -> bool:
        """
        Verify model integrity against signature
        
        Args:
            model_path: Path to model file
            signature_data: Expected signature data
            
        Returns:
            True if valid
        """
        import hashlib
        import hmac
        import json
        
        # 1. Verify Checksum
        current_checksum = self.generate_checksum(model_path)
        if current_checksum != signature_data['checksum']:
            logger.warning("Checksum mismatch! Model may be corrupted or tampered.")
            return False
        
        # 2. Verify HMAC Signature (if key exists)
        if self.key and 'signature' in signature_data:
            expected_sig = signature_data['signature']
            
            # Reconstruct message
            verify_data = signature_data.copy()
            del verify_data['signature']
            message = json.dumps(verify_data, sort_keys=True).encode()
            
            computed_sig = hmac.new(self.key, message, hashlib.sha256).hexdigest()
            
            if not hmac.compare_digest(expected_sig, computed_sig):
                logger.warning("Signature mismatch! Metadata may be tampered.")
                return False
        
        logger.info(f"Model integrity verified for {model_path}")
        return True


if __name__ == "__main__":
    logger.info("Utility functions loaded successfully")
    print("Available utility functions:")
    print("- setup_logging()")
    print("- generate_sequences()")
    print("- generate_sequences_for_prediction()")
    print("- save_model() / load_model()")
    print("- save_scaler() / load_scaler()")
    print("- save_results() / load_results()")
    print("- add_remaining_useful_life()")
    print("- calculate_rate_of_change()")
    print("- add_rolling_statistics()")
    print("- asymmetric_score()")

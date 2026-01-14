"""
Anomaly Detection for Early Fault Warning
Detects abnormal sensor patterns that indicate impending failures
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple
import pickle
import config
from utils import setup_logging

logger = setup_logging(__name__)


class AnomalyDetector:
    """
    Anomaly detection for early fault warning in turbofan engines
    """
    
    def __init__(self, method: str = 'isolation_forest'):
        """
        Initialize anomaly detector
        
        Args:
            method: Detection method ('isolation_forest' or 'statistical')
        """
        self.method = method
        self.model = None
        self.scaler = StandardScaler()
        self.threshold = None
        
        if method == 'isolation_forest':
            self.model = IsolationForest(
                contamination=config.ANOMALY_CONFIG['contamination'],
                random_state=config.ANOMALY_CONFIG['random_state'],
                n_estimators=100
            )
            logger.info("Initialized Isolation Forest anomaly detector")
        elif method == 'statistical':
            logger.info("Initialized Statistical anomaly detector")
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def fit(self, X: np.ndarray) -> None:
        """
        Fit the anomaly detector on normal (healthy) data
        
        Args:
            X: Training features (should be from healthy engines)
        """
        logger.info(f"Fitting anomaly detector on {len(X)} samples...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        if self.method == 'isolation_forest':
            self.model.fit(X_scaled)
        elif self.method == 'statistical':
            # Compute mean and std for each feature
            self.mean = np.mean(X_scaled, axis=0)
            self.std = np.std(X_scaled, axis=0)
            # Threshold: 3 standard deviations
            self.threshold = 3.0
        
        logger.info("Anomaly detector fitted")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies
        
        Args:
            X: Features to check for anomalies
            
        Returns:
            Array of predictions: 1 for normal, -1 for anomaly
        """
        if self.model is None and self.method == 'isolation_forest':
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        if self.method == 'isolation_forest':
            predictions = self.model.predict(X_scaled)
        elif self.method == 'statistical':
            # Calculate z-scores
            z_scores = np.abs((X_scaled - self.mean) / (self.std + 1e-6))
            # Flag as anomaly if any feature exceeds threshold
            max_z_scores = np.max(z_scores, axis=1)
            predictions = np.where(max_z_scores > self.threshold, -1, 1)
        
        return predictions
    
    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Get anomaly scores (lower = more anomalous)
        
        Args:
            X: Features
            
        Returns:
            Array of anomaly scores
        """
        if self.method != 'isolation_forest':
            logger.warning("Anomaly scores only available for Isolation Forest")
            return None
        
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        scores = self.model.score_samples(X_scaled)
        
        return scores
    
    def detect_anomalies_per_engine(self, df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
        """
        Detect anomalies for each engine in the dataset
        
        Args:
            df: DataFrame with unit_id and features
            feature_cols: List of feature column names
            
        Returns:
            DataFrame with added 'anomaly' column (-1 for anomaly, 1 for normal)
        """
        logger.info("Detecting anomalies per engine...")
        
        df_copy = df.copy()
        
        # Extract features
        X = df[feature_cols].values
        
        # Predict anomalies
        predictions = self.predict(X)
        df_copy['anomaly'] = predictions
        
        # Count anomalies per engine
        anomaly_counts = df_copy.groupby('unit_id')['anomaly'].apply(
            lambda x: (x == -1).sum()
        ).reset_index()
        anomaly_counts.columns = ['unit_id', 'anomaly_count']
        
        logger.info(f"Detected anomalies in {len(anomaly_counts[anomaly_counts['anomaly_count'] > 0])} engines")
        
        return df_copy
    
    def evaluate(self, df: pd.DataFrame, rul_col: str = 'RUL', critical_rul: int = 30) -> Dict:
        """
        Evaluate anomaly detection effectiveness
        
        Args:
            df: DataFrame with 'anomaly' and RUL columns
            rul_col: Name of RUL column
            critical_rul: RUL threshold for considering engine as failing
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating anomaly detector...")
        
        # Define ground truth: engines with low RUL are failing
        df['is_critical'] = (df[rul_col] < critical_rul).astype(int)
        df['is_anomaly'] = (df['anomaly'] == -1).astype(int)
        
        # Calculate metrics
        true_positives = ((df['is_critical'] == 1) & (df['is_anomaly'] == 1)).sum()
        false_positives = ((df['is_critical'] == 0) & (df['is_anomaly'] == 1)).sum()
        true_negatives = ((df['is_critical'] == 0) & (df['is_anomaly'] == 0)).sum()
        false_negatives = ((df['is_critical'] == 1) & (df['is_anomaly'] == 0)).sum()
        
        precision = true_positives / (true_positives + false_positives + 1e-6)
        recall = true_positives / (true_positives + false_negatives + 1e-6)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': int(true_positives),
            'false_positives': int(false_positives),
            'true_negatives': int(true_negatives),
            'false_negatives': int(false_negatives)
        }
        
        logger.info(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1_score:.3f}")
        
        return metrics
    
    def save(self, filepath: str) -> None:
        """
        Save the anomaly detector
        
        Args:
            filepath: Path to save the detector
        """
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'method': self.method,
                'threshold': self.threshold,
                'mean': getattr(self, 'mean', None),
                'std': getattr(self, 'std', None)
            }, f)
        
        logger.info(f"Anomaly detector saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        Load an anomaly detector
        
        Args:
            filepath: Path to the saved detector
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.model = data['model']
        self.scaler = data['scaler']
        self.method = data['method']
        self.threshold = data['threshold']
        if data.get('mean') is not None:
            self.mean = data['mean']
            self.std = data['std']
        
        logger.info(f"Anomaly detector loaded from {filepath}")


if __name__ == "__main__":
    # Test anomaly detector
    print("="*60)
    print("Testing Anomaly Detector")
    print("="*60)
    
    # Generate synthetic data
    np.random.seed(42)
    
    # Normal data (healthy engines)
    X_normal = np.random.randn(800, 20)
    
    # Anomalous data (failing engines with shifted distribution)
    X_anomaly = np.random.randn(200, 20) + 2.0  # Shifted mean
    
    # Combined test data
    X_test = np.vstack([X_normal[:100], X_anomaly[:50]])
    y_true = np.array([1]*100 + [-1]*50)  # Ground truth labels
    
    # Create detector and fit
    detector = AnomalyDetector('isolation_forest')
    detector.fit(X_normal)
    
    # Predict
    predictions = detector.predict(X_test)
    
    # Calculate accuracy
    accuracy = (predictions == y_true).mean()
    print(f"\nAccuracy: {accuracy:.3f}")
    
    # Count predictions
    num_anomalies = (predictions == -1).sum()
    num_normal = (predictions == 1).sum()
    print(f"Predicted anomalies: {num_anomalies}")
    print(f"Predicted normal: {num_normal}")
    
    # Get anomaly scores
    scores = detector.predict_scores(X_test)
    if scores is not None:
        print(f"\nAnomaly score range: [{scores.min():.3f}, {scores.max():.3f}]")
    
    print("\n" + "="*60)
    print("Anomaly detector test complete!")
    print("="*60)

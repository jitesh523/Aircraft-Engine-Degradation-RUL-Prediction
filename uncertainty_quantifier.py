"""
Uncertainty Quantification for RUL Predictions
Implements Monte Carlo Dropout to estimate prediction confidence intervals
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict
import config
from utils import setup_logging

logger = setup_logging(__name__)


class UncertaintyQuantifier:
    """
    Uncertainty quantification using Monte Carlo Dropout
    Provides confidence intervals for predictions
    """
    
    def __init__(self, n_iterations: int = 100, confidence_level: float = 0.95):
        """
        Initialize uncertainty quantifier
        
        Args:
            n_iterations: Number of Monte Carlo forward passes
            confidence_level: Confidence level for intervals (default: 95%)
        """
        self.n_iterations = n_iterations
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        logger.info(f"Initialized Uncertainty Quantifier with {n_iterations} iterations, "
                   f"{confidence_level*100}% confidence level")
    
    def predict_with_uncertainty(self, 
                                model, 
                                X: np.ndarray,
                                is_keras: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty estimates using Monte Carlo Dropout
        
        Args:
            model: Trained model (must have dropout layers)
            X: Input data
            is_keras: Whether model is Keras/TensorFlow (enables training mode during prediction)
            
        Returns:
            Tuple of (mean_predictions, lower_bound, upper_bound)
        """
        logger.info(f"Generating predictions with uncertainty using {self.n_iterations} forward passes...")
        
        predictions = []
        
        if is_keras:
            # Enable dropout during prediction by setting training=True
            import tensorflow as tf
            
            for i in range(self.n_iterations):
                # Forward pass with dropout enabled
                pred = model.model(X, training=True)  # Keep dropout active
                predictions.append(pred.numpy().flatten())
                
                if (i + 1) % 20 == 0:
                    logger.info(f"Completed {i+1}/{self.n_iterations} iterations")
        else:
            logger.warning("Monte Carlo Dropout currently only supported for Keras models")
            # For non-Keras models, just repeat predictions (no uncertainty)
            for _ in range(self.n_iterations):
                pred = model.predict(X)
                predictions.append(pred)
        
        predictions = np.array(predictions)  # Shape: (n_iterations, n_samples)
        
        # Calculate statistics
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        # Calculate confidence intervals
        lower_percentile = (self.alpha / 2) * 100
        upper_percentile = (1 - self.alpha / 2) * 100
        
        lower_bound = np.percentile(predictions, lower_percentile, axis=0)
        upper_bound = np.percentile(predictions, upper_percentile, axis=0)
        
        # Clip negative values
        mean_pred = np.maximum(mean_pred, 0)
        lower_bound = np.maximum(lower_bound, 0)
        upper_bound = np.maximum(upper_bound, 0)
        
        logger.info(f"Uncertainty quantification complete.")
        logger.info(f"Mean uncertainty (std): {np.mean(std_pred):.2f} cycles")
        logger.info(f"Mean confidence interval width: {np.mean(upper_bound - lower_bound):.2f} cycles")
        
        return mean_pred, lower_bound, upper_bound
    
    def get_prediction_intervals(self, 
                                mean_pred: np.ndarray,
                                lower_bound: np.ndarray,
                                upper_bound: np.ndarray) -> pd.DataFrame:
        """
        Create DataFrame with prediction intervals
        
        Args:
            mean_pred: Mean predictions
            lower_bound: Lower confidence bounds
            upper_bound: Upper confidence bounds
            
        Returns:
            DataFrame with predictions and intervals
        """
        df = pd.DataFrame({
            'prediction': mean_pred,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'interval_width': upper_bound - lower_bound,
            'relative_uncertainty': (upper_bound - lower_bound) / (mean_pred + 1e-6)
        })
        
        return df
    
    def classify_uncertainty(self, interval_width: np.ndarray) -> np.ndarray:
        """
        Classify predictions by uncertainty level
        
        Args:
            interval_width: Width of confidence intervals
            
        Returns:
            Array of uncertainty classifications
        """
        # Classify based on interval width
        classifications = np.empty(len(interval_width), dtype=object)
        
        # Low uncertainty: < 15 cycles
        # Medium uncertainty: 15-30 cycles
        # High uncertainty: > 30 cycles
        
        classifications[interval_width < 15] = 'Low'
        classifications[(interval_width >= 15) & (interval_width < 30)] = 'Medium'
        classifications[interval_width >= 30] = 'High'
        
        return classifications
    
    def evaluate_uncertainty_calibration(self,
                                        y_true: np.ndarray,
                                        lower_bound: np.ndarray,
                                        upper_bound: np.ndarray) -> Dict:
        """
        Evaluate how well calibrated the uncertainty estimates are
        
        Args:
            y_true: True RUL values
            lower_bound: Lower confidence bounds
            upper_bound: Upper confidence bounds
            
        Returns:
            Dictionary with calibration metrics
        """
        logger.info("Evaluating uncertainty calibration...")
        
        # Check if true values fall within confidence intervals
        within_interval = (y_true >= lower_bound) & (y_true <= upper_bound)
        coverage = np.mean(within_interval) * 100
        
        # Expected coverage for 95% confidence interval
        expected_coverage = self.confidence_level * 100
        
        # Average interval width
        avg_width = np.mean(upper_bound - lower_bound)
        
        # Sharpness (narrower is better, but must maintain coverage)
        sharpness = avg_width
        
        calibration_metrics = {
            'coverage': float(coverage),
            'expected_coverage': float(expected_coverage),
            'calibration_error': float(abs(coverage - expected_coverage)),
            'average_interval_width': float(avg_width),
            'sharpness': float(sharpness),
            'well_calibrated': abs(coverage - expected_coverage) < 5.0  # Within 5%
        }
        
        logger.info(f"Coverage: {coverage:.1f}% (expected: {expected_coverage:.0f}%)")
        logger.info(f"Calibration error: {calibration_metrics['calibration_error']:.1f}%")
        logger.info(f"Average interval width: {avg_width:.2f} cycles")
        logger.info(f"Well calibrated: {calibration_metrics['well_calibrated']}")
        
        return calibration_metrics
    
    def get_high_uncertainty_engines(self,
                                    interval_df: pd.DataFrame,
                                    unit_ids: np.ndarray,
                                    top_k: int = 10) -> pd.DataFrame:
        """
        Identify engines with highest prediction uncertainty
        
        Args:
            interval_df: DataFrame with prediction intervals
            unit_ids: Engine unit IDs
            top_k: Number of top uncertain engines to return
            
        Returns:
            DataFrame with most uncertain engines
        """
        interval_df['unit_id'] = unit_ids
        interval_df['uncertainty_class'] = self.classify_uncertainty(
            interval_df['interval_width'].values
        )
        
        # Sort by interval width (descending)
        high_uncertainty = interval_df.nlargest(top_k, 'interval_width')
        
        logger.info(f"\nTop {top_k} engines with highest prediction uncertainty:")
        for idx, row in high_uncertainty.iterrows():
            logger.info(f"  Unit {row['unit_id']}: RUL = {row['prediction']:.0f} "
                       f"± {row['interval_width']/2:.0f} cycles [{row['lower_bound']:.0f}, {row['upper_bound']:.0f}]")
        
        return high_uncertainty


if __name__ == "__main__":
    # Test uncertainty quantifier
    print("="*60)
    print("Testing Uncertainty Quantifier")
    print("="*60)
    
    # Simulate predictions from Monte Carlo  dropout
    np.random.seed(42)
    n_samples = 50
    n_iterations = 100
    
    # Simulate predictions with varying uncertainty
    base_predictions = np.random.rand(n_samples) * 100
    
    # Add random noise to simulate Monte Carlo dropout
    predictions_mc = []
    for _ in range(n_iterations):
        noise = np.random.randn(n_samples) * 10  # Std of 10 cycles
        predictions_mc.append(base_predictions + noise)
    
    predictions_mc = np.array(predictions_mc)
    
    # Calculate statistics
    mean_pred = np.mean(predictions_mc, axis=0)
    lower_bound = np.percentile(predictions_mc, 2.5, axis=0)
    upper_bound = np.percentile(predictions_mc, 97.5, axis=0)
    
    # True values (for calibration check)
    y_true = mean_pred + np.random.randn(n_samples) * 5
    
    # Test quantifier
    quantifier = UncertaintyQuantifier(n_iterations=100)
    
    # Get prediction intervals
    print("\n1. Prediction Intervals:")
    interval_df = quantifier.get_prediction_intervals(mean_pred, lower_bound, upper_bound)
    print(interval_df.head())
    
    # Evaluate calibration
    print("\n2. Calibration Metrics:")
    calibration = quantifier.evaluate_uncertainty_calibration(y_true, lower_bound, upper_bound)
    for key, value in calibration.items():
        print(f"  {key}: {value}")
    
    # High uncertainty engines
    print("\n3. High Uncertainty Engines:")
    unit_ids = np.arange(1, n_samples + 1)
    high_unc = quantifier.get_high_uncertainty_engines(interval_df, unit_ids, top_k=5)
    
    print("\n" + "="*60)
    print("Uncertainty quantifier test complete!")
    print("="*60)

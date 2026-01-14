"""
Model Explainability using SHAP (SHapley Additive exPlanations)
Analyzes feature importance and explains individual predictions
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from typing import List, Dict
import os
import config
from utils import setup_logging

logger = setup_logging(__name__)


class SHAPExplainer:
    """
    SHAP-based model explainability for LSTM and baseline models
    """
    
    def __init__(self, model, model_type: str = 'lstm'):
        """
        Initialize SHAP explainer
        
        Args:
            model: Trained model to explain
            model_type: Type of model ('lstm', 'random_forest', 'linear_regression')
        """
        self.model = model
        self.model_type = model_type
        self.explainer = None
        self.shap_values = None
        self.feature_names = None
        
        logger.info(f"Initialized SHAP Explainer for {model_type} model")
    
    def create_explainer(self, X_background: np.ndarray, feature_names: List[str] = None):
        """
        Create SHAP explainer
        
        Args:
            X_background: Background data for SHAP (subset of training data)
            feature_names: List of feature names
        """
        self.feature_names = feature_names
        
        logger.info(f"Creating SHAP explainer with {len(X_background)} background samples...")
        
        if self.model_type == 'lstm':
            # For LSTM, use DeepExplainer
            self.explainer = shap.DeepExplainer(
                self.model.model,  # Keras model
                X_background
            )
            logger.info("Created DeepExplainer for LSTM")
            
        elif self.model_type in ['random_forest', 'linear_regression']:
            # For tree-based and linear models, use Explainer
            self.explainer = shap.Explainer(
                self.model.model,
                X_background,
                feature_names=feature_names
            )
            logger.info(f"Created TreeExplainer for {self.model_type}")
        
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def explain_predictions(self, X_test: np.ndarray, max_samples: int = 100):
        """
        Generate SHAP values for test data
        
        Args:
            X_test: Test data to explain
            max_samples: Maximum number of samples to explain (for speed)
            
        Returns:
            SHAP values
        """
        if self.explainer is None:
            raise ValueError("Explainer not created. Call create_explainer() first.")
        
        # Limit samples for computational efficiency
        n_samples = min(len(X_test), max_samples)
        X_explain = X_test[:n_samples]
        
        logger.info(f"Generating SHAP values for {n_samples} samples...")
        
        if self.model_type == 'lstm':
            # For LSTM with sequences, SHAP values have shape (samples, timesteps, features)
            self.shap_values = self.explainer.shap_values(X_explain)
            
            # Average over time dimension for summary
            # Shape: (samples, features)
            if isinstance(self.shap_values, list):
                # Multi-output, take first output
                self.shap_values_summary = np.mean(np.abs(self.shap_values[0]), axis=1)
            else:
                self.shap_values_summary = np.mean(np.abs(self.shap_values), axis=1)
        else:
            # For baseline models
            self.shap_values = self.explainer.shap_values(X_explain)
            self.shap_values_summary = self.shap_values
        
        logger.info("SHAP values generated")
        return self.shap_values
    
    def plot_summary(self, save_path: str = None):
        """
        Create SHAP summary plot showing feature importance
        
        Args:
            save_path: Path to save plot
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Call explain_predictions() first.")
        
        logger.info("Creating SHAP summary plot...")
        
        plt.figure(figsize=(10, 8))
        
        if self.model_type == 'lstm':
            # Use time-averaged SHAP values for summary
            shap.summary_plot(
                self.shap_values_summary,
                feature_names=self.feature_names,
                show=False
            )
        else:
            shap.summary_plot(
                self.shap_values,
                feature_names=self.feature_names,
                show=False
            )
        
        plt.title(f'SHAP Feature Importance - {self.model_type.upper()} Model', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=config.PLOT_CONFIG['dpi'], bbox_inches='tight')
            logger.info(f"Saved SHAP summary plot to {save_path}")
            plt.close()
        else:
            plt.show()
    
    def plot_waterfall(self, sample_idx: int, save_path: str = None):
        """
        Create waterfall plot for individual prediction explanation
        
        Args:
            sample_idx: Index of sample to explain
            save_path: Path to save plot
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Call explain_predictions() first.")
        
        logger.info(f"Creating waterfall plot for sample {sample_idx}...")
        
        plt.figure(figsize=(10, 6))
        
        if self.model_type == 'lstm':
            # For LSTM, use time-averaged SHAP values
            shap_explanation = shap.Explanation(
                values=self.shap_values_summary[sample_idx],
                base_values=np.mean(self.shap_values_summary),
                data=None,
                feature_names=self.feature_names
            )
        else:
            shap_explanation = shap.Explanation(
                values=self.shap_values[sample_idx],
                base_values=self.explainer.expected_value,
                data=None,
                feature_names=self.feature_names
            )
        
        shap.waterfall_plot(shap_explanation, show=False)
        
        plt.title(f'SHAP Waterfall Plot - Sample {sample_idx}', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=config.PLOT_CONFIG['dpi'], bbox_inches='tight')
            logger.info(f"Saved waterfall plot to {save_path}")
            plt.close()
        else:
            plt.show()
    
    def get_feature_importance(self, top_k: int = 20) -> pd.DataFrame:
        """
        Get feature importance ranking based on mean absolute SHAP values
        
        Args:
            top_k: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Call explain_predictions() first.")
        
        logger.info("Computing feature importance from SHAP values...")
        
        # Calculate mean absolute SHAP value for each feature
        mean_abs_shap = np.mean(np.abs(self.shap_values_summary), axis=0)
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names if self.feature_names else [f'feature_{i}' for i in range(len(mean_abs_shap))],
            'importance': mean_abs_shap
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        logger.info(f"\nTop {min(top_k, len(importance_df))} Most Important Features:")
        for i, row in importance_df.head(top_k).iterrows():
            logger.info(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")
        
        return importance_df.head(top_k)
    
    def plot_feature_importance_bar(self, top_k: int = 20, save_path: str = None):
        """
        Create bar plot of top feature importances
        
        Args:
            top_k: Number of top features to plot
            save_path: Path to save plot
        """
        importance_df = self.get_feature_importance(top_k)
        
        plt.figure(figsize=(10, 8))
        
        # Create horizontal bar plot
        plt.barh(importance_df['feature'], importance_df['importance'])
        plt.xlabel('Mean |SHAP Value|', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.title(f'Top {top_k} Important Features - {self.model_type.upper()}', 
                 fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()  # Highest importance at top
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=config.PLOT_CONFIG['dpi'], bbox_inches='tight')
            logger.info(f"Saved feature importance bar plot to {save_path}")
            plt.close()
        else:
            plt.show()
    
    def explain_sensor_contributions(self, sample_idx: int) -> Dict:
        """
        Explain sensor contributions for a specific prediction
        
        Args:
            sample_idx: Index of sample to explain
            
        Returns:
            Dictionary with sensor contributions
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Call explain_predictions() first.")
        
        shap_vals = self.shap_values_summary[sample_idx]
        
        # Get indices of sensors (features starting with 'sensor_')
        sensor_features = [i for i, name in enumerate(self.feature_names) 
                          if name and 'sensor_' in name]
        
        # Calculate total SHAP value from sensors
        sensor_contributions = {}
        for idx in sensor_features:
            sensor_name = self.feature_names[idx]
            sensor_contributions[sensor_name] = float(shap_vals[idx])  
        
        # Sort by absolute contribution
        sensor_contributions = dict(sorted(
            sensor_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        ))
        
        logger.info(f"\nSensor Contributions for Sample {sample_idx}:")
        for sensor, contribution in list(sensor_contributions.items())[:10]:
            logger.info(f"  {sensor}: {contribution:+.4f}")
        
        return sensor_contributions


if __name__ == "__main__":
    # Test SHAP explainer
    print("="*60)
    print("Testing SHAP Explainer")
    print("="*60)
    
    print("\nNote: SHAP explainer requires trained models to test.")
    print("Run after training to generate explanations.")
    
    print("\n" + "="*60)
    print("SHAP explainer module loaded successfully!")
    print("="*60)

"""
Visualization Module for RUL Prediction Results
Creates plots for model evaluation and analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
import os
import config
from utils import setup_logging

logger = setup_logging(__name__)

# Set plotting style
plt.style.use(config.PLOT_CONFIG['style'])
sns.set_palette("husl")


class RULVisualizer:
    """
    Visualizer for RUL prediction results
    """
    
    def __init__(self, save_dir: str = None):
        """
        Initialize visualizer
        
        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = save_dir or config.PLOTS_DIR
        os.makedirs(self.save_dir, exist_ok=True)
        logger.info(f"Initialized visualizer. Plots will be saved to: {self.save_dir}")
    
    def plot_prediction_scatter(self, 
                                 y_true: np.ndarray,
                                 y_pred: np.ndarray, 
                                 model_name: str = 'Model',
                                 save_filename: str = None):
        """
        Scatter plot of predicted vs actual RUL
        
        Args:
            y_true: True RUL values
            y_pred: Predicted RUL values
            model_name: Name of the model
            save_filename: Filename to save plot
        """
        fig, ax = plt.subplots(figsize=config.PLOT_CONFIG['figure_size'])
        
        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.5, s=20)
        
        # Perfect prediction line
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        ax.set_xlabel('True RUL (cycles)', fontsize=12)
        ax.set_ylabel('Predicted RUL (cycles)', fontsize=12)
        ax.set_title(f'{model_name}: Predicted vs True RUL', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add RMSE annotation
        from sklearn.metrics import mean_squared_error
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        ax.text(0.05, 0.95, f'RMSE = {rmse:.2f}', 
               transform=ax.transAxes, fontsize=11,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_filename:
            filepath = os.path.join(self.save_dir, save_filename)
            plt.savefig(filepath, dpi=config.PLOT_CONFIG['dpi'], bbox_inches='tight')
            logger.info(f"Saved plot to {filepath}")
        
        plt.close()
    
    def plot_error_distribution(self,
                               y_true: np.ndarray,
                               y_pred: np.ndarray,
                               model_name: str = 'Model',
                               save_filename: str = None):
        """
        Plot prediction error distribution
        
        Args:
            y_true: True RUL values
            y_pred: Predicted RUL values
            model_name: Name of the model
            save_filename: Filename to save plot
        """
        errors = y_pred - y_true
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        ax1.hist(errors, bins=50, edgecolor='black', alpha=0.7)
        ax1.axvline(x=0, color='r', linestyle='--', lw=2, label='Zero Error')
        ax1.axvline(x=np.mean(errors), color='g', linestyle='--', lw=2, label=f'Mean Error: {np.mean(errors):.2f}')
        ax1.set_xlabel('Prediction Error (cycles)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title(f'{model_name}: Error Distribution', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2.boxplot(errors, vert=True)
        ax2.axhline(y=0, color='r', linestyle='--', lw=2)
        ax2.set_ylabel('Prediction Error (cycles)', fontsize=12)
        ax2.set_title(f'{model_name}: Error Box Plot', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_filename:
            filepath = os.path.join(self.save_dir, save_filename)
            plt.savefig(filepath, dpi=config.PLOT_CONFIG['dpi'], bbox_inches='tight')
            logger.info(f"Saved plot to {filepath}")
        
        plt.close()
    
    def plot_training_history(self,
                             history: Dict,
                             save_filename: str = None):
        """
        Plot LSTM training history
        
        Args:
            history: Training history dictionary
            save_filename: Filename to save plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss plot
        ax1.plot(history['loss'], label='Training Loss', lw=2)
        if history.get('val_loss'):
            ax1.plot(history['val_loss'], label='Validation Loss', lw=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss (MSE)', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # MAE plot
        ax2.plot(history['mae'], label='Training MAE', lw=2)
        if history.get('val_mae'):
            ax2.plot(history['val_mae'], label='Validation MAE', lw=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('MAE (cycles)', fontsize=12)
        ax2.set_title('Training and Validation MAE', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_filename:
            filepath = os.path.join(self.save_dir, save_filename)
            plt.savefig(filepath, dpi=config.PLOT_CONFIG['dpi'], bbox_inches='tight')
            logger.info(f"Saved plot to {filepath}")
        
        plt.close()
    
    def plot_engine_trajectory(self,
                              df: pd.DataFrame,
                              unit_ids: List[int],
                              rul_col: str = 'RUL',
                              pred_col: str = 'RUL_pred',
                              save_filename: str = None):
        """
        Plot RUL trajectories for specific engines
        
        Args:
            df: DataFrame with unit_id, time_cycles, RUL, RUL_pred
            unit_ids: List of engine unit IDs to plot
            rul_col: Name of true RUL column
            pred_col: Name of predicted RUL column
            save_filename: Filename to save plot
        """
        n_engines = len(unit_ids)
        fig, axes = plt.subplots(n_engines, 1, figsize=(12, 4*n_engines))
        
        if n_engines == 1:
            axes = [axes]
        
        for i, unit_id in enumerate(unit_ids):
            engine_data = df[df['unit_id'] == unit_id].sort_values('time_cycles')
            
            if len(engine_data) == 0:
                continue
            
            ax = axes[i]
            ax.plot(engine_data['time_cycles'], engine_data[rul_col], 
                   label='True RUL', lw=2, marker='o', markersize=4)
            if pred_col in engine_data.columns:
                ax.plot(engine_data['time_cycles'], engine_data[pred_col], 
                       label='Predicted RUL', lw=2, marker='x', markersize=4)
            
            ax.set_xlabel('Time (cycles)', fontsize=11)
            ax.set_ylabel('RUL (cycles)', fontsize=11)
            ax.set_title(f'Engine Unit {unit_id}: RUL Trajectory', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.axhline(y=config.MAINTENANCE_THRESHOLDS['critical'], 
                      color='r', linestyle='--', alpha=0.5, label='Critical Threshold')
        
        plt.tight_layout()
        
        if save_filename:
            filepath = os.path.join(self.save_dir, save_filename)
            plt.savefig(filepath, dpi=config.PLOT_CONFIG['dpi'], bbox_inches='tight')
            logger.info(f"Saved plot to {filepath}")
        
        plt.close()
    
    def plot_sensor_trends(self,
                          df: pd.DataFrame,
                          unit_id: int,
                          sensor_cols: List[str],
                          save_filename: str = None):
        """
        Plot sensor trends for a specific engine
        
        Args:
            df: DataFrame with sensor data
            unit_id: Engine unit ID
            sensor_cols: List of sensor column names to plot
            save_filename: Filename to save plot
        """
        engine_data = df[df['unit_id'] == unit_id].sort_values('time_cycles')
        
        if len(engine_data) == 0:
            logger.warning(f"No data found for engine {unit_id}")
            return
        
        n_sensors = len(sensor_cols)
        n_cols = 3
        n_rows = (n_sensors + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for i, sensor in enumerate(sensor_cols):
            if sensor not in engine_data.columns:
                continue
            
            ax = axes[i]
            ax.plot(engine_data['time_cycles'], engine_data[sensor], lw=2)
            ax.set_xlabel('Time (cycles)', fontsize=10)
            ax.set_ylabel('Value', fontsize=10)
            ax.set_title(sensor, fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_sensors, len(axes)):
            axes[i].axis('off')
        
        fig.suptitle(f'Engine Unit {unit_id}: Sensor Trends', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_filename:
            filepath = os.path.join(self.save_dir, save_filename)
            plt.savefig(filepath, dpi=config.PLOT_CONFIG['dpi'], bbox_inches='tight')
            logger.info(f"Saved plot to {filepath}")
        
        plt.close()


if __name__ == "__main__":
    # Test visualizer
    print("="*60)
    print("Testing RUL Visualizer")
    print("="*60)
    
    # Generate synthetic data
    np.random.seed(42)
    y_true = np.random.randint(0, 150, size=100).astype(float)
    y_pred = y_true + np.random.randn(100) * 15
    y_pred = np.maximum(y_pred, 0)
    
    # Create visualizer
    visualizer = RULVisualizer(save_dir='/tmp/rul_plots')
    
    # Test plots
    print("\n1. Prediction Scatter Plot")
    visualizer.plot_prediction_scatter(y_true, y_pred, 'LSTM', 'test_scatter.png')
    
    print("\n2. Error Distribution Plot")
    visualizer.plot_error_distribution(y_true, y_pred, 'LSTM', 'test_error.png')
    
    # Test training history
    print("\n3. Training History Plot")
    history = {
        'loss': np.linspace(100, 10, 50),
        'val_loss': np.linspace(110, 15, 50),
        'mae': np.linspace(30, 5, 50),
        'val_mae': np.linspace(35, 8, 50)
    }
    visualizer.plot_training_history(history, 'test_history.png')
    
    print("\n" + "="*60)
    print(f"Visualizer test complete! Plots saved to /tmp/rul_plots")
    print("="*60)

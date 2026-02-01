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


class InteractiveVisualizer:
    """
    Interactive visualization and dashboard data generator
    Creates fleet overviews, trend comparisons, and exportable reports
    """
    
    def __init__(self, save_dir: str = None):
        """
        Initialize interactive visualizer
        
        Args:
            save_dir: Directory to save outputs
        """
        self.save_dir = save_dir or config.PLOTS_DIR
        os.makedirs(self.save_dir, exist_ok=True)
        self.dashboard_data = {}
        logger.info(f"Initialized InteractiveVisualizer (output: {self.save_dir})")
    
    def create_dashboard_data(self,
                              predictions_df: pd.DataFrame,
                              rul_col: str = 'RUL_pred') -> Dict:
        """
        Prepare data for dashboard display
        
        Args:
            predictions_df: DataFrame with predictions
            rul_col: RUL prediction column name
            
        Returns:
            Dashboard data dictionary
        """
        logger.info("Creating dashboard data...")
        
        # Get latest RUL per engine
        if 'time_cycles' in predictions_df.columns:
            latest = predictions_df.groupby('unit_id').last().reset_index()
        else:
            latest = predictions_df
        
        # Health classification
        def classify(rul):
            if rul < 30:
                return 'Critical'
            elif rul < 50:
                return 'Warning'
            elif rul < 80:
                return 'Caution'
            else:
                return 'Healthy'
        
        latest['status'] = latest[rul_col].apply(classify)
        
        # Status summary
        status_counts = latest['status'].value_counts().to_dict()
        
        # Top critical engines
        critical_engines = latest[latest['status'] == 'Critical'].sort_values(rul_col)
        
        self.dashboard_data = {
            'summary': {
                'total_engines': len(latest),
                'status_distribution': status_counts,
                'avg_rul': float(latest[rul_col].mean()),
                'min_rul': float(latest[rul_col].min()),
                'max_rul': float(latest[rul_col].max())
            },
            'critical_engines': critical_engines[['unit_id', rul_col, 'status']].to_dict('records')[:10],
            'rul_histogram': {
                'bins': [0, 30, 50, 80, 100, 150],
                'counts': [
                    len(latest[latest[rul_col] < 30]),
                    len(latest[(latest[rul_col] >= 30) & (latest[rul_col] < 50)]),
                    len(latest[(latest[rul_col] >= 50) & (latest[rul_col] < 80)]),
                    len(latest[(latest[rul_col] >= 80) & (latest[rul_col] < 100)]),
                    len(latest[latest[rul_col] >= 100])
                ]
            }
        }
        
        return self.dashboard_data
    
    def plot_fleet_overview(self,
                           predictions_df: pd.DataFrame,
                           rul_col: str = 'RUL_pred',
                           save_filename: str = None):
        """
        Create fleet-wide health visualization
        
        Args:
            predictions_df: DataFrame with predictions
            rul_col: RUL column name
            save_filename: Output filename
        """
        logger.info("Plotting fleet overview...")
        
        # Get dashboard data if not available
        if not self.dashboard_data:
            self.create_dashboard_data(predictions_df, rul_col)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Status pie chart
        status_counts = self.dashboard_data['summary']['status_distribution']
        colors = {'Critical': 'red', 'Warning': 'orange', 'Caution': 'yellow', 'Healthy': 'green'}
        ax1 = axes[0, 0]
        wedges, _, autotexts = ax1.pie(
            status_counts.values(), 
            labels=status_counts.keys(),
            autopct='%1.1f%%',
            colors=[colors.get(k, 'gray') for k in status_counts.keys()],
            explode=[0.05 if k == 'Critical' else 0 for k in status_counts.keys()]
        )
        ax1.set_title('Fleet Health Distribution', fontsize=12, fontweight='bold')
        
        # 2. RUL histogram
        ax2 = axes[0, 1]
        if 'time_cycles' in predictions_df.columns:
            latest = predictions_df.groupby('unit_id').last().reset_index()
        else:
            latest = predictions_df
        ax2.hist(latest[rul_col], bins=20, edgecolor='black', alpha=0.7)
        ax2.axvline(x=30, color='r', linestyle='--', label='Critical Threshold')
        ax2.axvline(x=50, color='orange', linestyle='--', label='Warning Threshold')
        ax2.set_xlabel('RUL (cycles)')
        ax2.set_ylabel('Number of Engines')
        ax2.set_title('RUL Distribution', fontsize=12, fontweight='bold')
        ax2.legend()
        
        # 3. Summary statistics
        ax3 = axes[1, 0]
        summary = self.dashboard_data['summary']
        stats_text = [
            f"Total Engines: {summary['total_engines']}",
            f"Average RUL: {summary['avg_rul']:.1f} cycles",
            f"Minimum RUL: {summary['min_rul']:.1f} cycles",
            f"Maximum RUL: {summary['max_rul']:.1f} cycles",
            "",
            "Status Breakdown:",
        ]
        for status, count in summary['status_distribution'].items():
            stats_text.append(f"  • {status}: {count}")
        
        ax3.axis('off')
        ax3.text(0.1, 0.9, '\n'.join(stats_text), fontsize=11,
                transform=ax3.transAxes, verticalalignment='top',
                fontfamily='monospace')
        ax3.set_title('Fleet Statistics', fontsize=12, fontweight='bold')
        
        # 4. Critical engines bar chart
        ax4 = axes[1, 1]
        critical = self.dashboard_data['critical_engines'][:5]
        if critical:
            engine_ids = [str(e['unit_id']) for e in critical]
            ruls = [e[rul_col] for e in critical]
            colors_bar = ['red' if r < 30 else 'orange' for r in ruls]
            ax4.barh(engine_ids, ruls, color=colors_bar)
            ax4.set_xlabel('RUL (cycles)')
            ax4.set_ylabel('Engine ID')
            ax4.set_title('Top 5 Critical Engines', fontsize=12, fontweight='bold')
        else:
            ax4.text(0.5, 0.5, 'No Critical Engines', ha='center', va='center')
            ax4.axis('off')
        
        plt.suptitle('Fleet Health Overview Dashboard', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_filename:
            filepath = os.path.join(self.save_dir, save_filename)
            plt.savefig(filepath, dpi=config.PLOT_CONFIG['dpi'], bbox_inches='tight')
            logger.info(f"Saved fleet overview to {filepath}")
        
        plt.close()
    
    def plot_degradation_trends(self,
                               predictions_df: pd.DataFrame,
                               engine_ids: List[int] = None,
                               rul_col: str = 'RUL_pred',
                               save_filename: str = None):
        """
        Compare degradation trends across multiple engines
        
        Args:
            predictions_df: DataFrame with time series predictions
            engine_ids: List of engine IDs to compare
            rul_col: RUL column name
            save_filename: Output filename
        """
        logger.info("Plotting degradation trends...")
        
        if engine_ids is None:
            # Select engines with diverse RUL values
            latest = predictions_df.groupby('unit_id')[rul_col].last()
            engine_ids = [
                latest.idxmin(),  # Lowest RUL
                latest.idxmax(),  # Highest RUL
                latest.sort_values().index[len(latest)//2]  # Median RUL
            ]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(engine_ids)))
        
        for i, engine_id in enumerate(engine_ids):
            engine_data = predictions_df[predictions_df['unit_id'] == engine_id].sort_values('time_cycles')
            if len(engine_data) > 0:
                final_rul = engine_data[rul_col].iloc[-1]
                ax.plot(engine_data['time_cycles'], engine_data[rul_col],
                       label=f'Engine {engine_id} (RUL: {final_rul:.0f})',
                       color=colors[i], linewidth=2)
        
        ax.axhline(y=30, color='r', linestyle='--', alpha=0.5, label='Critical')
        ax.axhline(y=50, color='orange', linestyle='--', alpha=0.5, label='Warning')
        
        ax.set_xlabel('Time (cycles)', fontsize=12)
        ax.set_ylabel('RUL (cycles)', fontsize=12)
        ax.set_title('Engine Degradation Trends Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_filename:
            filepath = os.path.join(self.save_dir, save_filename)
            plt.savefig(filepath, dpi=config.PLOT_CONFIG['dpi'], bbox_inches='tight')
            logger.info(f"Saved degradation trends to {filepath}")
        
        plt.close()
    
    def generate_html_report(self,
                            predictions_df: pd.DataFrame,
                            model_name: str = 'RUL Model',
                            y_true: np.ndarray = None,
                            y_pred: np.ndarray = None,
                            filename: str = None) -> str:
        """
        Generate interactive HTML report
        
        Args:
            predictions_df: DataFrame with predictions
            model_name: Model name for report
            y_true: True RUL values (optional)
            y_pred: Predicted values (optional)
            filename: Output filename
            
        Returns:
            HTML string
        """
        logger.info("Generating HTML report...")
        
        from datetime import datetime
        
        # Get dashboard data
        if not self.dashboard_data:
            self.create_dashboard_data(predictions_df)
        
        summary = self.dashboard_data['summary']
        
        # Build HTML
        html_parts = [
            "<!DOCTYPE html>",
            "<html><head>",
            "<title>RUL Prediction Report</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }",
            ".container { max-width: 1000px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }",
            "h1 { color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }",
            "h2 { color: #555; margin-top: 30px; }",
            ".stat-box { display: inline-block; padding: 20px; margin: 10px; background: #e8f5e9; border-radius: 8px; text-align: center; }",
            ".stat-value { font-size: 28px; font-weight: bold; color: #2196F3; }",
            ".stat-label { color: #666; font-size: 14px; }",
            ".critical { color: red; font-weight: bold; }",
            ".warning { color: orange; }",
            ".healthy { color: green; }",
            "table { width: 100%; border-collapse: collapse; margin-top: 20px; }",
            "th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }",
            "th { background: #4CAF50; color: white; }",
            "tr:hover { background: #f5f5f5; }",
            "</style></head><body>",
            "<div class='container'>",
            f"<h1>🔧 {model_name} - Fleet Health Report</h1>",
            f"<p><em>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>",
            "<h2>📊 Fleet Summary</h2>",
            "<div class='stats'>",
            f"<div class='stat-box'><div class='stat-value'>{summary['total_engines']}</div><div class='stat-label'>Total Engines</div></div>",
            f"<div class='stat-box'><div class='stat-value'>{summary['avg_rul']:.1f}</div><div class='stat-label'>Avg RUL (cycles)</div></div>",
            f"<div class='stat-box'><div class='stat-value'>{summary['min_rul']:.0f}</div><div class='stat-label'>Min RUL</div></div>",
            f"<div class='stat-box'><div class='stat-value'>{summary['max_rul']:.0f}</div><div class='stat-label'>Max RUL</div></div>",
            "</div>",
            "<h2>🚨 Health Status</h2>",
            "<ul>"
        ]
        
        for status, count in summary['status_distribution'].items():
            css_class = status.lower() if status in ['Critical', 'Warning', 'Healthy'] else ''
            html_parts.append(f"<li class='{css_class}'>{status}: {count} engines</li>")
        
        html_parts.append("</ul>")
        
        # Critical engines table
        critical = self.dashboard_data.get('critical_engines', [])
        if critical:
            html_parts.extend([
                "<h2>⚠️ Critical Engines</h2>",
                "<table><tr><th>Engine ID</th><th>RUL (cycles)</th><th>Status</th></tr>"
            ])
            for eng in critical[:10]:
                html_parts.append(
                    f"<tr><td>{eng['unit_id']}</td><td>{eng.get('RUL_pred', 0):.0f}</td><td class='critical'>{eng['status']}</td></tr>"
                )
            html_parts.append("</table>")
        
        # Add model metrics if available
        if y_true is not None and y_pred is not None:
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            html_parts.extend([
                "<h2>📈 Model Performance</h2>",
                f"<div class='stat-box'><div class='stat-value'>{rmse:.2f}</div><div class='stat-label'>RMSE</div></div>",
                f"<div class='stat-box'><div class='stat-value'>{mae:.2f}</div><div class='stat-label'>MAE</div></div>",
                f"<div class='stat-box'><div class='stat-value'>{r2:.4f}</div><div class='stat-label'>R²</div></div>"
            ])
        
        html_parts.extend([
            "</div></body></html>"
        ])
        
        html = '\n'.join(html_parts)
        
        if filename:
            filepath = os.path.join(self.save_dir, filename)
            with open(filepath, 'w') as f:
                f.write(html)
            logger.info(f"HTML report saved to {filepath}")
        
        return html


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

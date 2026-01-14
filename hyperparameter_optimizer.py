"""
Hyperparameter Optimization using Optuna
Automatically tunes LSTM model hyperparameters for best performance
"""

import numpy as np
import pandas as pd
import optuna
from optuna.pruners import MedianPruner
from typing import Dict
import os
import config
from utils import setup_logging, generate_sequences
from models.lstm_model import LSTMModel

logger = setup_logging(__name__)


class HyperparameterOptimizer:
    """
    Hyperparameter optimizer using Optuna for LSTM models
    """
    
    def __init__(self, n_trials: int = 50, timeout: int = 3600):
        """
        Initialize hyperparameter optimizer
        
        Args:
            n_trials: Number of optimization trials
            timeout: Maximum optimization time in seconds
        """
        self.n_trials = n_trials
        self.timeout = timeout
        self.study = None
        self.best_params = None
        logger.info(f"Initialized Hyperparameter Optimizer with {n_trials} trials, "
                   f"{timeout}s timeout")
    
    def create_objective(self, X_train, y_train, X_val, y_val, num_features):
        """
        Create Optuna objective function for LSTM hyperparameter optimization
        
        Args:
            X_train, y_train: Training sequences and targets  
            X_val, y_val: Validation sequences and targets
            num_features: Number of input features
            
        Returns:
            Objective function
        """
        def objective(trial):
            """
            Objective function to minimize validation RMSE
            
            Args:
                trial: Optuna trial object
                
            Returns:
                Validation RMSE
            """
            # Sample hyperparameters
            sequence_length = trial.suggest_int('sequence_length', 20, 50, step=5)
            lstm_units_1 = trial.suggest_int('lstm_units_1', 50, 150, step=25)
            lstm_units_2 = trial.suggest_int('lstm_units_2', 25, 100, step=25)
            dropout_rate = trial.suggest_float('dropout', 0.1, 0.4, step=0.1)
            learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
            batch_size = trial.suggest_categorical('batch_size', [128, 256, 512])
            
            # Note: sequence_length change requires regenerating sequences
            # For simplicity, we'll use the fixed sequence from training data
            # In production, regenerate sequences for each trial
            
            try:
                # Build model with sampled hyperparameters
                model = LSTMModel(
                    sequence_length=X_train.shape[1],  # Use existing sequence length
                    num_features=num_features,
                    lstm_units=[lstm_units_1, lstm_units_2],
                    dropout_rate=dropout_rate,
                    learning_rate=learning_rate
                )
                model.build_model(num_features=num_features)
                
                # Train model (fewer epochs for faster trials)
                model.train(
                    X_train, y_train,
                    X_val, y_val,
                    epochs=30,  # Reduced for optimization
                    batch_size=batch_size,
                    patience=5,  # Earlier stopping for trials
                    verbose=0
                )
                
                # Evaluate on validation set
                metrics = model.evaluate(X_val, y_val)
                val_rmse = metrics['rmse']
                
                # Report intermediate value for pruning
                trial.report(val_rmse, step=1)
                
                # Optuna minimizes the objective
                return val_rmse
                
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return float('inf')  # Return worst score on failure
        
        return objective
    
    def optimize(self, X_train, y_train, X_val, y_val, num_features):
        """
        Run hyperparameter optimization
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            num_features: Number of features
            
        Returns:
            Dictionary with best hyperparameters
        """
        logger.info(f"Starting hyperparameter optimization (up to {self.n_trials} trials)...")
        
        # Create study
        sampler = optuna.samplers.TPESampler(seed=config.RANDOM_SEED)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        
        self.study = optuna.create_study(
            direction='minimize',  # Minimize RMSE
            sampler=sampler,
            pruner=pruner,
            study_name='lstm_rul_optimization'
        )
        
        # Create objective function
        objective = self.create_objective(X_train, y_train, X_val, y_val, num_features)
        
        # Optimize
        self.study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True,
            n_jobs=1  # Sequential trials (parallel would need separate environments)
        )
        
        # Get best parameters
        self.best_params = self.study.best_params
        best_rmse = self.study.best_value
        
        logger.info("="*60)
        logger.info("Optimization Complete!")
        logger.info(f"Best validation RMSE: {best_rmse:.2f}")
        logger.info(f"Best hyperparameters:")
        for key, value in self.best_params.items():
            logger.info(f"  {key}: {value}")
        logger.info("="*60)
        
        return self.best_params
    
    def get_optimization_history(self) -> pd.DataFrame:
        """
        Get optimization history as DataFrame
        
        Returns:
            DataFrame with trial history
        """
        if self.study is None:
            logger.warning("No optimization study available")
            return None
        
        trials_df = self.study.trials_dataframe()
        
        # Sort by value (RMSE)
        trials_df = trials_df.sort_values('value')
        
        return trials_df
    
    def plot_optimization_history(self, save_path: str = None):
        """
        Plot optimization history
        
        Args:
            save_path: Path to save plot
        """
        if self.study is None:
            logger.warning("No optimization study available")
            return
        
        import matplotlib.pyplot as plt
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot 1: Optimization history
        trials = self.study.trials_dataframe()
        ax1.plot(trials.index, trials['value'], 'b-', alpha=0.6)
        ax1.axhline(y=self.study.best_value, color='r', linestyle='--', 
                   label=f'Best: {self.study.best_value:.2f}')
        ax1.set_xlabel('Trial', fontsize=12)
        ax1.set_ylabel('Validation RMSE', fontsize=12)
        ax1.set_title('Optimization History', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Parameter importance
        try:
            from optuna.visualization.matplotlib import plot_param_importances
            importances = optuna.importance.get_param_importances(self.study)
            
            params = list(importances.keys())
            values = list(importances.values())
            
            ax2.barh(params, values)
            ax2.set_xlabel('Importance', fontsize=12)
            ax2.set_title('Hyperparameter Importance', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='x')
        except:
            ax2.text(0.5, 0.5, 'Parameter importance\nnot available', 
                    ha='center', va='center', fontsize=12)
            ax2.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=config.PLOT_CONFIG['dpi'], bbox_inches='tight')
            logger.info(f"Saved optimization history to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def save_best_params(self, filepath: str):
        """
        Save best hyperparameters to file
        
        Args:
            filepath: Path to save parameters
        """
        if self.best_params is None:
            logger.warning("No  best parameters available")
            return
        
        import json
        import os
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.best_params, f, indent=4)
        
        logger.info(f"Best hyperparameters saved to {filepath}")
    
    def load_best_params(self, filepath: str) -> Dict:
        """
        Load best hyperparameters from file
        
        Args:
            filepath: Path to saved parameters
            
        Returns:
            Dictionary of hyperparameters
        """
        import json
        
        with open(filepath, 'r') as f:
            params = json.load(f)
        
        self.best_params = params
        logger.info(f"Loaded best hyperparameters from {filepath}")
        
        return params


if __name__ == "__main__":
    # Test hyperparameter optimizer
    print("="*60)
    print("Testing Hyperparameter Optimizer")
    print("="*60)
    
    # Generate synthetic data
    np.random.seed(42)
    sequence_length = 30
    num_features = 20
    num_samples = 500
    
    X_train = np.random.randn(num_samples, sequence_length, num_features)
    y_train = np.random.randint(0, 150, size=num_samples).astype(float)
    X_val = np.random.randn(100, sequence_length, num_features)
    y_val = np.random.randint(0, 150, size=100).astype(float)
    
    # Create optimizer
    print("\nInitializing optimizer...")
    optimizer = HyperparameterOptimizer(n_trials=5, timeout=300)  # Small test
    
    # Run optimization
    print("\nRunning optimization (5 trials)...")
    best_params = optimizer.optimize(X_train, y_train, X_val, y_val, num_features)
    
    # Get history
    print("\nOptimization History:")
    history = optimizer.get_optimization_history()
    print(history[['number', 'value', 'params_lstm_units_1', 'params_dropout']].head())
    
    # Plot
    print("\nGenerating plots...")
    optimizer.plot_optimization_history('/tmp/optuna_history.png')
    
    # Save params
    optimizer.save_best_params('/tmp/best_params.json')
    
    print("\n" + "="*60)
    print("Hyperparameter optimizer test complete!")
    print("="*60)

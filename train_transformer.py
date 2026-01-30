"""
Training Script for Transformer Model (Phase 3)
Trains the Transformer model on C-MAPSS data and tracks experiments with MLflow
"""

import os
import numpy as np
import pandas as pd
import argparse
import mlflow
import mlflow.keras
from datetime import datetime

import config
from utils import setup_logging
from data_loader import DataLoader
from preprocessor import CMAPSSPreprocessor
from feature_engineer import FeatureEngineer
from models.transformer_model import TransformerModel
from visualizer import Visualizer

logger = setup_logging(__name__)


class TrainingMonitor:
    """
    Enhanced training progress monitor with checkpointing and analysis
    Tracks metrics, saves best models, and provides training insights
    """
    
    def __init__(self, 
                 model_name: str = 'model',
                 checkpoint_dir: str = None,
                 patience: int = 15,
                 min_delta: float = 0.001):
        """
        Initialize training monitor
        
        Args:
            model_name: Name of the model being trained
            checkpoint_dir: Directory for checkpoints (default: config.MODELS_DIR)
            patience: Early stopping patience
            min_delta: Minimum improvement for early stopping
        """
        self.model_name = model_name
        self.checkpoint_dir = checkpoint_dir or config.MODELS_DIR
        self.patience = patience
        self.min_delta = min_delta
        
        self.history = {
            'epoch': [],
            'loss': [],
            'val_loss': [],
            'mae': [],
            'val_mae': [],
            'learning_rate': []
        }
        
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        self.training_start_time = None
        self.epoch_times = []
        
        # Create checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        logger.info(f"Initialized TrainingMonitor for {model_name}")
    
    def on_training_start(self):
        """Called when training begins"""
        self.training_start_time = datetime.now()
        logger.info(f"Training started at {self.training_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    def log_epoch(self,
                  epoch: int,
                  loss: float,
                  val_loss: float,
                  metrics: dict = None,
                  learning_rate: float = None):
        """
        Log metrics for an epoch
        
        Args:
            epoch: Current epoch number
            loss: Training loss
            val_loss: Validation loss
            metrics: Additional metrics dict
            learning_rate: Current learning rate
        """
        import time
        
        self.history['epoch'].append(epoch)
        self.history['loss'].append(loss)
        self.history['val_loss'].append(val_loss)
        
        if metrics:
            if 'mae' in metrics:
                self.history['mae'].append(metrics['mae'])
            if 'val_mae' in metrics:
                self.history['val_mae'].append(metrics['val_mae'])
        
        if learning_rate:
            self.history['learning_rate'].append(learning_rate)
        
        # Track epoch time
        if len(self.epoch_times) > 0:
            epoch_time = time.time() - self.epoch_times[-1]
        else:
            epoch_time = 0
        self.epoch_times.append(time.time())
        
        # Check for improvement
        improved = False
        if val_loss < (self.best_val_loss - self.min_delta):
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
            improved = True
        else:
            self.epochs_without_improvement += 1
        
        # Logging
        improvement_marker = " ⭐" if improved else ""
        logger.info(
            f"Epoch {epoch}: loss={loss:.4f}, val_loss={val_loss:.4f}"
            f"{improvement_marker}"
        )
        
        return improved
    
    def should_stop(self) -> tuple:
        """
        Check if training should stop early
        
        Returns:
            (should_stop, reason)
        """
        if self.epochs_without_improvement >= self.patience:
            reason = f"No improvement for {self.patience} epochs"
            return True, reason
        return False, None
    
    def auto_checkpoint(self, model, epoch: int, save_best_only: bool = True):
        """
        Save checkpoint if conditions are met
        
        Args:
            model: Model to checkpoint
            epoch: Current epoch
            save_best_only: Only save if this is the best model
        """
        if save_best_only and epoch != self.best_epoch:
            return
        
        checkpoint_path = os.path.join(
            self.checkpoint_dir, 
            f"{self.model_name}_checkpoint_epoch{epoch}.h5"
        )
        
        if hasattr(model, 'save'):
            model.save(checkpoint_path)
        
        # Also save best model separately
        if epoch == self.best_epoch:
            best_path = os.path.join(
                self.checkpoint_dir,
                f"{self.model_name}_best.h5"
            )
            if hasattr(model, 'save'):
                model.save(best_path)
            logger.info(f"Best model saved to {best_path}")
    
    def get_training_summary(self) -> dict:
        """
        Get comprehensive training summary
        
        Returns:
            Dictionary with training statistics
        """
        if not self.history['epoch']:
            return {}
        
        training_duration = None
        if self.training_start_time:
            training_duration = (datetime.now() - self.training_start_time).total_seconds()
        
        return {
            'model_name': self.model_name,
            'total_epochs': len(self.history['epoch']),
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_val_loss,
            'final_loss': self.history['loss'][-1],
            'final_val_loss': self.history['val_loss'][-1],
            'improvement_ratio': (self.history['val_loss'][0] - self.best_val_loss) / self.history['val_loss'][0] * 100,
            'training_duration_seconds': training_duration,
            'avg_epoch_time_seconds': np.mean(np.diff(self.epoch_times)) if len(self.epoch_times) > 1 else None
        }
    
    def early_stop_analysis(self) -> dict:
        """
        Analyze training for early stopping insights
        
        Returns:
            Dictionary with early stopping analysis
        """
        if len(self.history['val_loss']) < 5:
            return {'status': 'insufficient_data'}
        
        val_losses = np.array(self.history['val_loss'])
        
        # Find optimal stopping point
        optimal_epoch = np.argmin(val_losses)
        
        # Calculate overfitting indicator
        if optimal_epoch < len(val_losses) - 1:
            overfit_increase = (val_losses[-1] - val_losses[optimal_epoch]) / val_losses[optimal_epoch] * 100
        else:
            overfit_increase = 0
        
        # Calculate convergence rate
        early_loss = np.mean(val_losses[:5])
        mid_loss = np.mean(val_losses[len(val_losses)//2:len(val_losses)//2+5]) if len(val_losses) > 10 else early_loss
        late_loss = np.mean(val_losses[-5:])
        
        convergence_rate = (early_loss - late_loss) / len(val_losses)
        
        return {
            'optimal_epoch': int(optimal_epoch) + 1,
            'actual_epochs': len(val_losses),
            'wasted_epochs': max(0, len(val_losses) - optimal_epoch - 1),
            'overfit_increase_pct': float(overfit_increase),
            'convergence_rate': float(convergence_rate),
            'recommendation': 'Consider reducing patience' if len(val_losses) - optimal_epoch > 10 else 'Patience seems appropriate'
        }
    
    def print_summary(self):
        """Print formatted training summary"""
        summary = self.get_training_summary()
        stop_analysis = self.early_stop_analysis()
        
        print("\n" + "="*60)
        print(f"TRAINING SUMMARY: {self.model_name}")
        print("="*60)
        
        if summary:
            print(f"\n📊 Training Statistics:")
            print(f"  Total Epochs: {summary['total_epochs']}")
            print(f"  Best Epoch: {summary['best_epoch']}")
            print(f"  Best Val Loss: {summary['best_val_loss']:.4f}")
            print(f"  Improvement: {summary['improvement_ratio']:.1f}%")
            
            if summary['training_duration_seconds']:
                mins = summary['training_duration_seconds'] / 60
                print(f"  Training Time: {mins:.1f} minutes")
        
        if stop_analysis.get('status') != 'insufficient_data':
            print(f"\n⏱️ Early Stopping Analysis:")
            print(f"  Optimal Stop: Epoch {stop_analysis['optimal_epoch']}")
            print(f"  Wasted Epochs: {stop_analysis['wasted_epochs']}")
            print(f"  Recommendation: {stop_analysis['recommendation']}")
        
        print("="*60 + "\n")


def train_transformer(dataset_name='FD001', epochs=None, batch_size=None, no_mlflow=False):
    """
    Train Transformer model pipeline
    """
    logger.info(f"Starting Phase 3 Transformer training for {dataset_name}")
    
    # Setup MLflow
    if not no_mlflow:
        mlflow.set_tracking_uri(config.MLFLOW_CONFIG['tracking_uri'])
        mlflow.set_experiment(config.MLFLOW_CONFIG['experiment_name'])
        mlflow.start_run(run_name=f"Transformer_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # Log params
        mlflow.log_param("dataset", dataset_name)
        mlflow.log_param("model_type", "Transformer")
        mlflow.log_params(config.TRANSFORMER_CONFIG)
    
    try:
        # 1. Load Data
        loader = DataLoader()
        train_data = loader.load_data(dataset_name, split='train')
        test_data = loader.load_data(dataset_name, split='test')
        test_rul = loader.load_data(dataset_name, split='rul')
        
        # 2. Preprocess
        preprocessor = CMAPSSPreprocessor()
        train_processed = preprocessor.preprocess_train(train_data)
        test_processed = preprocessor.preprocess_test(test_data, test_rul)
        
        # 3. Feature Engineering
        engineer = FeatureEngineer()
        train_featured = engineer.create_all_features(train_processed)
        test_featured = engineer.create_all_features(test_processed)
        
        # 4. Prepare Sequences
        sequence_length = config.LSTM_CONFIG['sequence_length']
        target_col = 'RUL'
        feature_cols = [c for c in train_featured.columns if c not in ['unit_id', 'time_cycle', 'RUL', 'label_cls']]
        
        logger.info(f"Generating sequences (length={sequence_length})...")
        
        # Training sequences
        X_train, y_train = [], []
        for unit_id in train_featured['unit_id'].unique():
            group = train_featured[train_featured['unit_id'] == unit_id]
            if len(group) >= sequence_length:
                data = group[feature_cols].values
                target = group[target_col].values
                
                # Sliding window
                for i in range(len(data) - sequence_length + 1):
                    X_train.append(data[i:i+sequence_length])
                    y_train.append(target[i+sequence_length-1])
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Testing sequences (last sequence per engine)
        X_test, y_test = [], []
        for unit_id in test_featured['unit_id'].unique():
            group = test_featured[test_featured['unit_id'] == unit_id]
            if len(group) >= sequence_length:
                data = group[feature_cols].values
                target = group[target_col].iloc[-1] # True RUL
                
                # Take last sequence
                X_test.append(data[-sequence_length:])
                y_test.append(target)
        
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        
        logger.info(f"Training data shape: {X_train.shape}")
        
        # Split validation
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=config.LSTM_CONFIG['validation_split'], random_state=config.RANDOM_SEED
        )
        
        # 5. Train Transformer
        model = TransformerModel(
            sequence_length=sequence_length,
            num_features=len(feature_cols)
        )
        
        model.build_model()
        model.train(
            X_train, y_train, 
            X_val, y_val,
            epochs=epochs,
            batch_size=batch_size
        )
        
        # 6. Evaluate
        metrics = model.evaluate(X_test, y_test)
        
        # Log metrics to MLflow
        if not no_mlflow:
            mlflow.log_metrics(metrics)
            
            # Save model
            model_path = os.path.join(config.MODELS_DIR, 'transformer_model.h5')
            model.save(model_path)
            mlflow.keras.log_model(model.model, "model")
            
            logger.info(f"Model saved to {model_path} and logged to MLflow")
            
    except Exception as e:
        logger.error(f"Training failed: {e}")
        if not no_mlflow:
            mlflow.end_run()
        raise
    
    if not no_mlflow:
        mlflow.end_run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Phase 3 Transformer Model")
    parser.add_argument('--dataset', type=str, default='FD001', help='Dataset to use (FD001-FD004)')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size')
    parser.add_argument('--no-mlflow', action='store_true', help='Disable MLflow tracking')
    
    args = parser.parse_args()
    
    train_transformer(
        dataset_name=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        no_mlflow=args.no_mlflow
    )

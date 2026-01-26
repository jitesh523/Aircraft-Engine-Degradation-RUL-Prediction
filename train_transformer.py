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

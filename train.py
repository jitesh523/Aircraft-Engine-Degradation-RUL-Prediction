"""
Main Training Pipeline for RUL Prediction Models
Trains baseline, LSTM, and anomaly detection models on NASA C-MAPSS data
"""

import numpy as np
import pandas as pd
import sys
import os
import argparse
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from utils import setup_logging, generate_sequences, save_model, save_scaler, save_results
from data_loader import load_dataset
from preprocessor import preprocess_data
from feature_engineer import engineer_features
from models.baseline_model import BaselineModel
from models.lstm_model import LSTMModel
from models.anomaly_detector import AnomalyDetector

logger = setup_logging(__name__)


def train_baseline_models(X_train, y_train, X_val, y_val, feature_names):
    """
    Train baseline models (Random Forest and Linear Regression)
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        feature_names: List of feature names
        
    Returns:
        Dictionary of trained models and metrics
    """
    logger.info("="*60)
    logger.info("Training Baseline Models")
    logger.info("="*60)
    
    results = {}
    
    # Random Forest
    logger.info("\n1. Training Random Forest...")
    rf_model = BaselineModel('random_forest')
    rf_model.train(X_train, y_train)
    rf_metrics = rf_model.evaluate(X_val, y_val)
    rf_model.save(os.path.join(config.MODELS_DIR, 'baseline_rf.pkl'))
    
    logger.info(f"Random Forest - RMSE: {rf_metrics['rmse']:.2f}, MAE: {rf_metrics['mae']:.2f}, R2: {rf_metrics['r2']:.4f}")
    results['random_forest'] = {'model': rf_model, 'metrics': rf_metrics}
    
    # Linear Regression
    logger.info("\n2. Training Linear Regression...")
    lr_model = BaselineModel('linear_regression')
    lr_model.train(X_train, y_train)
    lr_metrics = lr_model.evaluate(X_val, y_val)
    lr_model.save(os.path.join(config.MODELS_DIR, 'baseline_lr.pkl'))
    
    logger.info(f"Linear Regression - RMSE: {lr_metrics['rmse']:.2f}, MAE: {lr_metrics['mae']:.2f}, R2: {lr_metrics['r2']:.4f}")
    results['linear_regression'] = {'model': lr_model, 'metrics': lr_metrics}
    
    # Feature importance (RF only)
    feature_importance = rf_model.get_feature_importance(feature_names, top_k=20)
    logger.info("\nTop 10 Important Features (Random Forest):")
    for i, row in feature_importance.head(10).iterrows():
        logger.info(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")
    
    return results


def train_lstm_model(X_train_seq, y_train_seq, X_val_seq, y_val_seq, num_features):
    """
    Train LSTM model
    
    Args:
        X_train_seq, y_train_seq: Training sequences
        X_val_seq, y_val_seq: Validation sequences
        num_features: Number of features
        
    Returns:
        Trained LSTM model and metrics
    """
    logger.info("="*60)
    logger.info("Training LSTM Model")
    logger.info("="*60)
    
    # Build model
    lstm_model = LSTMModel()
    lstm_model.build_model(num_features=num_features)
    
    logger.info("\nModel Architecture:")
    lstm_model.summary()
    
    # Train model
    logger.info("\nTraining...")
    lstm_model.train(
        X_train_seq, y_train_seq,
        X_val_seq, y_val_seq,
        verbose=1
    )
    
    # Evaluate
    metrics = lstm_model.evaluate(X_val_seq, y_val_seq)
    logger.info(f"\nLSTM - RMSE: {metrics['rmse']:.2f}, MAE: {metrics['mae']:.2f}, R2: {metrics['r2']:.4f}")
    
    # Save model
    lstm_model.save(os.path.join(config.MODELS_DIR, 'lstm_model.h5'))
    
    return {'model': lstm_model, 'metrics': metrics}


def train_anomaly_detector(X_train, feature_names):
    """
    Train anomaly detector
    
    Args:
        X_train: Training features (healthy engines)
        feature_names: List of feature names
        
    Returns:
        Trained anomaly detector
    """
    logger.info("="*60)
    logger.info("Training Anomaly Detector")
    logger.info("="*60)
    
    detector = AnomalyDetector(config.ANOMALY_CONFIG['method'])
    detector.fit(X_train)
    
    # Save detector
    detector.save(os.path.join(config.MODELS_DIR, 'anomaly_detector.pkl'))
    
    logger.info("Anomaly detector trained and saved")
    
    return {'detector': detector}


def main(dataset_name='FD001', skip_baseline=False, skip_lstm=False, skip_anomaly=False):
    """
    Main training pipeline
    
    Args:
        dataset_name: Name of dataset to use ('FD001', 'FD002', 'FD003', 'FD004')
        skip_baseline: Skip baseline model training
        skip_lstm: Skip LSTM model training
        skip_anomaly: Skip anomaly detector training
    """
    logger.info("="*80)
    logger.info(f"AIRCRAFT ENGINE RUL PREDICTION - TRAINING PIPELINE")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)
    
    # Step 1: Load data
    logger.info("\n[Step 1/7] Loading dataset...")
    train_df, test_df, rul_df = load_dataset(dataset_name)
    logger.info(f"Loaded {len(train_df)} training samples, {len(test_df)} test samples")
    
    # Step 2: Preprocess data
    logger.info("\n[Step 2/7] Preprocessing data...")
    preprocessed = preprocess_data(train_df, test_df, rul_df)
    train_prep = preprocessed['train']
    val_prep = preprocessed['validation']
    test_prep = preprocessed['test']
    
    # Step 3: Feature engineering
    logger.info("\n[Step 3/7] Engineering features...")
    train_eng = engineer_features(train_prep)['train']
    val_eng = engineer_features(val_prep)['train']
    test_eng = engineer_features(test_prep)['train']
    
    # Get feature columns
    feature_cols = preprocessed['feature_columns']
    # Add engineered features
    engineered_cols = [col for col in train_eng.columns 
                      if 'rolling' in col or 'roc' in col or 'health' in col]
    feature_cols = feature_cols + engineered_cols
    
    logger.info(f"Total features: {len(feature_cols)}")
    
    # Prepare training data for baseline models
    X_train = train_eng[feature_cols].values
    y_train = train_eng['RUL'].values
    X_val = val_eng[feature_cols].values
    y_val = val_eng['RUL'].values
    
    # Train baseline models
    if not skip_baseline:
        logger.info("\n[Step 4/7] Training baseline models...")
        baseline_results = train_baseline_models(X_train, y_train, X_val, y_val, feature_cols)
        
        # Save baseline metrics
        baseline_metrics = {
            'random_forest': baseline_results['random_forest']['metrics'],
            'linear_regression': baseline_results['linear_regression']['metrics']
        }
        save_results(baseline_metrics, os.path.join(config.RESULTS_DIR, 'baseline_metrics.json'))
    else:
        logger.info("\n[Step 4/7] Skipping baseline models...")
    
    # Train LSTM model
    if not skip_lstm:
        logger.info("\n[Step 5/7] Preparing sequences for LSTM...")
        
        # Generate sequences
        X_train_seq, y_train_seq = generate_sequences(
            train_eng, 
            config.LSTM_CONFIG['sequence_length'], 
            feature_cols,
            'RUL'
        )
        X_val_seq, y_val_seq = generate_sequences(
            val_eng,
            config.LSTM_CONFIG['sequence_length'],
            feature_cols,
            'RUL'
        )
        
        logger.info(f"Training sequences: {X_train_seq.shape}")
        logger.info(f"Validation sequences: {X_val_seq.shape}")
        
        lstm_results = train_lstm_model(
            X_train_seq, y_train_seq, 
            X_val_seq, y_val_seq,
            num_features=len(feature_cols)
        )
        
        # Save LSTM metrics
        save_results(lstm_results['metrics'], os.path.join(config.RESULTS_DIR, 'lstm_metrics.json'))
    else:
        logger.info("\n[Step 5/7] Skipping LSTM model...")
    
    # Train anomaly detector
    if not skip_anomaly:
        logger.info("\n[Step 6/7] Training anomaly detector...")
        
        # Use only healthy engines (high RUL) for training
        healthy_mask = train_eng['RUL'] > 80
        X_healthy = train_eng[healthy_mask][feature_cols].values
        
        logger.info(f"Training on {len(X_healthy)} healthy samples")
        anomaly_results = train_anomaly_detector(X_healthy, feature_cols)
    else:
        logger.info("\n[Step 6/7] Skipping anomaly detector...")
    
    # Save preprocessor and scalers
    logger.info("\n[Step 7/7] Saving preprocessor and scalers...")
    preprocessed['preprocessor'].save_scaler(os.path.join(config.MODELS_DIR, 'scaler.pkl'))
    
    # Save feature column names
    feature_info = {
        'feature_columns': feature_cols,
        'num_features': len(feature_cols),
        'sequence_length': config.LSTM_CONFIG['sequence_length']
    }
    save_results(feature_info, os.path.join(config.MODELS_DIR, 'feature_info.json'))
    
    logger.info("="*80)
    logger.info("TRAINING COMPLETE!")
    logger.info(f"Models saved to: {config.MODELS_DIR}")
    logger.info(f"Results saved to: {config.RESULTS_DIR}")
    logger.info("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train RUL prediction models')
    parser.add_argument('--dataset', type=str, default='FD001',
                       choices=['FD001', 'FD002', 'FD003', 'FD004'],
                       help='Dataset to use for training')
    parser.add_argument('--skip-baseline', action='store_true',
                       help='Skip baseline model training')
    parser.add_argument('--skip-lstm', action='store_true',
                       help='Skip LSTM model training')
    parser.add_argument('--skip-anomaly', action='store_true',
                       help='Skip anomaly detector training')
    
    args = parser.parse_args()
    
    main(
        dataset_name=args.dataset,
        skip_baseline=args.skip_baseline,
        skip_lstm=args.skip_lstm,
        skip_anomaly=args.skip_anomaly
    )

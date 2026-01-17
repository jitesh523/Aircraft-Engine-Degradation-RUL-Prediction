"""
Enhanced Training Pipeline with MLflow and Advanced Ensemble Methods (Phase 1)
Trains gradient boosting models, stacking ensemble, and logs everything to MLflow
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
from mlflow_tracker import MLflowTracker
from models.gradient_boosting_models import XGBoostRULModel, LightGBMRULModel, CatBoostRULModel
from models.stacking_ensemble import StackingEnsemble, WeightedAverageEnsemble
from models.baseline_model import BaselineModel
from models.lstm_model import LSTMModel
from sklearn.linear_model import Ridge

logger = setup_logging(__name__)


def train_phase1_models(X_train, y_train, X_val, y_val, feature_names, mlflow_tracker=None):
    """
    Train Phase 1 models: XGBoost, LightGBM, CatBoost
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        feature_names: List of feature names
        mlflow_tracker: MLflow tracker instance
        
    Returns:
        Dictionary of trained models and metrics
    """
    logger.info("="*60)
    logger.info("Training Phase 1: Gradient Boosting Models")
    logger.info("="*60)
    
    results = {}
    
    # 1. XGBoost
    logger.info("\n1. Training XGBoost...")
    xgb_model = XGBoostRULModel(**config.XGBOOST_CONFIG)
    xgb_model.fit(X_train, y_train, 
                  eval_set=[(X_val, y_val)],
                  early_stopping_rounds=50,
                  verbose=False)
    
    xgb_metrics = {
        'rmse': np.sqrt(np.mean((xgb_model.predict(X_val) - y_val)**2)),
        'mae': np.mean(np.abs(xgb_model.predict(X_val) - y_val)),
        'r2': 1 - np.sum((y_val - xgb_model.predict(X_val))**2) / np.sum((y_val - np.mean(y_val))**2)
    }
    xgb_model.save(os.path.join(config.MODELS_DIR, 'xgboost_model.pkl'))
    logger.info(f"XGBoost - RMSE: {xgb_metrics['rmse']:.2f}, MAE: {xgb_metrics['mae']:.2f}, R2: {xgb_metrics['r2']:.4f}")
    
    if mlflow_tracker:
        mlflow_tracker.log_training_session(
            model=xgb_model.model,
            model_type='sklearn',
            params=config.XGBOOST_CONFIG,
            metrics=xgb_metrics,
            model_name='XGBoost_RUL'
        )
    
    results['xgboost'] = {'model': xgb_model, 'metrics': xgb_metrics}
    
    # 2. LightGBM
    logger.info("\n2. Training LightGBM...")
    lgb_model = LightGBMRULModel(**config.LIGHTGBM_CONFIG)
    lgb_model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  early_stopping_rounds=50,
                  verbose=False)
    
    lgb_metrics = {
        'rmse': np.sqrt(np.mean((lgb_model.predict(X_val) - y_val)**2)),
        'mae': np.mean(np.abs(lgb_model.predict(X_val) - y_val)),
        'r2': 1 - np.sum((y_val - lgb_model.predict(X_val))**2) / np.sum((y_val - np.mean(y_val))**2)
    }
    lgb_model.save(os.path.join(config.MODELS_DIR, 'lightgbm_model.pkl'))
    logger.info(f"LightGBM - RMSE: {lgb_metrics['rmse']:.2f}, MAE: {lgb_metrics['mae']:.2f}, R2: {lgb_metrics['r2']:.4f}")
    
    if mlflow_tracker:
        mlflow_tracker.log_training_session(
            model=lgb_model.model,
            model_type='sklearn',
            params=config.LIGHTGBM_CONFIG,
            metrics=lgb_metrics,
           model_name='LightGBM_RUL'
        )
    
    results['lightgbm'] = {'model': lgb_model, 'metrics': lgb_metrics}
    
    # 3. CatBoost
    logger.info("\n3. Training CatBoost...")
    cb_model = CatBoostRULModel(**config.CATBOOST_CONFIG)
    cb_model.fit(X_train, y_train,
                 eval_set=[(X_val, y_val)],
                 early_stopping_rounds=50,
                 verbose=False)
    
    cb_metrics = {
        'rmse': np.sqrt(np.mean((cb_model.predict(X_val) - y_val)**2)),
        'mae': np.mean(np.abs(cb_model.predict(X_val) - y_val)),
        'r2': 1 - np.sum((y_val - cb_model.predict(X_val))**2) / np.sum((y_val - np.mean(y_val))**2)
    }
    cb_model.save(os.path.join(config.MODELS_DIR, 'catboost_model.cbm'))
    logger.info(f"CatBoost - RMSE: {cb_metrics['rmse']:.2f}, MAE: {cb_metrics['mae']:.2f}, R2: {cb_metrics['r2']:.4f}")
    
    if mlflow_tracker:
        mlflow_tracker.log_training_session(
            model=cb_model.model,
            model_type='sklearn',
            params=config.CATBOOST_CONFIG,
            metrics=cb_metrics,
            model_name='CatBoost_RUL'
        )
    
    results['catboost'] = {'model': cb_model, 'metrics': cb_metrics}
    
    return results


def train_stacking_ensemble(base_models, X_train, y_train, X_val, y_val, mlflow_tracker=None):
    """
    Train stacking ensemble
    
    Args:
        base_models: List of (name, model) tuples
        X_train, y_train: Training data
        X_val, y_val: Validation data
        mlflow_tracker: MLflow tracker instance
        
    Returns:
        Trained ensemble and metrics
    """
    logger.info("="*60)
    logger.info("Training Stacking Ensemble")
    logger.info("="*60)
    
    # Create meta-learner
    if config.STACKING_CONFIG['meta_learner'] == 'ridge':
        meta_learner = Ridge(**config.STACKING_CONFIG['meta_learner_params']['ridge'])
    else:
        from sklearn.neural_network import MLPRegressor
        meta_learner = MLPRegressor(**config.STACKING_CONFIG['meta_learner_params']['mlp'])
    
    # Create and train ensemble
    ensemble = StackingEnsemble(
        base_models=base_models,
        meta_learner=meta_learner,
        n_folds=config.STACKING_CONFIG['n_folds']
    )
    
    ensemble.fit(X_train, y_train, verbose=True)
    
    # Evaluate
    predictions = ensemble.predict(X_val)
    metrics = {
        'rmse': np.sqrt(np.mean((predictions - y_val)**2)),
        'mae': np.mean(np.abs(predictions - y_val)),
        'r2': 1 - np.sum((y_val - predictions)**2) / np.sum((y_val - np.mean(y_val))**2)
    }
    
    logger.info(f"\nStacking Ensemble - RMSE: {metrics['rmse']:.2f}, MAE: {metrics['mae']:.2f}, R2: {metrics['r2']:.4f}")
    
    # Get all model scores
    scores_df = ensemble.get_base_model_scores(X_val, y_val)
    logger.info("\nModel Comparison:")
    logger.info(scores_df.to_string(index=False))
    
    # Save ensemble
    ensemble.save(os.path.join(config.MODELS_DIR, 'stacking_ensemble.pkl'))
    
    if mlflow_tracker:
        mlflow_tracker.log_training_session(
            model=ensemble,
            model_type='sklearn',
            params=config.STACKING_CONFIG,
            metrics=metrics,
            model_name='Stacking_Ensemble_RUL'
        )
    
    return {'model': ensemble, 'metrics': metrics, 'scores': scores_df}


def main(dataset_name='FD001', use_mlflow=True):
    """
    Main Phase 1 training pipeline
    
    Args:
        dataset_name: Name of dataset to use
        use_mlflow: Whether to use MLflow tracking
    """
    logger.info("="*80)
    logger.info(f"PHASE 1: ADVANCED ENSEMBLE METHODS - TRAINING PIPELINE")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)
    
    # Initialize MLflow
    mlflow_tracker = None
    if use_mlflow:
        logger.info("\n[MLflow] Initializing experiment tracking...")
        mlflow_tracker = MLflowTracker(**config.MLFLOW_CONFIG)
        logger.info(f"MLflow tracking URI: {mlflow_tracker.tracking_uri}")
    
    # Step 1: Load data
    logger.info("\n[Step 1/5] Loading dataset...")
    train_df, test_df, rul_df = load_dataset(dataset_name)
    logger.info(f"Loaded {len(train_df)} training samples, {len(test_df)} test samples")
    
    # Step 2: Preprocess data
    logger.info("\n[Step 2/5] Preprocessing data...")
    preprocessed = preprocess_data(train_df, test_df, rul_df)
    train_prep = preprocessed['train']
    val_prep = preprocessed['validation']
    
    # Step 3: Feature engineering
    logger.info("\n[Step 3/5] Engineering features...")
    train_eng = engineer_features(train_prep)['train']
    val_eng = engineer_features(val_prep)['train']
    
    # Get feature columns
    feature_cols = preprocessed['feature_columns']
    engineered_cols = [col for col in train_eng.columns 
                      if 'rolling' in col or 'roc' in col or 'health' in col]
    feature_cols = feature_cols + engineered_cols
    
    logger.info(f"Total features: {len(feature_cols)}")
    
    # Prepare data
    X_train = train_eng[feature_cols].values
    y_train = train_eng['RUL'].values
    X_val = val_eng[feature_cols].values
    y_val = val_eng['RUL'].values
    
    # Step 4: Train gradient boosting models
    logger.info("\n[Step 4/5] Training gradient boosting models...")
    gb_results = train_phase1_models(X_train, y_train, X_val, y_val, 
                                     feature_cols, mlflow_tracker)
    
    # Save metrics
    gb_metrics = {
        'xgboost': gb_results['xgboost']['metrics'],
        'lightgbm': gb_results['lightgbm']['metrics'],
        'catboost': gb_results['catboost']['metrics']
    }
    save_results(gb_metrics, os.path.join(config.RESULTS_DIR, 'phase1_gradient_boosting_metrics.json'))
    
    # Step 5: Train stacking ensemble
    logger.info("\n[Step 5/5] Training stacking ensemble...")
    
    # Prepare base models for stacking
    base_models = [
        ('xgboost', gb_results['xgboost']['model']),
        ('lightgbm', gb_results['lightgbm']['model']),
        ('catboost', gb_results['catboost']['model'])
    ]
    
    # Add baseline Random Forest if available
    if os.path.exists(os.path.join(config.MODELS_DIR, 'baseline_rf.pkl')):
        from sklearn.ensemble import RandomForestRegressor
        import pickle
        with open(os.path.join(config.MODELS_DIR, 'baseline_rf.pkl'), 'rb') as f:
            rf_model = pickle.load(f)
        base_models.append(('random_forest', rf_model.model))
    
    ensemble_results = train_stacking_ensemble(base_models, X_train, y_train, 
                                               X_val, y_val, mlflow_tracker)
    
    # Save ensemble metrics
    save_results(ensemble_results['metrics'], 
                os.path.join(config.RESULTS_DIR, 'stacking_ensemble_metrics.json'))
    ensemble_results['scores'].to_csv(
        os.path.join(config.RESULTS_DIR, 'model_comparison.csv'), index=False
    )
    
    if use_mlflow:
        logger.info(f"\n[MLflow] View results at: {mlflow_tracker.tracking_uri}")
        logger.info("Run: mlflow ui --port 5000")
    
    logger.info("="*80)
    logger.info("PHASE 1 TRAINING COMPLETE!")
    logger.info(f"Models saved to: {config.MODELS_DIR}")
    logger.info(f"Results saved to: {config.RESULTS_DIR}")
    logger.info("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Phase 1 models')
    parser.add_argument('--dataset', type=str, default='FD001',
                       choices=['FD001', 'FD002', 'FD003', 'FD004'],
                       help='Dataset to use for training')
    parser.add_argument('--no-mlflow', action='store_true',
                       help='Disable MLflow tracking')
    
    args = parser.parse_args()
    
    main(dataset_name=args.dataset, use_mlflow=not args.no_mlflow)

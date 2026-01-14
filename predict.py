"""
Prediction Script for RUL Estimation
Makes predictions using trained models
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
from utils import setup_logging, generate_sequences_for_prediction, load_model, load_scaler, load_results
from data_loader import load_dataset
from preprocessor import CMAPSSPreprocessor
from feature_engineer import engineer_features
from evaluator import RULEvaluator
from visualizer import RULVisualizer
from maintenance_planner import MaintenancePlanner
from models.lstm_model import LSTMModel

logger = setup_logging(__name__)


def load_trained_models(models_dir: str = None):
    """
    Load trained models and preprocessor
    
    Args:
        models_dir: Directory containing saved models
        
    Returns:
        Dictionary with loaded models and metadata
    """
    if models_dir is None:
        models_dir = config.MODELS_DIR
    
    logger.info(f"Loading trained models from {models_dir}...")
    
    # Load feature info
    feature_info = load_results(os.path.join(models_dir, 'feature_info.json'))
    
    # Load preprocessor/scaler
    preprocessor = CMAPSSPreprocessor()
    preprocessor.load_scaler(os.path.join(models_dir, 'scaler.pkl'))
    
    # Load LSTM model
    lstm_model = LSTMModel()
    lstm_model.load(os.path.join(models_dir, 'lstm_model.h5'))
    
    logger.info("Models loaded successfully")
    
    return {
        'lstm': lstm_model,
        'preprocessor': preprocessor,
        'feature_columns': feature_info['feature_columns'],
        'sequence_length': feature_info['sequence_length']
    }


def predict_rul(dataset_name='FD001', visualize=True):
    """
    Make RUL predictions on test data
    
    Args:
        dataset_name: Name of dataset
        visualize: Whether to create visualization plots
    """
    logger.info("="*80)
    logger.info(f"AIRCRAFT ENGINE RUL PREDICTION - INFERENCE")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)
    
    # Load test data
    logger.info("\n[Step 1/6] Loading test data...")
    train_df, test_df, rul_df = load_dataset(dataset_name)
    
    # Load trained models
    logger.info("\n[Step 2/6] Loading trained models...")
    models = load_trained_models()
    
    # Preprocess test data
    logger.info("\n[Step 3/6] Preprocessing test data...")
    preprocessor = models['preprocessor']
    test_prep, rul_labels = preprocessor.prepare_test_data(test_df, rul_df)
    
    # Normalize (using fitted scaler from training)
    test_norm, _ = preprocessor.normalize_features(test_prep, fit=False)
    
    # Feature engineering
    logger.info("\n[Step 4/6] Engineering features...")
    test_eng = engineer_features(test_norm)['train']
    
    # Generate sequences for LSTM
    logger.info("\n[Step 5/6] Making predictions...")
    X_test_seq, unit_ids = generate_sequences_for_prediction(
        test_eng,
        models['sequence_length'],
        models['feature_columns']
    )
    
    # Predict with LSTM
    lstm_predictions = models['lstm'].predict(X_test_seq)
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'unit_id': unit_ids,
        'RUL_true': rul_labels.set_index('unit_id').loc[unit_ids, 'RUL'].values,
        'RUL_pred': lstm_predictions
    })
    
    # Evaluate predictions
    logger.info("\n[Step 6/6] Evaluating predictions...")
    evaluator = RULEvaluator()
    metrics = evaluator.evaluate_predictions(
        results_df['RUL_true'].values,
        results_df['RUL_pred'].values,
        'LSTM'
    )
    
    # Check performance targets
    evaluator.check_performance_targets(metrics)
    
    # Visualizations
    if visualize:
        logger.info("\nCreating visualizations...")
        visualizer = RULVisualizer()
        
        # Prediction scatter plot
        visualizer.plot_prediction_scatter(
            results_df['RUL_true'].values,
            results_df['RUL_pred'].values,
            'LSTM Model',
            f'{dataset_name}_prediction_scatter.png'
        )
        
        # Error distribution
        visualizer.plot_error_distribution(
            results_df['RUL_true'].values,
            results_df['RUL_pred'].values,
            'LSTM Model',
            f'{dataset_name}_error_distribution.png'
        )
    
    # Maintenance planning
    logger.info("\nGenerating maintenance recommendations...")
    planner = MaintenancePlanner()
    maintenance_schedule = planner.create_maintenance_schedule(
        results_df[['unit_id', 'RUL_pred']]
    )
    
    # Simulate maintenance strategies
    logger.info("\nSimulating maintenance strategies...")
    
    # Traditional maintenance
    actual_lifetimes = rul_labels['RUL'].values
    trad_results = planner.simulate_traditional_maintenance(actual_lifetimes)
    
    # Predictive maintenance
    pred_results = planner.simulate_predictive_maintenance(
        results_df['RUL_pred'].values,
        results_df['RUL_true'].values
    )
    
    # Compare strategies
    comparison = planner.compare_strategies(trad_results, pred_results)
    
    # Save results
    logger.info("\nSaving results...")
    results_df.to_csv(os.path.join(config.RESULTS_DIR, f'{dataset_name}_predictions.csv'), index=False)
    maintenance_schedule.to_csv(os.path.join(config.RESULTS_DIR, f'{dataset_name}_maintenance_schedule.csv'), index=False)
    comparison.to_csv(os.path.join(config.RESULTS_DIR, f'{dataset_name}_strategy_comparison.csv'), index=False)
    
    from utils import save_results
    save_results(metrics, os.path.join(config.RESULTS_DIR, f'{dataset_name}_test_metrics.json'))
    
    logger.info("="*80)
    logger.info("PREDICTION COMPLETE!")
    logger.info(f"Results saved to: {config.RESULTS_DIR}")
    if visualize:
        logger.info(f"Plots saved to: {config.PLOTS_DIR}")
    logger.info("="*80)
    
    return results_df, metrics, maintenance_schedule


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict RUL for test engines')
    parser.add_argument('--dataset', type=str, default='FD001',
                       choices=['FD001', 'FD002', 'FD003', 'FD004'],
                       help='Dataset to use for prediction')
    parser.add_argument('--no-viz', action='store_true',
                       help='Skip visualization')
    
    args = parser.parse_args()
    
    predict_rul(
        dataset_name=args.dataset,
        visualize=not args.no_viz
    )

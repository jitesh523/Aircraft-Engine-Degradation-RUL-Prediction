"""
Hyperparameter optimization script for LSTM model
Runs Optuna optimization to find best LSTM configuration
"""

import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from utils import setup_logging, generate_sequences
from data_loader import load_dataset
from preprocessor import preprocess_data
from feature_engineer import engineer_features
from hyperparameter_optimizer import HyperparameterOptimizer

logger = setup_logging(__name__)


def main(dataset_name='FD001', n_trials=50, timeout=3600):
    """
    Run hyperparameter optimization
    
    Args:
        dataset_name: Dataset to use for optimization
        n_trials: Number of optimization trials
        timeout: Maximum optimization time in seconds
    """
    logger.info("="*80)
    logger.info(f"HYPERPARAMETER OPTIMIZATION - DATASET: {dataset_name}")
    logger.info("="*80)
    
    # Step 1: Load and preprocess data
    logger.info("\n[Step 1/4] Loading and preprocessing data...")
    train_df, test_df, rul_df = load_dataset(dataset_name)
    preprocessed = preprocess_data(train_df, test_df, rul_df)
    
    # Step 2: Engineer features
    logger.info("\n[Step 2/4] Engineering features...")
    train_eng = engineer_features(preprocessed['train'])['train']
    val_eng = engineer_features(preprocessed['validation'])['train']
    
    # Get feature columns
    feature_cols = preprocessed['feature_columns']
    engineered_cols = [col for col in train_eng.columns 
                      if 'rolling' in col or 'roc' in col or 'health' in col]
    feature_cols = feature_cols + engineered_cols
    
    logger.info(f"Total features: {len(feature_cols)}")
    
    # Step 3: Generate sequences
    logger.info("\n[Step 3/4] Generating sequences...")
    X_train, y_train = generate_sequences(
        train_eng,
        config.LSTM_CONFIG['sequence_length'],
        feature_cols,
        'RUL'
    )
    
    X_val, y_val = generate_sequences(
        val_eng,
        config.LSTM_CONFIG['sequence_length'],
        feature_cols,
        'RUL'
    )
    
    logger.info(f"Training sequences: {X_train.shape}")
    logger.info(f"Validation sequences: {X_val.shape}")
    
    # Step 4: Run optimization
    logger.info("\n[Step 4/4] Running hyperparameter optimization...")
    logger.info(f"This may take up to {timeout}s ({timeout/60:.0f} minutes)")
    
    optimizer = HyperparameterOptimizer(n_trials=n_trials, timeout=timeout)
    best_params = optimizer.optimize(
        X_train, y_train,
        X_val, y_val,
        num_features=len(feature_cols)
    )
    
    # Save results
    logger.info("\nSaving optimization results...")
    
    # Save best parameters
    optimizer.save_best_params(
        os.path.join(config.RESULTS_DIR, f'{dataset_name}_best_hyperparams.json')
    )
    
    # Save optimization history
    history = optimizer.get_optimization_history()
    history.to_csv(
        os.path.join(config.RESULTS_DIR, f'{dataset_name}_optimization_history.csv'),
        index=False
    )
    
    # Save optimization plots
    optimizer.plot_optimization_history(
        os.path.join(config.PLOTS_DIR, f'{dataset_name}_optimization_history.png')
    )
    
    logger.info("="*80)
    logger.info("OPTIMIZATION COMPLETE!")
    logger.info(f"Results saved to: {config.RESULTS_DIR}")
    logger.info(f"Plots saved to: {config.PLOTS_DIR}")
    logger.info("="*80)
    
    # Print summary
    logger.info("\n📊 Optimization Summary:")
    logger.info(f"  Trials completed: {len(optimizer.study.trials)}")
    logger.info(f"  Best RMSE: {optimizer.study.best_value:.2f}")
    logger.info(f"  Best hyperparameters:")
    for key, value in best_params.items():
        logger.info(f"    - {key}: {value}")
    
    return best_params, optimizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optimize LSTM hyperparameters')
    parser.add_argument('--dataset', type=str, default='FD001',
                       choices=['FD001', 'FD002', 'FD003', 'FD004'],
                       help='Dataset to use for optimization')
    parser.add_argument('--trials', type=int, default=50,
                       help='Number of optimization trials')
    parser.add_argument('--timeout', type=int, default=3600,
                       help='Maximum optimization time in seconds')
    
    args = parser.parse_args()
    
    main(
        dataset_name=args.dataset,
        n_trials=args.trials,
        timeout=args.timeout
    )

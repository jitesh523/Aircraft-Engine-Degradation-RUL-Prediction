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
from models.baseline_model import BaselineModel
from ensemble_predictor import EnsemblePredictor
from uncertainty_quantifier import UncertaintyQuantifier
from shap_explainer import SHAPExplainer

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
    
    # Load baseline models
    rf_model = BaselineModel('random_forest')
    rf_model.load(os.path.join(models_dir, 'baseline_rf.pkl'))
    
    lr_model = BaselineModel('linear_regression')
    lr_model.load(os.path.join(models_dir, 'baseline_lr.pkl'))
    
    logger.info("Models loaded successfully")
    
    return {
        'lstm': lstm_model,
        'random_forest': rf_model,
        'linear_regression': lr_model,
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
    
    # Generate uncertainty estimates for LSTM predictions
    logger.info("\nGenerating uncertainty estimates for LSTM predictions...")
    uncertainty = UncertaintyQuantifier(n_iterations=100, confidence_level=0.95)
    lstm_mean, lstm_lower, lstm_upper = uncertainty.predict_with_uncertainty(
        models['lstm'],
        X_test_seq,
        is_keras=True
    )
    
    # Predict with baseline models (on flattened last timestep features)
    # Use last timestep of each sequence for baseline models
    X_test_flat = test_eng.groupby('unit_id')[models['feature_columns']].last().loc[unit_ids].values
    
    rf_predictions = models['random_forest'].predict(X_test_flat)
    lr_predictions = models['linear_regression'].predict(X_test_flat)
    
    # Create predictions dictionary
    all_predictions = {
        'LSTM': lstm_predictions,
        'Random Forest': rf_predictions,
        'Linear Regression': lr_predictions
    }
    
    # Create ensemble predictor
    logger.info("\nCreating ensemble prediction...")
    ensemble = EnsemblePredictor('weighted_average')
    
    # Optimize weights on a validation split (use 20% for weight optimization)
    split_idx = int(len(unit_ids) * 0.8)
    val_predictions = {k: v[split_idx:] for k, v in all_predictions.items()}
    val_true = rul_labels.set_index('unit_id').loc[unit_ids[split_idx:], 'RUL'].values
    
    ensemble.optimize_weights(val_predictions, val_true)
    
    # Get ensemble predictions on full test set
    ensemble_predictions = ensemble.predict(all_predictions)
    all_predictions['Ensemble'] = ensemble_predictions
    
    # Create results DataFrame with all predictions
    results_df = pd.DataFrame({
        'unit_id': unit_ids,
        'RUL_true': rul_labels.set_index('unit_id').loc[unit_ids, 'RUL'].values,
        'RUL_pred_LSTM': lstm_mean,  # Use mean from uncertainty quantification
        'RUL_pred_LSTM_lower': lstm_lower,
        'RUL_pred_LSTM_upper': lstm_upper,
        'RUL_pred_RF': rf_predictions,
        'RUL_pred_LR': lr_predictions,
        'RUL_pred_Ensemble': ensemble_predictions
    })
    
    # Evaluate predictions
    logger.info("\n[Step 6/6] Evaluating predictions...")
    evaluator = RULEvaluator()
    
    # Evaluate all models
    y_true = results_df['RUL_true'].values
    
    metrics_all = {}
    for model_name in ['LSTM', 'RF', 'LR', 'Ensemble']:
        pred_col = f'RUL_pred_{model_name}' if model_name != 'Ensemble' else 'RUL_pred_Ensemble'
        if model_name == 'RF':
            pred_col = 'RUL_pred_RF'
        elif model_name == 'LR':
            pred_col = 'RUL_pred_LR'
        
        metrics = evaluator.evaluate_predictions(
            y_true,
            results_df[pred_col].values,
            model_name
        )
        metrics_all[model_name] = metrics
    
    # Check performance targets for ensemble
    logger.info("\nEnsemble Performance:")
    evaluator.check_performance_targets(metrics_all['Ensemble'])
    
    # Show model contributions
    logger.info("\nEnsemble Model Contributions:")
    contributions = ensemble.get_model_contributions(all_predictions)
    logger.info(f"\n{contributions.to_string(index=False)}")
    
    # Evaluate uncertainty calibration
    logger.info("\nEvaluating LSTM Uncertainty Calibration:")
    calibration = uncertainty.evaluate_uncertainty_calibration(
        results_df['RUL_true'].values,
        results_df['RUL_pred_LSTM_lower'].values,
        results_df['RUL_pred_LSTM_upper'].values
    )
    
    # Identify high uncertainty predictions
    interval_df = uncertainty.get_prediction_intervals(
        results_df['RUL_pred_LSTM'].values,
        results_df['RUL_pred_LSTM_lower'].values,
        results_df['RUL_pred_LSTM_upper'].values
    )
    high_unc_engines = uncertainty.get_high_uncertainty_engines(
        interval_df,
        results_df['unit_id'].values,
        top_k=10
    )
    
    # SHAP Explainability
    logger.info("\nGenerating SHAP explanations for LSTM model...")
    
    # Create background dataset (100 random samples from training)
    np.random.seed(config.RANDOM_SEED)
    background_indices = np.random.choice(len(X_test_seq), size=min(100, len(X_test_seq)), replace=False)
    X_background = X_test_seq[background_indices]
    
    # Create SHAP explainer for LSTM
    shap_lstm = SHAPExplainer(models['lstm'], model_type='lstm')
    shap_lstm.create_explainer(X_background, feature_names=models['feature_columns'])
    
    # Explain predictions (limited samples for speed)
    shap_lstm.explain_predictions(X_test_seq, max_samples=100)
    
    # Get feature importance
    lstm_importance = shap_lstm.get_feature_importance(top_k=20)
    
    # Visualizations
    if visualize:
        logger.info("\nCreating visualizations...")
        visualizer = RULVisualizer()
        
        # Prediction scatter plot for ensemble
        visualizer.plot_prediction_scatter(
            results_df['RUL_true'].values,
            results_df['RUL_pred_Ensemble'].values,
            'Ensemble Model (LSTM + RF + LR)',
            f'{dataset_name}_ensemble_prediction_scatter.png'
        )
        
        # Error distribution for ensemble
        visualizer.plot_error_distribution(
            results_df['RUL_true'].values,
            results_df['RUL_pred_Ensemble'].values,
            'Ensemble Model',
            f'{dataset_name}_ensemble_error_distribution.png'
        )
        
        # Comparison plot for LSTM vs Ensemble
        visualizer.plot_prediction_scatter(
            results_df['RUL_true'].values,
            results_df['RUL_pred_LSTM'].values,
            'LSTM Model',
            f'{dataset_name}_lstm_prediction_scatter.png'
        )
        
        # Uncertainty visualization for LSTM
        logger.info("Creating uncertainty visualization...")
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Sort by true RUL for better visualization
        sorted_idx = np.argsort(results_df['RUL_true'].values)
        x = np.arange(len(sorted_idx))
        
        true_rul = results_df['RUL_true'].values[sorted_idx]
        pred_rul = results_df['RUL_pred_LSTM'].values[sorted_idx]
        lower = results_df['RUL_pred_LSTM_lower'].values[sorted_idx]
        upper = results_df['RUL_pred_LSTM_upper'].values[sorted_idx]
        
        # Plot
        ax.plot(x, true_rul, 'g-', label='True RUL', alpha=0.7, lw=2)
        ax.plot(x, pred_rul, 'b-', label='Predicted RUL (mean)', alpha=0.7, lw=2)
        ax.fill_between(x, lower, upper, alpha=0.3, label='95% Confidence Interval')
        
        ax.set_xlabel('Engine Index (sorted by true RUL)', fontsize=12)
        ax.set_ylabel('RUL (cycles)', fontsize=12)
        ax.set_title('LSTM Predictions with Uncertainty Bounds', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(config.PLOTS_DIR, f'{dataset_name}_uncertainty_bounds.png'), 
                   dpi=config.PLOT_CONFIG['dpi'], bbox_inches='tight')
        plt.close()
        logger.info(f"Saved: {dataset_name}_uncertainty_bounds.png")
        
        # SHAP visualizations
        logger.info("Creating SHAP visualizations...")
        
        # SHAP summary plot
        shap_lstm.plot_summary(
            save_path=os.path.join(config.PLOTS_DIR, f'{dataset_name}_shap_summary.png')
        )
        
        # SHAP feature importance bar plot
        shap_lstm.plot_feature_importance_bar(
            top_k=20,
            save_path=os.path.join(config.PLOTS_DIR, f'{dataset_name}_shap_importance.png')
        )
        
        # SHAP waterfall plot for a sample prediction
        shap_lstm.plot_waterfall(
            sample_idx=0,
            save_path=os.path.join(config.PLOTS_DIR, f'{dataset_name}_shap_waterfall_sample0.png')
        )
    
    # Maintenance planning - use ensemble predictions
    logger.info("\nGenerating maintenance recommendations...")
    planner = MaintenancePlanner()
    maintenance_schedule = planner.create_maintenance_schedule(
        results_df[['unit_id', 'RUL_pred_Ensemble']].rename(columns={'RUL_pred_Ensemble': 'RUL_pred'})
    )
    
    # Simulate maintenance strategies
    logger.info("\nSimulating maintenance strategies...")
    
    # Traditional maintenance
    actual_lifetimes = rul_labels['RUL'].values
    trad_results = planner.simulate_traditional_maintenance(actual_lifetimes)
    
    # Predictive maintenance - use ensemble predictions
    pred_results = planner.simulate_predictive_maintenance(
        results_df['RUL_pred_Ensemble'].values,
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
    save_results(metrics_all, os.path.join(config.RESULTS_DIR, f'{dataset_name}_all_metrics.json'))
    save_results(ensemble.weights, os.path.join(config.RESULTS_DIR, f'{dataset_name}_ensemble_weights.json'))
    save_results(calibration, os.path.join(config.RESULTS_DIR, f'{dataset_name}_uncertainty_calibration.json'))
    high_unc_engines.to_csv(os.path.join(config.RESULTS_DIR, f'{dataset_name}_high_uncertainty_engines.csv'), index=False)
    lstm_importance.to_csv(os.path.join(config.RESULTS_DIR, f'{dataset_name}_shap_feature_importance.csv'), index=False)
    
    logger.info("="*80)
    logger.info("PREDICTION COMPLETE!")
    logger.info(f"Results saved to: {config.RESULTS_DIR}")
    if visualize:
        logger.info(f"Plots saved to: {config.PLOTS_DIR}")
    logger.info("="*80)
    
    return results_df, metrics_all, maintenance_schedule, ensemble


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

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


class BatchPredictionManager:
    """
    Manages batch predictions with parallel processing and progress tracking
    Supports resumable jobs and multiple export formats
    """
    
    def __init__(self, 
                 batch_size: int = 100,
                 n_workers: int = 4,
                 output_dir: str = None):
        """
        Initialize batch prediction manager
        
        Args:
            batch_size: Number of samples per batch
            n_workers: Number of parallel workers
            output_dir: Directory for output files
        """
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.output_dir = output_dir or os.path.join(config.RESULTS_DIR, 'batch_predictions')
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.job_status = {}
        self.results_cache = {}
        logger.info(f"Initialized BatchPredictionManager (batch_size={batch_size}, workers={n_workers})")
    
    def create_batches(self, data: pd.DataFrame, id_col: str = 'unit_id') -> list:
        """
        Split data into batches
        
        Args:
            data: Input DataFrame
            id_col: Column for grouping
            
        Returns:
            List of batch DataFrames
        """
        unique_ids = data[id_col].unique()
        batches = []
        
        for i in range(0, len(unique_ids), self.batch_size):
            batch_ids = unique_ids[i:i + self.batch_size]
            batch_df = data[data[id_col].isin(batch_ids)]
            batches.append({
                'batch_idx': len(batches),
                'ids': batch_ids.tolist(),
                'data': batch_df,
                'size': len(batch_ids)
            })
        
        logger.info(f"Created {len(batches)} batches from {len(unique_ids)} units")
        return batches
    
    def process_batch(self,
                      batch: dict,
                      model,
                      feature_cols: list,
                      preprocess_fn=None) -> dict:
        """
        Process a single batch
        
        Args:
            batch: Batch dictionary
            model: Prediction model
            feature_cols: Feature columns
            preprocess_fn: Optional preprocessing function
            
        Returns:
            Batch results
        """
        batch_idx = batch['batch_idx']
        data = batch['data']
        
        try:
            # Preprocess if needed
            if preprocess_fn:
                data = preprocess_fn(data)
            
            # Extract features
            X = data[feature_cols].values
            
            # Predict
            predictions = model.predict(X)
            if hasattr(predictions, 'ravel'):
                predictions = predictions.ravel()
            
            # Build results
            result = {
                'batch_idx': batch_idx,
                'status': 'completed',
                'ids': batch['ids'],
                'predictions': predictions.tolist(),
                'n_samples': len(predictions),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.debug(f"Batch {batch_idx} completed: {len(predictions)} predictions")
            
        except Exception as e:
            result = {
                'batch_idx': batch_idx,
                'status': 'failed',
                'error': str(e),
                'ids': batch['ids'],
                'timestamp': datetime.now().isoformat()
            }
            logger.error(f"Batch {batch_idx} failed: {e}")
        
        return result
    
    def run_batch_prediction(self,
                             data: pd.DataFrame,
                             model,
                             feature_cols: list,
                             job_id: str = None,
                             preprocess_fn=None) -> dict:
        """
        Run batch predictions with progress tracking
        
        Args:
            data: Input DataFrame
            model: Prediction model
            feature_cols: Feature columns
            job_id: Optional job identifier
            preprocess_fn: Optional preprocessing function
            
        Returns:
            Job results
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        job_id = job_id or f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Starting batch prediction job: {job_id}")
        
        # Create batches
        batches = self.create_batches(data)
        
        # Initialize job status
        self.job_status[job_id] = {
            'status': 'running',
            'total_batches': len(batches),
            'completed_batches': 0,
            'failed_batches': 0,
            'start_time': datetime.now().isoformat()
        }
        
        results = []
        
        # Process batches with progress
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {
                executor.submit(self.process_batch, batch, model, feature_cols, preprocess_fn): batch
                for batch in batches
            }
            
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                
                if result['status'] == 'completed':
                    self.job_status[job_id]['completed_batches'] += 1
                else:
                    self.job_status[job_id]['failed_batches'] += 1
                
                # Progress logging
                completed = self.job_status[job_id]['completed_batches']
                total = len(batches)
                logger.info(f"Progress: {completed}/{total} batches ({completed/total*100:.1f}%)")
        
        # Finalize job
        self.job_status[job_id]['status'] = 'completed'
        self.job_status[job_id]['end_time'] = datetime.now().isoformat()
        
        # Cache results
        self.results_cache[job_id] = results
        
        logger.info(f"Job {job_id} completed: {self.job_status[job_id]['completed_batches']} successful, "
                   f"{self.job_status[job_id]['failed_batches']} failed")
        
        return {
            'job_id': job_id,
            'status': self.job_status[job_id],
            'results': results
        }
    
    def export_results(self,
                       job_id: str,
                       format: str = 'csv') -> str:
        """
        Export prediction results
        
        Args:
            job_id: Job identifier
            format: Export format ('csv', 'json', 'parquet')
            
        Returns:
            Path to exported file
        """
        if job_id not in self.results_cache:
            raise ValueError(f"Job {job_id} not found in cache")
        
        results = self.results_cache[job_id]
        
        # Aggregate results
        all_ids = []
        all_predictions = []
        
        for r in results:
            if r['status'] == 'completed':
                all_ids.extend(r['ids'])
                all_predictions.extend(r['predictions'])
        
        df = pd.DataFrame({
            'unit_id': all_ids,
            'RUL_pred': all_predictions
        })
        
        # Export
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format == 'csv':
            filepath = os.path.join(self.output_dir, f'{job_id}_{timestamp}.csv')
            df.to_csv(filepath, index=False)
        elif format == 'json':
            filepath = os.path.join(self.output_dir, f'{job_id}_{timestamp}.json')
            df.to_json(filepath, orient='records', indent=2)
        elif format == 'parquet':
            filepath = os.path.join(self.output_dir, f'{job_id}_{timestamp}.parquet')
            df.to_parquet(filepath, index=False)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        logger.info(f"Exported {len(df)} predictions to {filepath}")
        
        return filepath
    
    def get_job_status(self, job_id: str) -> dict:
        """Get status of a prediction job"""
        return self.job_status.get(job_id, {'status': 'not_found'})
    
    def get_summary_report(self) -> str:
        """Generate summary of all jobs"""
        lines = [
            "=" * 60,
            "BATCH PREDICTION SUMMARY",
            "=" * 60,
            "",
            f"Total jobs: {len(self.job_status)}",
            ""
        ]
        
        for job_id, status in self.job_status.items():
            lines.extend([
                f"Job: {job_id}",
                f"  Status: {status['status']}",
                f"  Batches: {status.get('completed_batches', 0)}/{status.get('total_batches', 0)}",
                ""
            ])
        
        lines.append("=" * 60)
        
        return '\n'.join(lines)


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

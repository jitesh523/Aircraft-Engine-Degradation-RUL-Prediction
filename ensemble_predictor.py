"""
Ensemble Predictor for RUL Prediction
Combines multiple models (LSTM, Random Forest) for improved accuracy
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import os
import config
from utils import setup_logging, load_results

logger = setup_logging(__name__)


class EnsemblePredictor:
    """
    Ensemble predictor combining multiple models
    Uses weighted averaging or stacking strategies
    """
    
    def __init__(self, strategy: str = 'weighted_average'):
        """
        Initialize ensemble predictor
        
        Args:
            strategy: Ensemble strategy ('weighted_average', 'simple_average', 'voting')
        """
        self.strategy = strategy
        self.models = {}
        self.weights = {}
        logger.info(f"Initialized Ensemble Predictor with {strategy} strategy")
    
    def add_model(self, name: str, model: any, weight: float = 1.0):
        """
        Add a model to the ensemble
        
        Args:
            name: Model name
            model: Model object with predict() method
            weight: Weight for weighted averaging (default: 1.0)
        """
        self.models[name] = model
        self.weights[name] = weight
        logger.info(f"Added model '{name}' with weight {weight}")
    
    def set_weights(self, weights: Dict[str, float]):
        """
        Set custom weights for models
        
        Args:
            weights: Dictionary of {model_name: weight}
        """
        for name, weight in weights.items():
            if name in self.models:
                self.weights[name] = weight
        
        # Normalize weights to sum to 1
        total_weight = sum(self.weights.values())
        self.weights = {k: v/total_weight for k, v in self.weights.items()}
        
        logger.info(f"Updated weights: {self.weights}")
    
    def optimize_weights(self, predictions_dict: Dict[str, np.ndarray], y_true: np.ndarray):
        """
        Optimize weights using grid search to minimize RMSE on validation set
        
        Args:
            predictions_dict: Dictionary of {model_name: predictions}
            y_true: True RUL values
        """
        from sklearn.metrics import mean_squared_error
        from itertools import product
        
        logger.info("Optimizing ensemble weights...")
        
        model_names = list(predictions_dict.keys())
        if len(model_names) < 2:
            logger.warning("Need at least 2 models for ensemble")
            return
        
        # Grid search over weight combinations
        best_rmse = float('inf')
        best_weights = None
        
        # For 2 models, search weight space
        if len(model_names) == 2:
            for w1 in np.linspace(0, 1, 21):  # 0.0, 0.05, 0.10, ..., 1.0
                w2 = 1 - w1
                weights = {model_names[0]: w1, model_names[1]: w2}
                
                # Calculate ensemble prediction
                ensemble_pred = sum(predictions_dict[name] * weights[name] 
                                  for name in model_names)
                
                rmse = np.sqrt(mean_squared_error(y_true, ensemble_pred))
                
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_weights = weights.copy()
        
        # For 3+ models, use simple grid search (coarse)
        else:
            weight_options = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
            for weight_combo in product(weight_options, repeat=len(model_names)):
                if abs(sum(weight_combo) - 1.0) > 0.01:  # Skip if doesn't sum to 1
                    continue
                
                weights = dict(zip(model_names, weight_combo))
                
                ensemble_pred = sum(predictions_dict[name] * weights[name] 
                                  for name in model_names)
                
                rmse = np.sqrt(mean_squared_error(y_true, ensemble_pred))
                
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_weights = weights.copy()
        
        if best_weights:
            self.weights = best_weights
            logger.info(f"Optimized weights: {self.weights}")
            logger.info(f"Best validation RMSE: {best_rmse:.2f}")
        else:
            logger.warning("Weight optimization failed, using equal weights")
            self.set_weights({name: 1.0 for name in model_names})
    
    def predict(self, predictions_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Make ensemble prediction
        
        Args:
            predictions_dict: Dictionary of {model_name: predictions array}
            
        Returns:
            Ensemble predictions
        """
        if not predictions_dict:
            raise ValueError("No predictions provided")
        
        if self.strategy == 'weighted_average':
            # Weighted average of predictions
            ensemble_pred = np.zeros_like(list(predictions_dict.values())[0])
            
            for name, predictions in predictions_dict.items():
                weight = self.weights.get(name, 1.0 / len(predictions_dict))
                ensemble_pred += weight * predictions
            
        elif self.strategy == 'simple_average':
            # Simple average
            ensemble_pred = np.mean(list(predictions_dict.values()), axis=0)
        
        elif self.strategy == 'median':
            # Median (robust to outliers)
            ensemble_pred = np.median(list(predictions_dict.values()), axis=0)
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        # Clip negative predictions
        ensemble_pred = np.maximum(ensemble_pred, 0)
        
        return ensemble_pred
    
    def evaluate_ensemble(self, 
                         predictions_dict: Dict[str, np.ndarray],
                         y_true: np.ndarray) -> Dict:
        """
        Evaluate ensemble performance
        
        Args:
            predictions_dict: Dictionary of {model_name: predictions}
            y_true: True RUL values
            
        Returns:
            Dictionary with individual and ensemble metrics
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        logger.info("Evaluating ensemble performance...")
        
        results = {}
        
        # Evaluate individual models
        for name, predictions in predictions_dict.items():
            rmse = np.sqrt(mean_squared_error(y_true, predictions))
            mae = mean_absolute_error(y_true, predictions)
            r2 = r2_score(y_true, predictions)
            
            results[name] = {
                'rmse': float(rmse),
                'mae': float(mae),
                'r2': float(r2)
            }
            
            logger.info(f"{name}: RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.4f}")
        
        # Evaluate ensemble
        ensemble_pred = self.predict(predictions_dict)
        rmse = np.sqrt(mean_squared_error(y_true, ensemble_pred))
        mae = mean_absolute_error(y_true, ensemble_pred)
        r2 = r2_score(y_true, ensemble_pred)
        
        results['ensemble'] = {
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'strategy': self.strategy,
            'weights': self.weights
        }
        
        logger.info(f"Ensemble: RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.4f}")
        
        # Calculate improvement
        best_individual_rmse = min(r['rmse'] for r in results.values() if isinstance(r, dict) and 'rmse' in r and r != results['ensemble'])
        improvement = ((best_individual_rmse - rmse) / best_individual_rmse) * 100
        
        results['improvement'] = {
            'best_individual_rmse': float(best_individual_rmse),
            'ensemble_rmse': float(rmse),
            'improvement_pct': float(improvement)
        }
        
        logger.info(f"Improvement over best individual model: {improvement:.2f}%")
        
        return results
    
    def get_model_contributions(self, predictions_dict: Dict[str, np.ndarray]) -> pd.DataFrame:
        """
        Get contribution of each model to ensemble prediction
        
        Args:
            predictions_dict: Dictionary of predictions
            
        Returns:
            DataFrame with model contributions
        """
        ensemble_pred = self.predict(predictions_dict)
        
        contributions = []
        for name, predictions in predictions_dict.items():
            weight = self.weights.get(name, 1.0 / len(predictions_dict))
            contribution = weight * predictions
            
            contributions.append({
                'model': name,
                'weight': weight,
                'mean_prediction': np.mean(predictions),
                'mean_contribution': np.mean(contribution),
                'contribution_pct': (np.mean(contribution) / np.mean(ensemble_pred)) * 100
            })
        
        return pd.DataFrame(contributions)


if __name__ == "__main__":
    # Test ensemble predictor
    print("="*60)
    print("Testing Ensemble Predictor")
    print("="*60)
    
    # Generate synthetic predictions
    np.random.seed(42)
    n_samples = 100
    y_true = np.random.randint(0, 150, size=n_samples).astype(float)
    
    # Simulate predictions from different models
    lstm_pred = y_true + np.random.randn(n_samples) * 15  # RMSE ~15
    rf_pred = y_true + np.random.randn(n_samples) * 20     # RMSE ~20
    lr_pred = y_true + np.random.randn(n_samples) * 25     # RMSE ~25
    
    # Clip negative values
    lstm_pred = np.maximum(lstm_pred, 0)
    rf_pred = np.maximum(rf_pred, 0)
    lr_pred = np.maximum(lr_pred, 0)
    
    predictions = {
        'LSTM': lstm_pred,
        'Random Forest': rf_pred,
        'Linear Regression': lr_pred
    }
    
    # Test 1: Simple average
    print("\n1. Simple Average Ensemble:")
    ensemble = EnsemblePredictor('simple_average')
    results = ensemble.evaluate_ensemble(predictions, y_true)
    
    # Test 2: Weighted average with optimization
    print("\n2. Optimized Weighted Ensemble:")
    ensemble_opt = EnsemblePredictor('weighted_average')
    ensemble_opt.optimize_weights(predictions, y_true)
    results_opt = ensemble_opt.evaluate_ensemble(predictions, y_true)
    
    # Test 3: Model contributions
    print("\n3. Model Contributions:")
    contributions = ensemble_opt.get_model_contributions(predictions)
    print(contributions.to_string(index=False))
    
    print("\n" + "="*60)
    print("Ensemble predictor test complete!")
    print("="*60)

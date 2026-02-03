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


class AdaptiveEnsembleOptimizer:
    """
    Dynamic ensemble weight optimization
    Adjusts weights based on recent performance and RUL ranges
    """
    
    def __init__(self):
        """Initialize adaptive ensemble optimizer"""
        self.global_weights = {}
        self.range_weights = {}  # Weights optimized per RUL range
        self.performance_history = []
        self.rul_ranges = [
            (0, 30, 'critical'),
            (30, 80, 'warning'),
            (80, 150, 'healthy'),
            (150, float('inf'), 'new')
        ]
        logger.info("Initialized AdaptiveEnsembleOptimizer")
    
    def optimize_per_range(self,
                          predictions_dict: Dict[str, np.ndarray],
                          y_true: np.ndarray) -> Dict:
        """
        Optimize weights for each RUL range separately
        
        Different models may perform better at different RUL ranges
        
        Args:
            predictions_dict: Dictionary of model predictions
            y_true: True RUL values
            
        Returns:
            Dictionary with per-range optimized weights
        """
        from sklearn.metrics import mean_squared_error
        
        logger.info("Optimizing weights per RUL range...")
        
        model_names = list(predictions_dict.keys())
        self.range_weights = {}
        
        for low, high, range_name in self.rul_ranges:
            # Get samples in this range
            mask = (y_true >= low) & (y_true < high)
            n_samples = mask.sum()
            
            if n_samples < 10:
                # Not enough samples, use equal weights
                self.range_weights[range_name] = {name: 1.0/len(model_names) for name in model_names}
                continue
            
            y_range = y_true[mask]
            
            # Evaluate each model in this range
            model_rmses = {}
            for name, preds in predictions_dict.items():
                preds_range = preds[mask]
                rmse = np.sqrt(mean_squared_error(y_range, preds_range))
                model_rmses[name] = rmse
            
            # Calculate inverse RMSE weights (lower RMSE = higher weight)
            total_inv_rmse = sum(1.0 / rmse for rmse in model_rmses.values())
            weights = {name: (1.0 / rmse) / total_inv_rmse for name, rmse in model_rmses.items()}
            
            self.range_weights[range_name] = weights
            
            logger.info(f"  {range_name} range ({low}-{high}): "
                       f"{n_samples} samples, best={min(model_rmses, key=model_rmses.get)}")
        
        return self.range_weights
    
    def predict_adaptive(self,
                        predictions_dict: Dict[str, np.ndarray],
                        reference_rul: np.ndarray = None) -> np.ndarray:
        """
        Make predictions using range-specific weights
        
        Args:
            predictions_dict: Dictionary of model predictions
            reference_rul: Reference RUL values for range selection (optional)
            
        Returns:
            Adaptive ensemble predictions
        """
        if not self.range_weights:
            # Fall back to equal weights
            return np.mean(list(predictions_dict.values()), axis=0)
        
        n_samples = len(list(predictions_dict.values())[0])
        ensemble_pred = np.zeros(n_samples)
        
        # If we have reference RUL, use adaptive per-sample weights
        if reference_rul is not None:
            for i in range(n_samples):
                rul_ref = reference_rul[i]
                
                # Find appropriate range
                range_name = 'new'
                for low, high, name in self.rul_ranges:
                    if low <= rul_ref < high:
                        range_name = name
                        break
                
                weights = self.range_weights.get(range_name, {})
                
                # Calculate weighted prediction
                pred = 0
                for model_name, model_preds in predictions_dict.items():
                    weight = weights.get(model_name, 1.0/len(predictions_dict))
                    pred += weight * model_preds[i]
                
                ensemble_pred[i] = pred
        else:
            # Use global average weights
            avg_weights = self._calculate_average_weights()
            for model_name, model_preds in predictions_dict.items():
                weight = avg_weights.get(model_name, 1.0/len(predictions_dict))
                ensemble_pred += weight * model_preds
        
        return np.maximum(ensemble_pred, 0)
    
    def _calculate_average_weights(self) -> Dict[str, float]:
        """Calculate average weights across all ranges"""
        if not self.range_weights:
            return {}
        
        all_models = set()
        for weights in self.range_weights.values():
            all_models.update(weights.keys())
        
        avg_weights = {}
        for model in all_models:
            weights_list = [w.get(model, 0) for w in self.range_weights.values()]
            avg_weights[model] = np.mean(weights_list)
        
        # Normalize
        total = sum(avg_weights.values())
        if total > 0:
            avg_weights = {k: v/total for k, v in avg_weights.items()}
        
        return avg_weights
    
    def analyze_model_diversity(self,
                                predictions_dict: Dict[str, np.ndarray]) -> Dict:
        """
        Analyze diversity among ensemble models
        
        Diverse models typically produce better ensembles
        
        Args:
            predictions_dict: Dictionary of model predictions
            
        Returns:
            Diversity analysis results
        """
        logger.info("Analyzing model diversity...")
        
        model_names = list(predictions_dict.keys())
        preds_array = np.array(list(predictions_dict.values()))
        
        # Calculate pairwise correlations
        correlations = {}
        for i, name1 in enumerate(model_names):
            for j, name2 in enumerate(model_names):
                if i < j:
                    corr = np.corrcoef(preds_array[i], preds_array[j])[0, 1]
                    correlations[(name1, name2)] = float(corr)
        
        # Average correlation
        avg_correlation = np.mean(list(correlations.values())) if correlations else 1.0
        
        # Prediction variance (higher = more diverse)
        prediction_std = np.mean(np.std(preds_array, axis=0))
        
        # Disagreement rate (how often models differ by >10 cycles)
        disagreements = 0
        total_pairs = 0
        for i in range(len(model_names)):
            for j in range(i+1, len(model_names)):
                diff = np.abs(preds_array[i] - preds_array[j])
                disagreements += (diff > 10).sum()
                total_pairs += len(diff)
        
        disagreement_rate = disagreements / total_pairs if total_pairs > 0 else 0
        
        # Diversity score (0-100)
        diversity_score = (1 - avg_correlation) * 50 + disagreement_rate * 50
        
        result = {
            'pairwise_correlations': correlations,
            'average_correlation': float(avg_correlation),
            'prediction_variance': float(prediction_std),
            'disagreement_rate': float(disagreement_rate),
            'diversity_score': float(diversity_score),
            'diversity_level': 'High' if diversity_score > 40 else 
                              'Medium' if diversity_score > 20 else 'Low'
        }
        
        logger.info(f"Model diversity score: {diversity_score:.1f} ({result['diversity_level']})")
        
        return result
    
    def update_weights_on_feedback(self,
                                   predictions_dict: Dict[str, np.ndarray],
                                   y_true: np.ndarray,
                                   learning_rate: float = 0.1):
        """
        Update weights based on recent prediction performance
        
        Args:
            predictions_dict: New predictions
            y_true: Actual RUL values
            learning_rate: How quickly to adapt (0-1)
        """
        from sklearn.metrics import mean_squared_error
        
        logger.info("Updating weights based on feedback...")
        
        # Calculate current performance
        model_rmses = {}
        for name, preds in predictions_dict.items():
            rmse = np.sqrt(mean_squared_error(y_true, preds))
            model_rmses[name] = rmse
        
        # Calculate new weights based on inverse RMSE
        total_inv_rmse = sum(1.0 / rmse for rmse in model_rmses.values())
        new_weights = {name: (1.0 / rmse) / total_inv_rmse for name, rmse in model_rmses.items()}
        
        # Blend with existing weights
        if self.global_weights:
            for name in new_weights:
                if name in self.global_weights:
                    self.global_weights[name] = (
                        (1 - learning_rate) * self.global_weights[name] +
                        learning_rate * new_weights[name]
                    )
                else:
                    self.global_weights[name] = new_weights[name]
        else:
            self.global_weights = new_weights
        
        # Normalize
        total = sum(self.global_weights.values())
        self.global_weights = {k: v/total for k, v in self.global_weights.items()}
        
        # Record performance
        self.performance_history.append({
            'model_rmses': model_rmses,
            'weights_used': self.global_weights.copy(),
            'n_samples': len(y_true)
        })
        
        logger.info(f"Updated weights: {self.global_weights}")
    
    def get_optimization_report(self) -> Dict:
        """
        Generate report on optimization status
        
        Returns:
            Dictionary with optimization statistics
        """
        return {
            'global_weights': self.global_weights,
            'range_specific_weights': self.range_weights,
            'updates_performed': len(self.performance_history),
            'rul_ranges_configured': [f"{low}-{high} ({name})" 
                                      for low, high, name in self.rul_ranges]
        }


class DynamicEnsembleOptimizer:
    """
    Dynamic ensemble optimization with automatic weight tuning
    and model selection based on recent performance
    """
    
    def __init__(self, 
                 cv_folds: int = 5,
                 optimization_metric: str = 'rmse'):
        """
        Initialize dynamic ensemble optimizer
        
        Args:
            cv_folds: Number of cross-validation folds
            optimization_metric: Metric to optimize ('rmse', 'mae', 'r2')
        """
        self.cv_folds = cv_folds
        self.optimization_metric = optimization_metric
        self.optimized_weights = {}
        self.model_scores = {}
        self.excluded_models = set()
        self.optimization_history = []
        logger.info(f"Initialized DynamicEnsembleOptimizer (cv={cv_folds}, metric={optimization_metric})")
    
    def cross_validate_weights(self,
                               predictions_dict: Dict[str, np.ndarray],
                               y_true: np.ndarray) -> Dict[str, float]:
        """
        Optimize weights using cross-validation
        
        Args:
            predictions_dict: Dictionary of {model_name: predictions}
            y_true: True RUL values
            
        Returns:
            Optimized weights dictionary
        """
        from sklearn.model_selection import KFold
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        logger.info("Optimizing weights via cross-validation...")
        
        model_names = list(predictions_dict.keys())
        n_models = len(model_names)
        
        if n_models < 2:
            return {model_names[0]: 1.0} if n_models == 1 else {}
        
        # Stack predictions
        pred_matrix = np.column_stack([predictions_dict[name] for name in model_names])
        
        # Cross-validate to find best weights
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        best_weights = np.ones(n_models) / n_models
        best_score = float('inf')
        
        # Grid search over weight simplex
        weight_grid = self._generate_weight_grid(n_models, steps=10)
        
        for weights in weight_grid:
            fold_scores = []
            
            for train_idx, val_idx in kf.split(pred_matrix):
                val_preds = pred_matrix[val_idx]
                val_true = y_true[val_idx]
                
                ensemble_pred = np.dot(val_preds, weights)
                
                if self.optimization_metric == 'rmse':
                    score = np.sqrt(mean_squared_error(val_true, ensemble_pred))
                elif self.optimization_metric == 'mae':
                    score = mean_absolute_error(val_true, ensemble_pred)
                else:
                    score = -r2_score(val_true, ensemble_pred)  # Negate for minimization
                
                fold_scores.append(score)
            
            mean_score = np.mean(fold_scores)
            
            if mean_score < best_score:
                best_score = mean_score
                best_weights = weights
        
        # Store results
        self.optimized_weights = dict(zip(model_names, best_weights))
        
        self.optimization_history.append({
            'timestamp': pd.Timestamp.now().isoformat(),
            'weights': self.optimized_weights.copy(),
            'score': best_score,
            'metric': self.optimization_metric
        })
        
        logger.info(f"Optimized weights: {self.optimized_weights}")
        logger.info(f"Best CV score ({self.optimization_metric}): {best_score:.4f}")
        
        return self.optimized_weights
    
    def _generate_weight_grid(self, n_models: int, steps: int = 10) -> List[np.ndarray]:
        """Generate weight combinations that sum to 1"""
        from itertools import product
        
        if n_models == 2:
            weights_list = []
            for w0 in np.linspace(0, 1, steps + 1):
                weights_list.append(np.array([w0, 1 - w0]))
            return weights_list
        
        elif n_models == 3:
            weights_list = []
            for w0 in np.linspace(0, 1, steps + 1):
                for w1 in np.linspace(0, 1 - w0, steps + 1):
                    w2 = 1 - w0 - w1
                    if w2 >= 0:
                        weights_list.append(np.array([w0, w1, w2]))
            return weights_list
        
        else:
            # For more models, use random sampling
            np.random.seed(42)
            weights_list = []
            for _ in range(steps * 20):
                w = np.random.dirichlet(np.ones(n_models))
                weights_list.append(w)
            return weights_list
    
    def evaluate_model_performance(self,
                                   predictions_dict: Dict[str, np.ndarray],
                                   y_true: np.ndarray) -> Dict[str, Dict]:
        """
        Evaluate individual model performance
        
        Args:
            predictions_dict: Dictionary of model predictions
            y_true: True values
            
        Returns:
            Performance scores for each model
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        scores = {}
        
        for name, preds in predictions_dict.items():
            rmse = np.sqrt(mean_squared_error(y_true, preds))
            mae = mean_absolute_error(y_true, preds)
            r2 = r2_score(y_true, preds)
            
            scores[name] = {
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'rank': 0  # Will be set below
            }
        
        # Rank models by RMSE
        sorted_models = sorted(scores.keys(), key=lambda x: scores[x]['rmse'])
        for rank, name in enumerate(sorted_models, 1):
            scores[name]['rank'] = rank
        
        self.model_scores = scores
        
        return scores
    
    def select_best_models(self,
                          predictions_dict: Dict[str, np.ndarray],
                          y_true: np.ndarray,
                          top_k: int = None,
                          min_improvement: float = 0.95) -> Dict[str, np.ndarray]:
        """
        Select best models for ensemble based on performance
        
        Args:
            predictions_dict: All model predictions
            y_true: True values
            top_k: Number of top models to select (None = auto)
            min_improvement: Min relative performance vs best model
            
        Returns:
            Filtered predictions dictionary
        """
        scores = self.evaluate_model_performance(predictions_dict, y_true)
        
        # Sort by RMSE
        sorted_models = sorted(scores.keys(), key=lambda x: scores[x]['rmse'])
        best_rmse = scores[sorted_models[0]]['rmse']
        
        if top_k:
            selected = sorted_models[:top_k]
        else:
            # Auto-select based on performance threshold
            selected = []
            for name in sorted_models:
                if scores[name]['rmse'] <= best_rmse / min_improvement:
                    selected.append(name)
            
            # At least include top 2
            if len(selected) < 2 and len(sorted_models) >= 2:
                selected = sorted_models[:2]
        
        self.excluded_models = set(predictions_dict.keys()) - set(selected)
        
        logger.info(f"Selected {len(selected)} models: {selected}")
        if self.excluded_models:
            logger.info(f"Excluded: {self.excluded_models}")
        
        return {name: predictions_dict[name] for name in selected}
    
    def get_ensemble_prediction(self,
                               predictions_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Get ensemble prediction using optimized weights
        
        Args:
            predictions_dict: Model predictions
            
        Returns:
            Ensemble predictions
        """
        if not self.optimized_weights:
            # Equal weights fallback
            weights = {name: 1.0 / len(predictions_dict) for name in predictions_dict}
        else:
            weights = self.optimized_weights
        
        # Filter to only active models
        active_preds = {k: v for k, v in predictions_dict.items() 
                       if k not in self.excluded_models and k in weights}
        
        if not active_preds:
            active_preds = predictions_dict
            weights = {name: 1.0 / len(active_preds) for name in active_preds}
        
        # Renormalize weights
        total = sum(weights.get(k, 0) for k in active_preds)
        if total == 0:
            total = 1
        
        ensemble_pred = sum(active_preds[name] * weights.get(name, 0) / total 
                           for name in active_preds)
        
        return ensemble_pred
    
    def get_optimization_summary(self) -> str:
        """Generate optimization summary"""
        lines = [
            "=" * 60,
            "ENSEMBLE OPTIMIZATION SUMMARY",
            "=" * 60,
            ""
        ]
        
        if self.optimized_weights:
            lines.append("Optimized Weights:")
            for name, weight in self.optimized_weights.items():
                status = " (excluded)" if name in self.excluded_models else ""
                lines.append(f"  {name}: {weight:.4f}{status}")
        
        if self.model_scores:
            lines.extend(["", "Model Performance:"])
            for name, scores in self.model_scores.items():
                lines.append(f"  {name}: RMSE={scores['rmse']:.2f}, Rank={scores['rank']}")
        
        if self.optimization_history:
            last = self.optimization_history[-1]
            lines.extend([
                "",
                f"Last Optimization: {last['timestamp']}",
                f"Best Score ({last['metric']}): {last['score']:.4f}"
            ])
        
        lines.append("=" * 60)
        
        return '\n'.join(lines)


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

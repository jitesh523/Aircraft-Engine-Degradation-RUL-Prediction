"""
Stacking Ensemble for RUL Prediction
Combines multiple base models with a meta-learner
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import warnings
warnings.filterwarnings('ignore')


class StackingEnsemble:
    """Stacking ensemble combining multiple RUL prediction models"""
    
    def __init__(self,
                 base_models: List[Tuple[str, Any]],
                 meta_learner: Any = None,
                 n_folds: int = 5,
                 use_probas: bool = False,
                 random_state: int = 42):
        """
        Initialize stacking ensemble
        
        Args:
            base_models: List of (name, model) tuples
            meta_learner: Meta-learner model (default: Ridge)
            n_folds: Number of folds for out-of-fold predictions
            use_probas: Whether to use probability predictions (not applicable for regression)
            random_state: Random seed
        """
        self.base_models = base_models
        self.meta_learner = meta_learner if meta_learner else Ridge(alpha=10.0)
        self.n_folds = n_folds
        self.random_state = random_state
        self.meta_features_shape_ = None
        self.base_model_weights_ = None
    
    def fit(self, X, y, verbose: bool = True) -> 'StackingEnsemble':
        """
        Train the stacking ensemble using out-of-fold predictions
        
        Args:
            X: Training features
            y: Training targets
            verbose: Print training progress
            
        Returns:
            self
        """
        if verbose:
            print("Training Stacking Ensemble...")
            print(f"Base models: {[name for name, _ in self.base_models]}")
        
        # Generate out-of-fold predictions for meta-features
        meta_features = self._generate_meta_features(X, y, verbose)
        
        if verbose:
            print(f"\nTraining meta-learner on {meta_features.shape} meta-features...")
        
        # Train meta-learner
        self.meta_learner.fit(meta_features, y)
        
        # Train base models on full dataset
        if verbose:
            print("Retraining base models on full dataset...")
        
        for name, model in self.base_models:
            if verbose:
                print(f"  Training {name}...")
            
            # Check if model needs special handling for eval_set
            if hasattr(model, 'fit'):
                try:
                    # Try with validation set for early stopping
                    n_samples = len(X)
                    n_val = int(0.2 * n_samples)
                    X_train, X_val = X[:-n_val], X[-n_val:]
                    y_train, y_val = y[:-n_val], y[-n_val:]
                    
                    model.fit(X_train, y_train, 
                             eval_set=[(X_val, y_val)],
                             early_stopping_rounds=50,
                             verbose=False)
                except (TypeError, AttributeError):
                    # Fallback to simple fit
                    model.fit(X, y)
        
        self.meta_features_shape_ = meta_features.shape[1]
        
        if verbose:
            print("Stacking Ensemble training complete!")
        
        return self
    
    def _generate_meta_features(self, X, y, verbose: bool = True) -> np.ndarray:
        """
        Generate meta-features using out-of-fold predictions
        
        Args:
            X: Features
            y: Targets
            verbose: Print progress
            
        Returns:
            Meta-features array
        """
        n_models = len(self.base_models)
        n_samples = len(X)
        meta_features = np.zeros((n_samples, n_models))
        
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        for model_idx, (name, model) in enumerate(self.base_models):
            if verbose:
                print(f"Generating out-of-fold predictions for {name}...")
            
            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
                if verbose and fold_idx % 2 == 0:
                    print(f"  Fold {fold_idx + 1}/{self.n_folds}")
                
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]
                
                # Clone model to avoid modifying the original
                from copy import deepcopy
                fold_model = deepcopy(model)
                
                # Train on fold
                try:
                    fold_model.fit(X_train_fold, y_train_fold,
                                  eval_set=[(X_val_fold, y_val_fold)],
                                  early_stopping_rounds=30,
                                  verbose=False)
                except (TypeError, AttributeError):
                    fold_model.fit(X_train_fold, y_train_fold)
                
                # Predict on validation fold
                meta_features[val_idx, model_idx] = fold_model.predict(X_val_fold)
        
        return meta_features
    
    def predict(self, X) -> np.ndarray:
        """
        Make predictions using the stacking ensemble
        
        Args:
            X: Features
            
        Returns:
            Predictions
        """
        # Generate meta-features from base models
        meta_features = np.zeros((len(X), len(self.base_models)))
        
        for model_idx, (name, model) in enumerate(self.base_models):
            meta_features[:, model_idx] = model.predict(X)
        
        # Use meta-learner for final prediction
        predictions = self.meta_learner.predict(meta_features)
        
        # Ensure non-negative RUL
        return np.maximum(predictions, 0)
    
    def predict_with_base_models(self, X) -> Dict[str, np.ndarray]:
        """
        Get predictions from all models including base models
        
        Args:
            X: Features
            
        Returns:
            Dictionary of predictions from each model
        """
        predictions = {}
        
        # Base model predictions
        for name, model in self.base_models:
            predictions[name] = model.predict(X)
        
        # Ensemble prediction
        predictions['stacking_ensemble'] = self.predict(X)
        
        return predictions
    
    def get_base_model_scores(self, X, y) -> pd.DataFrame:
        """
        Evaluate individual base models and the ensemble
        
        Args:
            X: Test features
            y: Test targets
            
        Returns:
            DataFrame with scores for each model
        """
        scores = []
        
        # Evaluate base models
        for name, model in self.base_models:
            preds = model.predict(X)
            scores.append({
                'model': name,
                'rmse': np.sqrt(mean_squared_error(y, preds)),
                'mae': mean_absolute_error(y, preds),
                'r2': r2_score(y, preds)
            })
        
        # Evaluate ensemble
        ensemble_preds = self.predict(X)
        scores.append({
            'model': 'stacking_ensemble',
            'rmse': np.sqrt(mean_squared_error(y, ensemble_preds)),
            'mae': mean_absolute_error(y, ensemble_preds),
            'r2': r2_score(y, ensemble_preds)
        })
        
        return pd.DataFrame(scores).sort_values('rmse')
    
    def analyze_model_contributions(self, X, y) -> pd.DataFrame:
        """
        Analyze how much each base model contributes to the ensemble
        
        Args:
            X: Features
            y: Targets
            
        Returns:
            DataFrame with contribution analysis
        """
        # Get meta-features
        meta_features = np.zeros((len(X), len(self.base_models)))
        for model_idx, (name, model) in enumerate(self.base_models):
            meta_features[:, model_idx] = model.predict(X)
        
        # Get meta-learner coefficients if available
        contributions = []
        if hasattr(self.meta_learner, 'coef_'):
            coefficients = self.meta_learner.coef_
            for idx, (name, _) in enumerate(self.base_models):
                contributions.append({
                    'model': name,
                    'weight': coefficients[idx],
                    'abs_weight': abs(coefficients[idx])
                })
        else:
            # For models without coefficients, use correlation as proxy
            ensemble_preds = self.predict(X)
            for idx, (name, _) in enumerate(self.base_models):
                corr = np.corrcoef(meta_features[:, idx], ensemble_preds)[0, 1]
                contributions.append({
                    'model': name,
                    'correlation': corr,
                    'abs_correlation': abs(corr)
                })
        
        return pd.DataFrame(contributions)
    
    def save(self, filepath: str):
        """Save ensemble to file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(filepath: str) -> 'StackingEnsemble':
        """Load ensemble from file"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


class WeightedAverageEnsemble:
    """Simple weighted average ensemble as alternative to stacking"""
    
    def __init__(self,
                 models: List[Tuple[str, Any]],
                 weights: Optional[List[float]] = None):
        """
        Initialize weighted average ensemble
        
        Args:
            models: List of (name, model) tuples
            weights: Optional weights for each model (default: equal weights)
        """
        self.models = models
        if weights is None:
            self.weights = np.ones(len(models)) / len(models)
        else:
            self.weights = np.array(weights)
            self.weights = self.weights / self.weights.sum()  # Normalize
    
    def fit(self, X, y):
        """Train all models"""
        for name, model in self.models:
            print(f"Training {name}...")
            try:
                model.fit(X, y,
                         eval_set=[(X[-int(0.2*len(X)):], y[-int(0.2*len(X)):])],
                         early_stopping_rounds=50,
                         verbose=False)
            except (TypeError, AttributeError):
                model.fit(X, y)
        return self
    
    def predict(self, X) -> np.ndarray:
        """Make weighted average predictions"""
        predictions = np.zeros(len(X))
        
        for (name, model), weight in zip(self.models, self.weights):
            predictions += weight * model.predict(X)
        
        return np.maximum(predictions, 0)
    
    def optimize_weights(self, X_val, y_val) -> 'WeightedAverageEnsemble':
        """
        Optimize weights using validation set
        
        Args:
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            self
        """
        from scipy.optimize import minimize
        
        def objective(weights):
            weights = weights / weights.sum()
            predictions = np.zeros(len(y_val))
            for (name, model), weight in zip(self.models, weights):
                predictions += weight * model.predict(X_val)
            return mean_squared_error(y_val, predictions)
        
        # Initial weights (equal)
        x0 = np.ones(len(self.models))
        
        # Constraints: weights must be non-negative and sum to 1
        constraints = {'type': 'eq', 'fun': lambda w: w.sum() - 1}
        bounds = [(0, 1) for _ in range(len(self.models))]
        
        result = minimize(objective, x0, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        self.weights = result.x
        print(f"Optimized weights: {dict(zip([name for name, _ in self.models], self.weights))}")
        
        return self
    
    def save(self, filepath: str):
        """Save ensemble to file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(filepath: str) -> 'WeightedAverageEnsemble':
        """Load ensemble from file"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)

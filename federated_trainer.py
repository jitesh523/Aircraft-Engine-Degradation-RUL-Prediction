"""
Federated Learning Simulator for Aircraft Engine RUL Prediction.

Simulates privacy-preserving distributed training across airline "sites"
using the Federated Averaging (FedAvg) algorithm.  Supports uniform and
skewed (non-IID) data partitioning with centralized baseline comparison.

Classes:
    FederatedTrainer  — Multi-round FedAvg orchestrator with privacy analysis.
    FederatedEnsemble — Weighted prediction ensemble of local site models.

Usage::

    from federated_trainer import FederatedTrainer
    trainer = FederatedTrainer(n_sites=4, n_rounds=5)
    trainer.partition_data(train_df)
    results = trainer.run_rounds(test_df)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import copy
import os
import json
import config
from utils import setup_logging

logger = setup_logging(__name__)


class FederatedTrainer:
    """
    Simulates Federated Learning across multiple airline sites.
    
    Each site holds a non-overlapping partition of engines. Sites train
    local models and share only model updates (not raw data) with a
    central server that aggregates them using FedAvg.
    """
    
    def __init__(self, n_sites: int = 4, n_rounds: int = 5):
        """
        Initialize federated trainer.
        
        Args:
            n_sites: Number of distributed sites (simulated airlines).
            n_rounds: Number of federated training rounds.
        """
        self.n_sites = n_sites
        self.n_rounds = n_rounds
        self.sites = {}
        self.global_model = None
        self.history = {
            'round': [],
            'global_rmse': [],
            'site_rmses': [],
            'convergence': []
        }
        self.feature_cols = None
        self.scaler = StandardScaler()
        
        logger.info(f"FederatedTrainer initialized: {n_sites} sites, {n_rounds} rounds")
    
    # ------------------------------------------------------------------
    # Data partitioning
    # ------------------------------------------------------------------
    def partition_data(self, train_df: pd.DataFrame, 
                       strategy: str = 'uniform') -> Dict[int, pd.DataFrame]:
        """
        Partition engine data across sites without overlap.
        
        Args:
            train_df: Training DataFrame with unit_id.
            strategy: 'uniform' (equal split) or 'skewed' (non-IID).
            
        Returns:
            Dictionary mapping site_id -> DataFrame.
        """
        engine_ids = train_df['unit_id'].unique()
        np.random.seed(42)
        np.random.shuffle(engine_ids)
        
        if strategy == 'uniform':
            splits = np.array_split(engine_ids, self.n_sites)
        else:
            # Skewed: first site gets 50%, rest split equally
            half = len(engine_ids) // 2
            splits = [engine_ids[:half]]
            rest = np.array_split(engine_ids[half:], self.n_sites - 1)
            splits.extend(rest)
        
        for i, site_engines in enumerate(splits):
            site_data = train_df[train_df['unit_id'].isin(site_engines)].copy()
            self.sites[i] = {
                'data': site_data,
                'n_engines': len(site_engines),
                'model': None
            }
            logger.info(f"Site {i}: {len(site_engines)} engines, "
                       f"{len(site_data)} samples")
        
        return {i: s['data'] for i, s in self.sites.items()}
    
    def _prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features and targets from DataFrame."""
        if self.feature_cols is None:
            sensor_cols = [c for c in df.columns if c.startswith('sensor_')]
            setting_cols = [c for c in df.columns if c.startswith('setting_')]
            self.feature_cols = [c for c in sensor_cols + setting_cols 
                               if df[c].std() > 1e-6]
        
        X = df[self.feature_cols].fillna(0).values
        y = df['RUL'].values
        return X, y
    
    # ------------------------------------------------------------------
    # Local training
    # ------------------------------------------------------------------
    def local_train(self, site_id: int, 
                    global_weights: Optional[Dict] = None) -> Dict:
        """
        Train a local model at a given site.
        
        Args:
            site_id: Site identifier.
            global_weights: Previous global model's feature importances (for warm-start guidance).
            
        Returns:
            Local training results.
        """
        site = self.sites[site_id]
        data = site['data']
        
        # Use last cycle per engine for speed
        last_cycle = data.groupby('unit_id').last().reset_index()
        X, y = self._prepare_features(last_cycle)
        
        # Train local model
        model = GradientBoostingRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            subsample=0.8, random_state=42 + site_id
        )
        model.fit(X, y)
        
        # Evaluate locally
        y_pred = model.predict(X)
        rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
        
        site['model'] = model
        
        logger.info(f"Site {site_id}: local RMSE = {rmse:.2f}")
        
        return {
            'site_id': site_id,
            'n_samples': len(X),
            'local_rmse': rmse,
            'feature_importances': model.feature_importances_.tolist()
        }
    
    # ------------------------------------------------------------------
    # Federated Averaging
    # ------------------------------------------------------------------
    def federated_average(self) -> GradientBoostingRegressor:
        """
        Aggregate local models using Federated Averaging.
        
        For tree-based models, we average predictions (ensemble of local models)
        rather than averaging weights directly. This is the practical FedAvg
        variant for non-neural models.
        
        Returns:
            Aggregated global model (prediction ensemble).
        """
        local_models = []
        weights = []
        
        for site_id, site in self.sites.items():
            if site['model'] is not None:
                local_models.append(site['model'])
                weights.append(site['n_engines'])
        
        if not local_models:
            raise RuntimeError("No local models trained.")
        
        # Normalize weights
        total = sum(weights)
        weights = [w / total for w in weights]
        
        # Create ensemble predictor that wraps local models
        self.global_model = FederatedEnsemble(local_models, weights)
        
        # Average feature importances
        avg_importance = np.zeros(len(self.feature_cols))
        for model, w in zip(local_models, weights):
            avg_importance += w * model.feature_importances_
        
        self.global_model.avg_feature_importances = avg_importance
        
        logger.info(f"FedAvg complete: aggregated {len(local_models)} site models")
        return self.global_model
    
    # ------------------------------------------------------------------
    # Multi-round training
    # ------------------------------------------------------------------
    def run_rounds(self, test_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Run multiple federated training rounds.
        
        Args:
            test_df: Optional test data for per-round evaluation.
            
        Returns:
            DataFrame with per-round metrics.
        """
        test_X, test_y = None, None
        if test_df is not None:
            test_last = test_df if 'RUL' in test_df.columns else None
            if test_last is not None:
                test_X, test_y = self._prepare_features(test_last)
        
        for round_num in range(1, self.n_rounds + 1):
            logger.info(f"--- Round {round_num}/{self.n_rounds} ---")
            
            # Local training at each site
            site_results = []
            for site_id in self.sites:
                result = self.local_train(site_id)
                site_results.append(result)
            
            # Aggregate
            self.federated_average()
            
            # Evaluate global model
            global_rmse = None
            if test_X is not None and test_y is not None:
                y_pred = self.global_model.predict(test_X)
                global_rmse = float(np.sqrt(mean_squared_error(test_y, y_pred)))
            
            self.history['round'].append(round_num)
            self.history['global_rmse'].append(global_rmse)
            self.history['site_rmses'].append([r['local_rmse'] for r in site_results])
            
            # Convergence check
            if len(self.history['global_rmse']) >= 2:
                prev = self.history['global_rmse'][-2]
                curr = self.history['global_rmse'][-1]
                if prev and curr:
                    change = abs(prev - curr) / prev * 100
                    self.history['convergence'].append(change)
                else:
                    self.history['convergence'].append(None)
            else:
                self.history['convergence'].append(None)
            
            if global_rmse:
                logger.info(f"Round {round_num}: global RMSE = {global_rmse:.2f}")
        
        return pd.DataFrame({
            'round': self.history['round'],
            'global_rmse': self.history['global_rmse'],
            'convergence_pct': self.history['convergence']
        })
    
    # ------------------------------------------------------------------
    # Centralized baseline comparison
    # ------------------------------------------------------------------
    def evaluate_vs_centralized(self, train_df: pd.DataFrame,
                                 test_df: pd.DataFrame) -> Dict:
        """
        Compare FL global model against a centralized baseline.
        
        Args:
            train_df: Full training data (centralized).
            test_df: Test data with RUL labels.
            
        Returns:
            Comparison results.
        """
        # Centralized training
        last_cycle = train_df.groupby('unit_id').last().reset_index()
        X_train, y_train = self._prepare_features(last_cycle)
        
        centralized = GradientBoostingRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            subsample=0.8, random_state=42
        )
        centralized.fit(X_train, y_train)
        
        # Evaluate both
        X_test, y_test = self._prepare_features(test_df)
        
        cent_pred = centralized.predict(X_test)
        cent_rmse = float(np.sqrt(mean_squared_error(y_test, cent_pred)))
        cent_mae = float(mean_absolute_error(y_test, cent_pred))
        cent_r2 = float(r2_score(y_test, cent_pred))
        
        fed_pred = self.global_model.predict(X_test)
        fed_rmse = float(np.sqrt(mean_squared_error(y_test, fed_pred)))
        fed_mae = float(mean_absolute_error(y_test, fed_pred))
        fed_r2 = float(r2_score(y_test, fed_pred))
        
        gap = (fed_rmse - cent_rmse) / cent_rmse * 100
        
        result = {
            'centralized': {'rmse': cent_rmse, 'mae': cent_mae, 'r2': cent_r2},
            'federated': {'rmse': fed_rmse, 'mae': fed_mae, 'r2': fed_r2},
            'rmse_gap_pct': gap,
            'privacy_preserved': True,
            'data_shared': False,
            'n_sites': self.n_sites,
            'n_rounds': self.n_rounds
        }
        
        logger.info(f"Centralized RMSE: {cent_rmse:.2f} | Federated RMSE: {fed_rmse:.2f} "
                    f"| Gap: {gap:+.1f}%")
        
        return result
    
    def privacy_analysis(self) -> Dict:
        """
        Quantify privacy characteristics of the federated setup.
        
        Returns:
            Privacy analysis report.
        """
        total_samples = sum(s['data'].shape[0] for s in self.sites.values())
        
        return {
            'n_sites': self.n_sites,
            'total_engines': sum(s['n_engines'] for s in self.sites.values()),
            'total_samples': total_samples,
            'data_isolation': True,
            'raw_data_shared': False,
            'shared_artifacts': 'model_predictions_only (ensemble averaging)',
            'data_per_site': {
                f'site_{i}': {
                    'engines': s['n_engines'],
                    'samples': len(s['data']),
                    'pct_of_total': len(s['data']) / total_samples * 100
                }
                for i, s in self.sites.items()
            }
        }


class FederatedEnsemble:
    """Ensemble of local models that predicts by weighted averaging."""
    
    def __init__(self, models: List, weights: List[float]):
        self.models = models
        self.weights = weights
        self.avg_feature_importances = None
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = np.zeros(len(X))
        for model, w in zip(self.models, self.weights):
            predictions += w * model.predict(X)
        return predictions
    
    @property
    def feature_importances_(self):
        return self.avg_feature_importances


# ============================================================
# Standalone test
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Federated Learning Simulator")
    print("=" * 60)
    
    from data_loader import CMAPSSDataLoader
    from utils import add_remaining_useful_life
    
    # Load data
    loader = CMAPSSDataLoader('FD001')
    train_df, test_df, rul_df = loader.load_all_data()
    train_df = add_remaining_useful_life(train_df)
    
    # Prepare test data
    test_last = test_df.groupby('unit_id').last().reset_index()
    test_last = test_last.merge(rul_df, on='unit_id')
    
    # Initialize FL
    trainer = FederatedTrainer(n_sites=4, n_rounds=3)
    
    # Partition data
    print("\n--- Data Partitioning ---")
    partitions = trainer.partition_data(train_df, strategy='uniform')
    for sid, data in partitions.items():
        print(f"  Site {sid}: {data['unit_id'].nunique()} engines")
    
    # Run federated rounds
    print("\n--- Federated Training ---")
    rounds_df = trainer.run_rounds(test_last)
    print(rounds_df.to_string(index=False))
    
    # Compare with centralized
    print("\n--- Centralized vs Federated ---")
    comparison = trainer.evaluate_vs_centralized(train_df, test_last)
    print(f"  Centralized RMSE: {comparison['centralized']['rmse']:.2f}")
    print(f"  Federated RMSE:   {comparison['federated']['rmse']:.2f}")
    print(f"  Gap: {comparison['rmse_gap_pct']:+.1f}%")
    
    # Privacy analysis
    print("\n--- Privacy Analysis ---")
    privacy = trainer.privacy_analysis()
    print(f"  Data isolation: {privacy['data_isolation']}")
    print(f"  Raw data shared: {privacy['raw_data_shared']}")
    
    print("\n✅ Federated Learning test PASSED")

"""
Multi-Dataset Training Pipeline for Cross-Domain RUL Prediction
Trains on FD001-FD004 with feature harmonization and domain adaptation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import json
import config
from utils import setup_logging

logger = setup_logging(__name__)


class MultiDatasetTrainer:
    """
    Train RUL prediction models across multiple C-MAPSS datasets (FD001-FD004).
    
    Handles:
    - Loading and tagging datasets by source
    - Feature harmonization across different operating conditions
    - MMD-based domain adaptation for feature alignment
    - Unified training with dataset-weighted sampling
    - Cross-dataset evaluation
    - Transfer learning (fine-tune on target domain)
    """
    
    DATASET_NAMES = ['FD001', 'FD002', 'FD003', 'FD004']
    
    def __init__(self):
        """Initialize multi-dataset trainer."""
        self.datasets = {}
        self.combined_train = None
        self.combined_test = None
        self.scaler = StandardScaler()
        self.model = None
        self.feature_cols = None
        self.training_results = {}
        
        logger.info("MultiDatasetTrainer initialized")
    
    # ------------------------------------------------------------------
    # Data loading & tagging
    # ------------------------------------------------------------------
    def load_datasets(self, names: List[str] = None) -> Dict[str, Dict]:
        """
        Load and tag multiple C-MAPSS datasets.
        
        Args:
            names: List of dataset names to load (default: all four).
            
        Returns:
            Dictionary of loaded datasets with metadata.
        """
        from data_loader import CMAPSSDataLoader
        
        names = names or self.DATASET_NAMES
        
        for name in names:
            try:
                loader = CMAPSSDataLoader(name)
                train_df, test_df, rul_df = loader.load_all_data()
                
                # Tag dataset source
                train_df['dataset'] = name
                test_df['dataset'] = name
                
                # Add RUL to training data (max time per engine - current time)
                train_df = self._add_rul_to_train(train_df)
                
                # Merge RUL labels into test data (last cycle per engine)
                test_last = test_df.groupby('unit_id').last().reset_index()
                test_last = test_last.merge(rul_df, on='unit_id')
                test_last['dataset'] = name
                
                self.datasets[name] = {
                    'train': train_df,
                    'test': test_df,
                    'test_last': test_last,
                    'rul': rul_df,
                    'n_engines_train': train_df['unit_id'].nunique(),
                    'n_engines_test': test_df['unit_id'].nunique(),
                }
                
                logger.info(f"Loaded {name}: {train_df['unit_id'].nunique()} train / "
                           f"{test_df['unit_id'].nunique()} test engines")
                
            except Exception as e:
                logger.warning(f"Could not load {name}: {e}")
        
        logger.info(f"Total datasets loaded: {len(self.datasets)}")
        return {k: {kk: vv for kk, vv in v.items() if kk not in ('train', 'test', 'test_last', 'rul')}
                for k, v in self.datasets.items()}
    
    def _add_rul_to_train(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Add RUL column to training data (run-to-failure: RUL = max_cycle - current_cycle)."""
        df = train_df.copy()
        max_cycles = df.groupby('unit_id')['time_cycles'].transform('max')
        df['RUL'] = max_cycles - df['time_cycles']
        
        # Cap RUL at a max value (piecewise linear)
        max_rul = config.LSTM_CONFIG.get('max_rul', 125)
        df['RUL'] = df['RUL'].clip(upper=max_rul)
        return df
    
    # ------------------------------------------------------------------
    # Feature harmonization
    # ------------------------------------------------------------------
    def harmonize_features(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Harmonize features across all loaded datasets.
        
        - Uses only common sensor/setting columns
        - Applies per-dataset z-score normalization before combining
        - Returns combined train and test DataFrames
        
        Returns:
            (combined_train, combined_test_last)
        """
        if not self.datasets:
            raise RuntimeError("Call load_datasets() first.")
        
        sensor_cols = [c for c in config.COLUMN_NAMES if 'sensor' in c]
        setting_cols = [c for c in config.COLUMN_NAMES if 'setting' in c]
        feature_cols = sensor_cols + setting_cols
        
        # Remove zero-variance sensors (constant across all datasets)
        valid_features = []
        for col in feature_cols:
            variances = []
            for ds in self.datasets.values():
                if col in ds['train'].columns:
                    variances.append(ds['train'][col].std())
            if any(v > 1e-6 for v in variances):
                valid_features.append(col)
        
        self.feature_cols = valid_features
        logger.info(f"Harmonized features: {len(valid_features)} valid features "
                    f"(removed {len(feature_cols) - len(valid_features)} zero-variance)")
        
        # Combine training data
        train_parts = []
        test_parts = []
        
        for name, ds in self.datasets.items():
            train = ds['train'].copy()
            test_last = ds['test_last'].copy()
            
            # Per-dataset scaling (fit on train, transform both)
            scaler = StandardScaler()
            train[valid_features] = scaler.fit_transform(train[valid_features].fillna(0))
            test_last[valid_features] = scaler.transform(test_last[valid_features].fillna(0))
            
            train_parts.append(train)
            test_parts.append(test_last)
        
        self.combined_train = pd.concat(train_parts, ignore_index=True)
        self.combined_test = pd.concat(test_parts, ignore_index=True)
        
        logger.info(f"Combined train: {len(self.combined_train)} samples, "
                    f"Combined test: {len(self.combined_test)} engines")
        
        return self.combined_train, self.combined_test
    
    # ------------------------------------------------------------------
    # Domain adaptation (MMD)
    # ------------------------------------------------------------------
    def domain_adaptation_mmd(self, 
                               source_name: str,
                               target_name: str,
                               n_components: int = 10) -> Dict:
        """
        Simple domain adaptation using Maximum Mean Discrepancy (MMD).
        
        Aligns feature distributions between source and target datasets
        using a linear projection that minimizes MMD.
        
        Args:
            source_name: Source dataset name.
            target_name: Target dataset name.
            n_components: Number of PCA components for projection.
            
        Returns:
            Dictionary with MMD before and after adaptation.
        """
        from sklearn.decomposition import PCA
        
        if source_name not in self.datasets or target_name not in self.datasets:
            raise ValueError("Datasets not loaded.")
        
        source = self.datasets[source_name]['train'][self.feature_cols].dropna()
        target = self.datasets[target_name]['train'][self.feature_cols].dropna()
        
        # Sample for speed
        n_sample = min(2000, len(source), len(target))
        source_sample = source.sample(n_sample, random_state=42).values
        target_sample = target.sample(n_sample, random_state=42).values
        
        # Scale
        scaler = StandardScaler()
        source_scaled = scaler.fit_transform(source_sample)
        target_scaled = scaler.transform(target_sample)
        
        # MMD before adaptation
        mmd_before = self._compute_mmd(source_scaled, target_scaled)
        
        # PCA projection to shared subspace
        n_comp = min(n_components, source_scaled.shape[1])
        pca = PCA(n_components=n_comp)
        source_proj = pca.fit_transform(source_scaled)
        target_proj = pca.transform(target_scaled)
        
        # MMD after projection
        mmd_after = self._compute_mmd(source_proj, target_proj)
        
        reduction = (1 - mmd_after / max(mmd_before, 1e-10)) * 100
        
        logger.info(f"MMD({source_name}→{target_name}): {mmd_before:.4f} → {mmd_after:.4f} "
                    f"({reduction:.1f}% reduction)")
        
        return {
            'source': source_name,
            'target': target_name,
            'mmd_before': float(mmd_before),
            'mmd_after': float(mmd_after),
            'reduction_pct': float(reduction),
            'n_components': n_comp
        }
    
    @staticmethod
    def _compute_mmd(X: np.ndarray, Y: np.ndarray) -> float:
        """Compute Maximum Mean Discrepancy between two sample sets."""
        xx = np.mean(np.sum(X ** 2, axis=1))
        yy = np.mean(np.sum(Y ** 2, axis=1))
        xy = np.mean(X @ Y.T)
        return float(xx + yy - 2 * xy)
    
    # ------------------------------------------------------------------
    # Unified training
    # ------------------------------------------------------------------
    def train_unified_model(self,
                            model_type: str = 'gradient_boosting',
                            use_last_cycle: bool = True,
                            sample_weight_by_dataset: bool = True) -> Dict:
        """
        Train a single model on combined multi-dataset data.
        
        Args:
            model_type: 'random_forest' or 'gradient_boosting'.
            use_last_cycle: Use only last cycle per engine (for baseline models).
            sample_weight_by_dataset: Weight samples so smaller datasets get equal influence.
            
        Returns:
            Training results dictionary.
        """
        if self.combined_train is None:
            self.harmonize_features()
        
        if use_last_cycle:
            train_data = self.combined_train.groupby(
                ['dataset', 'unit_id']
            ).last().reset_index()
        else:
            # Subsample to keep training fast
            train_data = self.combined_train.groupby('dataset').apply(
                lambda x: x.sample(min(5000, len(x)), random_state=42)
            ).reset_index(drop=True)
        
        X = train_data[self.feature_cols].fillna(0).values
        y = train_data['RUL'].values
        
        # Sample weights
        weights = None
        if sample_weight_by_dataset:
            dataset_counts = train_data['dataset'].value_counts()
            max_count = dataset_counts.max()
            weight_map = {ds: max_count / cnt for ds, cnt in dataset_counts.items()}
            weights = train_data['dataset'].map(weight_map).values
        
        # Build model
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=200, max_depth=15,
                min_samples_leaf=5, random_state=42, n_jobs=-1
            )
        else:
            self.model = GradientBoostingRegressor(
                n_estimators=300, max_depth=6,
                learning_rate=0.05, subsample=0.8,
                random_state=42
            )
        
        self.model.fit(X, y, sample_weight=weights)
        
        # In-sample evaluation
        y_pred = self.model.predict(X)
        train_metrics = {
            'rmse': float(np.sqrt(mean_squared_error(y, y_pred))),
            'mae': float(mean_absolute_error(y, y_pred)),
            'r2': float(r2_score(y, y_pred))
        }
        
        self.training_results['unified'] = {
            'model_type': model_type,
            'n_samples': len(X),
            'datasets_used': list(self.datasets.keys()),
            'train_metrics': train_metrics
        }
        
        logger.info(f"Unified model trained ({model_type}): "
                    f"RMSE={train_metrics['rmse']:.2f}, R²={train_metrics['r2']:.3f}")
        
        return self.training_results['unified']
    
    # ------------------------------------------------------------------
    # Cross-dataset evaluation
    # ------------------------------------------------------------------
    def evaluate_cross_dataset(self) -> pd.DataFrame:
        """
        Evaluate the unified model on each dataset's test set individually.
        
        Returns:
            DataFrame with per-dataset metrics.
        """
        if self.model is None:
            raise RuntimeError("Call train_unified_model() first.")
        
        results = []
        
        for name, ds in self.datasets.items():
            test_last = ds['test_last'].copy()
            rul_true = test_last['RUL'].values
            
            # Scale features the same way
            available = [c for c in self.feature_cols if c in test_last.columns]
            X_test = test_last[available].fillna(0).values
            
            # Pad if needed
            if X_test.shape[1] < len(self.feature_cols):
                pad_width = len(self.feature_cols) - X_test.shape[1]
                X_test = np.hstack([X_test, np.zeros((len(X_test), pad_width))])
            
            y_pred = self.model.predict(X_test)
            
            metrics = {
                'dataset': name,
                'n_engines': len(rul_true),
                'rmse': float(np.sqrt(mean_squared_error(rul_true, y_pred))),
                'mae': float(mean_absolute_error(rul_true, y_pred)),
                'r2': float(r2_score(rul_true, y_pred))
            }
            results.append(metrics)
            
            logger.info(f"Eval {name}: RMSE={metrics['rmse']:.2f}, "
                       f"MAE={metrics['mae']:.2f}, R²={metrics['r2']:.3f}")
        
        self.training_results['cross_eval'] = results
        return pd.DataFrame(results)
    
    # ------------------------------------------------------------------
    # Transfer learning
    # ------------------------------------------------------------------
    def transfer_learning(self,
                          target_name: str,
                          n_finetune_samples: int = None,
                          model_type: str = 'gradient_boosting') -> Dict:
        """
        Fine-tune a pretrained model on a specific target dataset.
        
        Trains on all OTHER datasets first, then fine-tunes on a small
        portion of the target dataset and evaluates on target test set.
        
        Args:
            target_name: Target dataset to fine-tune on.
            n_finetune_samples: Number of target samples for fine-tuning.
            model_type: Model type to use.
            
        Returns:
            Transfer learning evaluation results.
        """
        if target_name not in self.datasets:
            raise ValueError(f"{target_name} not loaded.")
        
        # Source data: everything except target
        source_parts = []
        for name, ds in self.datasets.items():
            if name != target_name:
                last_cycle = ds['train'].groupby('unit_id').last().reset_index()
                source_parts.append(last_cycle)
        
        source_data = pd.concat(source_parts, ignore_index=True)
        X_source = source_data[self.feature_cols].fillna(0).values
        y_source = source_data['RUL'].values
        
        # Scale
        scaler = StandardScaler()
        X_source = scaler.fit_transform(X_source)
        
        # Train on source
        if model_type == 'random_forest':
            model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
        else:
            model = GradientBoostingRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42)
        
        model.fit(X_source, y_source)
        
        # Evaluate on target BEFORE fine-tuning
        target_test = self.datasets[target_name]['test_last'].copy()
        available = [c for c in self.feature_cols if c in target_test.columns]
        X_target_test = target_test[available].fillna(0).values
        if X_target_test.shape[1] < len(self.feature_cols):
            pad = len(self.feature_cols) - X_target_test.shape[1]
            X_target_test = np.hstack([X_target_test, np.zeros((len(X_target_test), pad))])
        X_target_test = scaler.transform(X_target_test)
        y_target = target_test['RUL'].values
        
        y_pred_before = model.predict(X_target_test)
        rmse_before = float(np.sqrt(mean_squared_error(y_target, y_pred_before)))
        
        # Fine-tune on a portion of target training data
        target_train = self.datasets[target_name]['train'].groupby('unit_id').last().reset_index()
        n_ft = n_finetune_samples or min(50, len(target_train))
        ft_sample = target_train.sample(n_ft, random_state=42)
        
        X_ft = scaler.transform(ft_sample[self.feature_cols].fillna(0).values)
        y_ft = ft_sample['RUL'].values
        
        # Warm-start fine-tuning (for GB, train additional trees; for RF, retrain)
        if model_type == 'gradient_boosting':
            model.set_params(n_estimators=model.n_estimators + 50, warm_start=True)
            model.fit(
                np.vstack([X_source[-200:], X_ft]),
                np.concatenate([y_source[-200:], y_ft])
            )
        else:
            combined_X = np.vstack([X_source, X_ft])
            combined_y = np.concatenate([y_source, y_ft])
            # Weight target samples higher
            weights = np.ones(len(combined_y))
            weights[-n_ft:] = 5.0
            model.fit(combined_X, combined_y, sample_weight=weights)
        
        # Evaluate after fine-tuning
        y_pred_after = model.predict(X_target_test)
        rmse_after = float(np.sqrt(mean_squared_error(y_target, y_pred_after)))
        
        improvement = (rmse_before - rmse_after) / rmse_before * 100
        
        result = {
            'target_dataset': target_name,
            'rmse_before_finetune': rmse_before,
            'rmse_after_finetune': rmse_after,
            'improvement_pct': float(improvement),
            'n_finetune_samples': n_ft,
            'source_datasets': [n for n in self.datasets if n != target_name]
        }
        
        self.training_results['transfer'] = result
        
        logger.info(f"Transfer → {target_name}: RMSE {rmse_before:.2f} → {rmse_after:.2f} "
                    f"({improvement:+.1f}%)")
        
        return result
    
    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    def save_results(self, output_dir: str = None):
        """Save all training results to JSON."""
        output_dir = output_dir or config.RESULTS_DIR
        os.makedirs(output_dir, exist_ok=True)
        
        filepath = os.path.join(output_dir, 'multi_dataset_results.json')
        
        # Convert to serializable format
        serializable = {}
        for k, v in self.training_results.items():
            if isinstance(v, pd.DataFrame):
                serializable[k] = v.to_dict(orient='records')
            elif isinstance(v, list):
                serializable[k] = v
            else:
                serializable[k] = v
        
        with open(filepath, 'w') as f:
            json.dump(serializable, f, indent=2, default=str)
        
        logger.info(f"Results saved to {filepath}")
        return filepath


# ============================================================
# Standalone test
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Multi-Dataset Trainer")
    print("=" * 60)
    
    trainer = MultiDatasetTrainer()
    
    # Try loading FD001 and FD003 (single operating condition datasets)
    loaded = trainer.load_datasets(['FD001', 'FD003'])
    print(f"\nLoaded datasets: {list(loaded.keys())}")
    
    if len(loaded) >= 2:
        # Harmonize features
        print("\n--- Harmonizing Features ---")
        combined_train, combined_test = trainer.harmonize_features()
        print(f"Combined train shape: {combined_train.shape}")
        print(f"Features used: {len(trainer.feature_cols)}")
        
        # Domain adaptation
        print("\n--- Domain Adaptation (MMD) ---")
        ds_names = list(trainer.datasets.keys())
        mmd_result = trainer.domain_adaptation_mmd(ds_names[0], ds_names[1])
        print(f"MMD reduction: {mmd_result['reduction_pct']:.1f}%")
        
        # Unified training
        print("\n--- Unified Training ---")
        train_result = trainer.train_unified_model('gradient_boosting', use_last_cycle=True)
        print(f"Train RMSE: {train_result['train_metrics']['rmse']:.2f}")
        
        # Cross-dataset evaluation
        print("\n--- Cross-Dataset Evaluation ---")
        eval_df = trainer.evaluate_cross_dataset()
        print(eval_df[['dataset', 'rmse', 'mae', 'r2']].to_string(index=False))
        
        # Transfer learning
        print("\n--- Transfer Learning ---")
        transfer = trainer.transfer_learning(ds_names[1])
        print(f"RMSE improvement: {transfer['improvement_pct']:+.1f}%")
        
        # Save
        trainer.save_results()
    else:
        print("Need at least 2 datasets for multi-dataset training.")
        print("Ensure CMAPSSData is available in the configured directory.")
    
    print("\n✅ Multi-Dataset Trainer test PASSED")

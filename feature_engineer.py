"""
Feature Engineering for NASA C-MAPSS Dataset.

Creates rolling statistics, rate-of-change features, and physics-based
health indicators from sensor data.  Also provides data augmentation
and automated feature selection.

Classes:
    FeatureEngineer           — Rolling stats, rate-of-change, health indicators.
    TimeSeriesAugmenter       — Jitter, scaling, window-slice, interpolation.
    FeatureImportanceAnalyzer — Permutation importance and redundancy detection.
    FeatureSelector           — Variance threshold and RFE-based selection.

Usage::

    from feature_engineer import engineer_features
    result = engineer_features(train_df)
    train_engineered = result['train']
"""

import pandas as pd
import numpy as np
from typing import List, Dict
import config
from utils import setup_logging, add_rolling_statistics, calculate_rate_of_change

logger = setup_logging(__name__)


class FeatureEngineer:
    """
    Feature engineering for turbofan engine degradation prediction
    Creates time-series features from sensor data
    """
    
    def __init__(self):
        """Initialize feature engineer"""
        self.engineered_features = []
        logger.info("Initialized Feature Engineer")
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all engineered features
        
        Args:
            df: DataFrame with sensor data
            
        Returns:
            DataFrame with added engineered features
        """
        logger.info("Creating all engineered features...")
        
        # Create a copy
        df_engineered = df.copy()
        
        # Get sensor columns
        sensor_cols = [col for col in df.columns if col.startswith('sensor_')]
        logger.info(f"Engineering features for {len(sensor_cols)} sensors")
        
        # 1. Rolling window statistics
        if config.ROLLING_WINDOW_SIZES:
            df_engineered = self.add_rolling_features(
                df_engineered, 
                sensor_cols,
                config.ROLLING_WINDOW_SIZES,
                config.ROLLING_FEATURES
            )
        
        # 2. Rate of change features
        if config.CREATE_RATE_OF_CHANGE:
            df_engineered = self.add_rate_of_change_features(df_engineered, sensor_cols)
        
        # 3. Health indicators (domain-specific combinations)
        df_engineered = self.add_health_indicators(df_engineered)
        
        # Track engineered features
        original_cols = set(df.columns)
        new_cols = set(df_engineered.columns) - original_cols
        self.engineered_features = list(new_cols)
        
        logger.info(f"Feature engineering complete. Added {len(self.engineered_features)} new features")
        logger.info(f"Final feature count: {len(df_engineered.columns)}")
        
        return df_engineered
    
    def add_rolling_features(self, 
                            df: pd.DataFrame, 
                            columns: List[str],
                            window_sizes: List[int],
                            stats: List[str]) -> pd.DataFrame:
        """
        Add rolling window statistics
        
        Args:
            df: DataFrame
            columns: Columns to compute rolling statistics
            window_sizes: List of window sizes
            stats: List of statistics ('mean', 'std', 'min', 'max')
            
        Returns:
            DataFrame with rolling features
        """
        logger.info(f"Adding rolling statistics for {len(columns)} columns...")
        
        df_rolled = add_rolling_statistics(df, columns, window_sizes, stats)
        
        return df_rolled
    
    def add_rate_of_change_features(self, 
                                    df: pd.DataFrame, 
                                    columns: List[str],
                                    periods: int = 1) -> pd.DataFrame:
        """
        Add rate of change (degradation rate) features
        
        Args:
            df: DataFrame
            columns: Columns to compute rate of change
            periods: Number of periods for difference calculation
            
        Returns:
            DataFrame with rate-of-change features
        """
        logger.info(f"Adding rate-of-change features for {len(columns)} columns...")
        
        df_roc = calculate_rate_of_change(df, columns, periods)
        
        return df_roc
    
    def add_health_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add domain-specific health indicators
        
        Combines multiple sensors to create meaningful health metrics
        based on turbofan engine physics
        
        Args:
            df: DataFrame with sensor data
            
        Returns:
            DataFrame with health indicator features
        """
        logger.info("Adding health indicators...")
        
        df_health = df.copy()
        
        # Health Indicator 1: Temperature Ratio
        # Ratio of HPC outlet temp to fan inlet temp (thermal efficiency indicator)
        if 'sensor_3' in df.columns and 'sensor_1' in df.columns:
            df_health['health_temp_ratio'] = df_health['sensor_3'] / (df_health['sensor_1'] + 1e-6)
        
        # Health Indicator 2: Pressure Ratio
        # Ratio of HPC outlet pressure to fan inlet pressure (compression efficiency)
        if 'sensor_7' in df.columns and 'sensor_5' in df.columns:
            df_health['health_pressure_ratio'] = df_health['sensor_7'] / (df_health['sensor_5'] + 1e-6)
        
        # Health Indicator 3: Speed Ratio
        # Ratio of physical core speed to fan speed (mechanical health)
        if 'sensor_9' in df.columns and 'sensor_8' in df.columns:
            df_health['health_speed_ratio'] = df_health['sensor_9'] / (df_health['sensor_8'] + 1e-6)
        
        # Health Indicator 4: Corrected Speed Average
        # Average of corrected fan and core speeds (overall rotational health)
        if 'sensor_13' in df.columns and 'sensor_14' in df.columns:
            df_health['health_corrected_speed_avg'] = (df_health['sensor_13'] + df_health['sensor_14']) / 2
        
        # Health Indicator 5: Temperature Spread
        # Difference between LPT outlet and fan inlet temperature (thermal stress)
        if 'sensor_4' in df.columns and 'sensor_1' in df.columns:
            df_health['health_temp_spread'] = df_health['sensor_4'] - df_health['sensor_1']
        
        # Health Indicator 6: Coolant Bleed Total
        # Sum of HPT and LPT coolant bleed (cooling demand indicator)
        if 'sensor_20' in df.columns and 'sensor_21' in df.columns:
            df_health['health_coolant_total'] = df_health['sensor_20'] + df_health['sensor_21']
        
        logger.info("Health indicators added")
        
        return df_health
    
    def get_engineered_feature_names(self) -> List[str]:
        """
        Get list of engineered feature names
        
        Returns:
            List of engineered feature names
        """
        return self.engineered_features
    
    def select_top_features(self, 
                           df: pd.DataFrame, 
                           target_col: str = 'RUL',
                           top_k: int = 50) -> List[str]:
        """
        Select top K most correlated features with target
        
        Args:
            df: DataFrame with features and target
            target_col: Target column name
            top_k: Number of top features to select
            
        Returns:
            List of selected feature names
        """
        logger.info(f"Selecting top {top_k} features based on correlation with {target_col}...")
        
        # Get feature columns (exclude meta columns)
        feature_cols = [col for col in df.columns 
                       if col not in ['unit_id', 'time_cycles', target_col]]
        
        # Calculate absolute correlation with target
        correlations = df[feature_cols + [target_col]].corr()[target_col].abs()
        correlations = correlations.drop(target_col).sort_values(ascending=False)
        
        # Select top K
        selected_features = correlations.head(top_k).index.tolist()
        
        logger.info(f"Selected {len(selected_features)} top features")
        logger.info(f"Top 5 features: {selected_features[:5]}")
        
        return selected_features
    
    # Backward compatible method aliases
    def create_rolling_features(self, df, columns, window_sizes, stats):
        """Alias for add_rolling_features"""
        return self.add_rolling_features(df, columns, window_sizes, stats)
    
    def create_rate_of_change_features(self, df, columns, periods=1):
        """Alias for add_rate_of_change_features"""
        return self.add_rate_of_change_features(df, columns, periods)
    
    def create_health_indicators(self, df):
        """Alias for add_health_indicators"""
        return self.add_health_indicators(df)


class TimeSeriesAugmenter:
    """
    Data augmentation techniques for time-series sensor data
    Improves model robustness by creating synthetic training samples
    """
    
    def __init__(self, random_seed: int = None):
        """
        Initialize augmenter
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_state = np.random.RandomState(random_seed)
        logger.info("Initialized TimeSeriesAugmenter")
    
    def jitter(self, 
               df: pd.DataFrame, 
               columns: List[str], 
               sigma: float = 0.03) -> pd.DataFrame:
        """
        Add Gaussian noise to sensor values
        
        Simulates sensor measurement noise and variations
        
        Args:
            df: DataFrame with sensor data
            columns: Columns to apply jittering
            sigma: Standard deviation of noise (as fraction of column std)
            
        Returns:
            Augmented DataFrame
        """
        df_aug = df.copy()
        
        for col in columns:
            if col in df_aug.columns:
                col_std = df_aug[col].std()
                noise = self.random_state.normal(0, sigma * col_std, size=len(df_aug))
                df_aug[col] = df_aug[col] + noise
        
        return df_aug
    
    def scaling(self,
                df: pd.DataFrame,
                columns: List[str],
                sigma: float = 0.1) -> pd.DataFrame:
        """
        Apply random scaling to sensor values
        
        Simulates calibration differences and unit-to-unit variations
        
        Args:
            df: DataFrame with sensor data
            columns: Columns to apply scaling
            sigma: Standard deviation of scaling factor
            
        Returns:
            Augmented DataFrame
        """
        df_aug = df.copy()
        
        for col in columns:
            if col in df_aug.columns:
                scale_factor = self.random_state.normal(1.0, sigma)
                df_aug[col] = df_aug[col] * scale_factor
        
        return df_aug
    
    def window_slice(self,
                     df: pd.DataFrame,
                     slice_ratio: float = 0.9) -> pd.DataFrame:
        """
        Randomly slice a portion of time series per engine
        
        Creates variation in degradation trajectory length
        
        Args:
            df: DataFrame with unit_id column
            slice_ratio: Ratio of sequence to keep (0.8-1.0)
            
        Returns:
            Augmented DataFrame with sliced sequences
        """
        df_aug = []
        
        for unit_id in df['unit_id'].unique():
            unit_data = df[df['unit_id'] == unit_id].copy()
            n_samples = len(unit_data)
            
            # Calculate slice size
            slice_size = int(n_samples * slice_ratio)
            if slice_size < 10:  # Minimum sequence length
                df_aug.append(unit_data)
                continue
            
            # Random start point (keeping end intact for RUL consistency)
            max_start = n_samples - slice_size
            start_idx = self.random_state.randint(0, max_start + 1)
            
            sliced_data = unit_data.iloc[start_idx:start_idx + slice_size].copy()
            
            # Update time_cycles if present
            if 'time_cycles' in sliced_data.columns:
                sliced_data['time_cycles'] = range(1, len(sliced_data) + 1)
            
            df_aug.append(sliced_data)
        
        return pd.concat(df_aug, ignore_index=True)
    
    def degradation_interpolation(self,
                                  df: pd.DataFrame,
                                  columns: List[str],
                                  n_new_engines: int = 10) -> pd.DataFrame:
        """
        Create synthetic engines by interpolating between existing degradation paths
        
        Generates new training samples by blending similar engines
        
        Args:
            df: DataFrame with unit_id and sensor data
            columns: Sensor columns to interpolate
            n_new_engines: Number of synthetic engines to create
            
        Returns:
            DataFrame with original + synthetic engines
        """
        df_aug = [df.copy()]
        
        unique_units = df['unit_id'].unique()
        max_unit_id = df['unit_id'].max()
        
        for i in range(n_new_engines):
            # Randomly select two engines to blend
            unit1, unit2 = self.random_state.choice(unique_units, size=2, replace=False)
            
            engine1 = df[df['unit_id'] == unit1].copy()
            engine2 = df[df['unit_id'] == unit2].copy()
            
            # Use the shorter sequence length
            min_len = min(len(engine1), len(engine2))
            engine1 = engine1.tail(min_len).reset_index(drop=True)
            engine2 = engine2.tail(min_len).reset_index(drop=True)
            
            # Random blend ratio
            alpha = self.random_state.uniform(0.3, 0.7)
            
            # Create synthetic engine
            synthetic = engine1.copy()
            synthetic['unit_id'] = max_unit_id + i + 1
            
            for col in columns:
                if col in synthetic.columns:
                    synthetic[col] = alpha * engine1[col].values + (1 - alpha) * engine2[col].values
            
            # Interpolate RUL if present
            if 'RUL' in synthetic.columns:
                synthetic['RUL'] = alpha * engine1['RUL'].values + (1 - alpha) * engine2['RUL'].values
            
            df_aug.append(synthetic)
        
        logger.info(f"Created {n_new_engines} synthetic engines via interpolation")
        return pd.concat(df_aug, ignore_index=True)
    
    def augment_dataset(self,
                        df: pd.DataFrame,
                        methods: List[str] = None,
                        n_augmentations: int = 1) -> pd.DataFrame:
        """
        Apply multiple augmentation methods to create expanded dataset
        
        Args:
            df: Original DataFrame
            methods: List of methods to apply ('jitter', 'scaling', 'slice', 'interpolation')
            n_augmentations: Number of augmented copies to create
            
        Returns:
            Augmented DataFrame (original + augmented samples)
        """
        if methods is None:
            methods = ['jitter', 'scaling']
        
        sensor_cols = [col for col in df.columns if col.startswith('sensor_')]
        
        all_data = [df.copy()]
        max_unit_id = df['unit_id'].max()
        
        for aug_idx in range(n_augmentations):
            df_aug = df.copy()
            
            # Update unit IDs to avoid duplicates
            df_aug['unit_id'] = df_aug['unit_id'] + max_unit_id * (aug_idx + 1)
            
            for method in methods:
                if method == 'jitter':
                    df_aug = self.jitter(df_aug, sensor_cols)
                elif method == 'scaling':
                    df_aug = self.scaling(df_aug, sensor_cols)
                elif method == 'slice':
                    df_aug = self.window_slice(df_aug, slice_ratio=0.9)
            
            all_data.append(df_aug)
        
        result = pd.concat(all_data, ignore_index=True)
        logger.info(f"Augmentation complete. Original: {len(df)}, Augmented: {len(result)}")
        
        return result


def engineer_features(train_df: pd.DataFrame, 
                     test_df: pd.DataFrame = None) -> Dict:
    """
    Convenience function to engineer features for train and test data
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame (optional)
        
    Returns:
        Dictionary with engineered DataFrames
    """
    logger.info("Starting feature engineering pipeline...")
    
    engineer = FeatureEngineer()
    
    # Engineer train features
    train_engineered = engineer.create_all_features(train_df)
    
    # Engineer test features if provided
    test_engineered = None
    if test_df is not None:
        test_engineered = engineer.create_all_features(test_df)
    
    logger.info("Feature engineering pipeline complete")
    
    return {
        'train': train_engineered,
        'test': test_engineered,
        'engineer': engineer,
        'feature_names': engineer.get_engineered_feature_names()
    }


class FeatureImportanceAnalyzer:
    """
    Analyze feature importance and correlations
    Identifies redundant features and recommends optimal subsets
    """
    
    def __init__(self):
        """Initialize feature importance analyzer"""
        self.importance_scores = {}
        self.correlations = None
        self.redundant_pairs = []
        logger.info("Initialized FeatureImportanceAnalyzer")
    
    def calculate_permutation_importance(self,
                                         model,
                                         X: np.ndarray,
                                         y: np.ndarray,
                                         feature_names: List[str],
                                         n_repeats: int = 10) -> Dict:
        """
        Calculate feature importance via permutation
        
        Args:
            model: Trained model with predict method
            X: Feature matrix
            y: Target values
            feature_names: List of feature names
            n_repeats: Number of permutation repeats
            
        Returns:
            Dictionary with importance scores
        """
        from sklearn.metrics import mean_squared_error
        
        logger.info(f"Calculating permutation importance for {len(feature_names)} features...")
        
        # Baseline score
        baseline_pred = model.predict(X)
        if hasattr(baseline_pred, 'ravel'):
            baseline_pred = baseline_pred.ravel()
        baseline_score = mean_squared_error(y, baseline_pred)
        
        importances = {}
        
        for i, feature in enumerate(feature_names):
            scores = []
            
            for _ in range(n_repeats):
                # Permute feature
                X_permuted = X.copy()
                X_permuted[:, i] = np.random.permutation(X_permuted[:, i])
                
                # Score with permuted feature
                perm_pred = model.predict(X_permuted)
                if hasattr(perm_pred, 'ravel'):
                    perm_pred = perm_pred.ravel()
                perm_score = mean_squared_error(y, perm_pred)
                
                scores.append(perm_score - baseline_score)
            
            importances[feature] = {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'importance_ratio': float(np.mean(scores) / (baseline_score + 1e-10))
            }
        
        # Sort by importance
        sorted_features = sorted(importances.items(), 
                                key=lambda x: x[1]['mean'], reverse=True)
        
        self.importance_scores = {
            'baseline_mse': float(baseline_score),
            'features': dict(sorted_features),
            'top_10': [f[0] for f in sorted_features[:10]]
        }
        
        logger.info(f"Top 5 important features: {[f[0] for f in sorted_features[:5]]}")
        
        return self.importance_scores
    
    def analyze_feature_correlations(self,
                                     df: pd.DataFrame,
                                     feature_cols: List[str],
                                     method: str = 'pearson') -> pd.DataFrame:
        """
        Analyze correlations between features
        
        Args:
            df: DataFrame with features
            feature_cols: Feature column names
            method: Correlation method ('pearson', 'spearman', 'kendall')
            
        Returns:
            Correlation matrix
        """
        logger.info(f"Analyzing correlations for {len(feature_cols)} features...")
        
        # Calculate correlation matrix
        self.correlations = df[feature_cols].corr(method=method)
        
        return self.correlations
    
    def identify_redundant_features(self,
                                    df: pd.DataFrame,
                                    feature_cols: List[str],
                                    threshold: float = 0.95) -> List[tuple]:
        """
        Identify highly correlated (redundant) feature pairs
        
        Args:
            df: DataFrame with features
            feature_cols: Feature column names
            threshold: Correlation threshold for redundancy
            
        Returns:
            List of (feature1, feature2, correlation) tuples
        """
        logger.info(f"Identifying redundant features (threshold: {threshold})...")
        
        if self.correlations is None:
            self.analyze_feature_correlations(df, feature_cols)
        
        redundant = []
        
        for i in range(len(feature_cols)):
            for j in range(i + 1, len(feature_cols)):
                corr_value = abs(self.correlations.iloc[i, j])
                if corr_value >= threshold:
                    redundant.append((
                        feature_cols[i],
                        feature_cols[j],
                        float(corr_value)
                    ))
        
        # Sort by correlation
        redundant.sort(key=lambda x: x[2], reverse=True)
        self.redundant_pairs = redundant
        
        logger.info(f"Found {len(redundant)} redundant feature pairs")
        
        return redundant
    
    def recommend_feature_subset(self,
                                 df: pd.DataFrame,
                                 feature_cols: List[str],
                                 target_col: str = 'RUL',
                                 max_features: int = None,
                                 correlation_threshold: float = 0.9) -> Dict:
        """
        Recommend optimal feature subset
        
        Removes redundant features while keeping most predictive ones
        
        Args:
            df: DataFrame with features and target
            feature_cols: All feature columns
            target_col: Target column name
            max_features: Maximum features to select
            correlation_threshold: Threshold for removing redundant features
            
        Returns:
            Feature selection recommendations
        """
        logger.info("Generating feature subset recommendations...")
        
        # Get target correlations
        target_corrs = df[feature_cols].corrwith(df[target_col]).abs()
        target_corrs = target_corrs.sort_values(ascending=False)
        
        # Identify redundant pairs
        self.identify_redundant_features(df, feature_cols, correlation_threshold)
        
        # Select features: prefer those with higher target correlation
        selected = []
        dropped = []
        
        for feature in target_corrs.index:
            if feature in dropped:
                continue
            
            selected.append(feature)
            
            # Remove redundant features
            for f1, f2, corr in self.redundant_pairs:
                if f1 == feature and f2 not in dropped and f2 not in selected:
                    dropped.append(f2)
                elif f2 == feature and f1 not in dropped and f1 not in selected:
                    dropped.append(f1)
        
        # Limit to max_features if specified
        if max_features and len(selected) > max_features:
            selected = selected[:max_features]
        
        recommendations = {
            'selected_features': selected,
            'dropped_features': dropped,
            'original_count': len(feature_cols),
            'selected_count': len(selected),
            'reduction_pct': (1 - len(selected) / len(feature_cols)) * 100,
            'target_correlations': {f: float(target_corrs[f]) for f in selected[:10]},
            'redundant_pairs_removed': len(self.redundant_pairs)
        }
        
        logger.info(f"Recommended {len(selected)} features "
                   f"(reduced by {recommendations['reduction_pct']:.1f}%)")
        
        return recommendations
    
    def get_importance_report(self) -> str:
        """Generate formatted importance report"""
        lines = [
            "=" * 60,
            "FEATURE IMPORTANCE REPORT",
            "=" * 60,
            ""
        ]
        
        if self.importance_scores:
            lines.append(f"Baseline MSE: {self.importance_scores['baseline_mse']:.4f}")
            lines.append("")
            lines.append("Top 10 Important Features:")
            
            for i, feat in enumerate(self.importance_scores['top_10'][:10], 1):
                imp_data = self.importance_scores['features'][feat]
                lines.append(f"  {i}. {feat}: {imp_data['mean']:.4f} (±{imp_data['std']:.4f})")
        
        if self.redundant_pairs:
            lines.extend([
                "",
                f"Redundant Feature Pairs: {len(self.redundant_pairs)}"
            ])
            for f1, f2, corr in self.redundant_pairs[:5]:
                lines.append(f"  • {f1} ↔ {f2}: {corr:.3f}")
        
        lines.append("=" * 60)
        
        return '\n'.join(lines)


        return '\n'.join(lines)


class FeatureSelector:
    """
    Automated feature selection
    Uses RFE and Variance Thresholding to identify best features
    """
    
    def __init__(self, n_features_to_select: int = None):
        """
        Initialize feature selector
        
        Args:
            n_features_to_select: Number of features to select (None = automatic)
        """
        self.n_features_to_select = n_features_to_select
        self.selected_features = None
        self.feature_importance = {}
        logger.info("Initialized FeatureSelector")
    
    def select_by_variance(self,
                          df: pd.DataFrame,
                          feature_cols: List[str],
                          threshold: float = 0.0) -> List[str]:
        """
        Select features by variance threshold
        
        Args:
            df: Feature DataFrame
            feature_cols: Features to check
            threshold: Minimum variance threshold
            
        Returns:
            List of selected feature names
        """
        from sklearn.feature_selection import VarianceThreshold
        
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(df[feature_cols])
        
        selected_mask = selector.get_support()
        selected_features = [f for f, s in zip(feature_cols, selected_mask) if s]
        
        logger.info(f"Variance Selection: {len(selected_features)}/{len(feature_cols)} features kept (threshold={threshold})")
        
        return selected_features
    
    def select_by_rfe(self,
                     df: pd.DataFrame,
                     target: pd.Series,
                     feature_cols: List[str],
                     n_features: int = 10) -> List[str]:
        """
        Select features using Recursive Feature Elimination
        
        Args:
            df: Feature DataFrame
            target: Target values
            feature_cols: Features to select from
            n_features: Number to select
            
        Returns:
            List of selected feature names
        """
        from sklearn.feature_selection import RFE
        from sklearn.ensemble import RandomForestRegressor
        
        estimator = RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=42)
        selector = RFE(estimator, n_features_to_select=n_features, step=1)
        
        logger.info(f"Running RFE to select {n_features} features from {len(feature_cols)} candidates...")
        selector.fit(df[feature_cols], target)
        
        selected_mask = selector.support_
        selected_features = [f for f, s in zip(feature_cols, selected_mask) if s]
        
        # Store rankings
        self.feature_importance = dict(zip(feature_cols, selector.ranking_))
        
        logger.info(f"RFE Selection complete: {len(selected_features)} features selected")
        
        return selected_features
    
    def run_automatic_selection(self,
                               df: pd.DataFrame,
                               target: pd.Series,
                               feature_cols: List[str]) -> Dict:
        """
        Run complete automated selection workflow
        
        Args:
            df: Feature DataFrame
            target: Target values
            feature_cols: Initial feature list
            
        Returns:
            Selection results
        """
        # 1. Variance Threshold
        low_variance_keepers = self.select_by_variance(df, feature_cols, threshold=0.01)
        
        # 2. RFE on remaining features
        n_select = self.n_features_to_select or max(5, len(low_variance_keepers) // 2)
        final_features = self.select_by_rfe(df, target, low_variance_keepers, n_features=n_select)
        
        self.selected_features = final_features
        
        result = {
            'initial_count': len(feature_cols),
            'after_variance': len(low_variance_keepers),
            'final_count': len(final_features),
            'selected_features': final_features,
            'dropped_variance': list(set(feature_cols) - set(low_variance_keepers)),
            'dropped_rfe': list(set(low_variance_keepers) - set(final_features))
        }
        
        return result
    
    def get_feature_rankings(self) -> pd.DataFrame:
        """Get feature importance rankings"""
        if not self.feature_importance:
            return pd.DataFrame()
        
        rankings = pd.DataFrame(list(self.feature_importance.items()), columns=['feature', 'rank'])
        return rankings.sort_values('rank')


if __name__ == "__main__":
    # Test the feature engineer
    from data_loader import load_dataset
    from preprocessor import preprocess_data
    
    print("="*60)
    print("Testing Feature Engineering")
    print("="*60)
    
    # Load and preprocess FD001 dataset
    train_df, test_df, rul_df = load_dataset('FD001')
    preprocessed = preprocess_data(train_df, test_df, rul_df)
    
    # Engineer features
    engineered = engineer_features(
        preprocessed['train'], 
        preprocessed['test']
    )
    
    print("\nEngineered Data Shapes:")
    print(f"  Train: {engineered['train'].shape}")
    if engineered['test'] is not None:
        print(f"  Test: {engineered['test'].shape}")
    
    print(f"\nNumber of engineered features: {len(engineered['feature_names'])}")
    print(f"Sample engineered features: {engineered['feature_names'][:10]}")
    
    print("\nTrain Data Sample (with engineered features):")
    sample_cols = ['unit_id', 'time_cycles', 'RUL'] + engineered['feature_names'][:3]
    print(engineered['train'][sample_cols].head())
    
    # Test feature selection
    top_features = engineered['engineer'].select_top_features(
        engineered['train'], 
        target_col='RUL', 
        top_k=20
    )
    print(f"\nTop 10 features by correlation with RUL:")
    for i, feat in enumerate(top_features[:10], 1):
        print(f"  {i}. {feat}")
    
    print("\n" + "="*60)
    print("Feature engineering test complete!")
    print("="*60)

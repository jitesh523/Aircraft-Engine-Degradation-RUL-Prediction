"""
Data Preprocessor for NASA C-MAPSS Dataset.

Handles RUL label creation, feature normalization (MinMax / Standard),
train-validation splitting by engine unit, and pipeline validation.

Classes:
    CMAPSSPreprocessor      — RUL labels, normalization, scaler save/load.
    DataPipelineValidator   — Leakage checks, RUL validation, scaling QA.
    DataAugmentor           — Noise injection, time warping, oversampling.

Usage::

    from preprocessor import preprocess_data
    result = preprocess_data(train_df, test_df, rul_df)
    train_norm = result['train']
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, List
import pickle
import config
from utils import setup_logging, add_remaining_useful_life

logger = setup_logging(__name__)


class CMAPSSPreprocessor:
    """
    Preprocessor for NASA C-MAPSS data
    Handles RUL calculation, normalization, and feature preparation
    """
    
    def __init__(self, normalization_method: str = 'minmax'):
        """
        Initialize preprocessor
        
        Args:
            normalization_method: 'minmax' or 'standard' normalization
        """
        self.normalization_method = normalization_method
        self.scaler = None
        self.feature_columns = None
        self.columns_to_drop = config.SENSORS_TO_DROP
        
        logger.info(f"Initialized preprocessor with {normalization_method} normalization")
    
    def prepare_train_data(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare training data with RUL labels
        
        Args:
            train_df: Raw training DataFrame
            
        Returns:
            Preprocessed training DataFrame with RUL column
        """
        logger.info("Preparing training data...")
        
        # Create a copy to avoid modifying original
        df = train_df.copy()
        
        # Add RUL labels
        df = add_remaining_useful_life(df)
        
        # Drop low-variance sensors
        columns_present = [col for col in self.columns_to_drop if col in df.columns]
        if columns_present:
            df = df.drop(columns=columns_present)
            logger.info(f"Dropped {len(columns_present)} low-variance sensor columns")
        
        logger.info(f"Training data prepared. Shape: {df.shape}")
        return df
    
    def prepare_test_data(self, test_df: pd.DataFrame, rul_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare test data with RUL labels from ground truth
        
        Args:
            test_df: Raw test DataFrame
            rul_df: Ground truth RUL labels
            
        Returns:
            Tuple of (test_df with features, rul_labels)
        """
        logger.info("Preparing test data...")
        
        # Create a copy
        df = test_df.copy()
        
        # Drop low-variance sensors (same as training)
        columns_present = [col for col in self.columns_to_drop if col in df.columns]
        if columns_present:
            df = df.drop(columns=columns_present)
        
        logger.info(f"Test data prepared. Shape: {df.shape}")
        return df, rul_df
    
    def normalize_features(self, 
                          train_df: pd.DataFrame, 
                          test_df: pd.DataFrame = None,
                          fit: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Normalize sensor and setting features
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame (optional)
            fit: Whether to fit the scaler on train_df
            
        Returns:
            Tuple of (normalized_train_df, normalized_test_df)
        """
        logger.info(f"Normalizing features using {self.normalization_method} method...")
        
        # Identify feature columns (sensors and settings)
        feature_cols = [col for col in train_df.columns 
                       if col.startswith('sensor_') or col.startswith('setting_')]
        
        # Also include any engineered features
        feature_cols += [col for col in train_df.columns 
                        if 'rolling' in col or 'roc' in col]
        
        # Remove duplicates
        feature_cols = list(set(feature_cols))
        self.feature_columns = feature_cols
        
        logger.info(f"Normalizing {len(feature_cols)} features")
        
        # Create scaler
        if fit:
            if self.normalization_method == 'minmax':
                self.scaler = MinMaxScaler()
            elif self.normalization_method == 'standard':
                self.scaler = StandardScaler()
            else:
                raise ValueError(f"Unknown normalization method: {self.normalization_method}")
            
            # Fit on training data
            self.scaler.fit(train_df[feature_cols])
            logger.info("Scaler fitted on training data")
        
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Set fit=True or load a fitted scaler.")
        
        # Transform training data
        train_normalized = train_df.copy()
        train_normalized[feature_cols] = self.scaler.transform(train_df[feature_cols])
        
        # Transform test data if provided
        test_normalized = None
        if test_df is not None:
            test_normalized = test_df.copy()
            test_normalized[feature_cols] = self.scaler.transform(test_df[feature_cols])
            logger.info("Test data normalized")
        
        logger.info("Feature normalization complete")
        return train_normalized, test_normalized
    
    def split_train_validation(self, 
                               train_df: pd.DataFrame, 
                               val_split: float = 0.2,
                               random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split training data into train and validation sets
        
        Split by engine units to ensure temporal integrity
        
        Args:
            train_df: Training DataFrame
            val_split: Fraction of engines to use for validation
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_df, val_df)
        """
        logger.info(f"Splitting training data with {val_split*100}% validation...")
        
        # Get unique engine units
        unique_units = train_df['unit_id'].unique()
        
        # Split engine units
        train_units, val_units = train_test_split(
            unique_units, 
            test_size=val_split, 
            random_state=random_state
        )
        
        # Create train and validation sets
        train_set = train_df[train_df['unit_id'].isin(train_units)].copy()
        val_set = train_df[train_df['unit_id'].isin(val_units)].copy()
        
        logger.info(f"Train set: {len(train_units)} engines, {len(train_set)} samples")
        logger.info(f"Validation set: {len(val_units)} engines, {len(val_set)} samples")
        
        return train_set, val_set
    
    def get_feature_columns(self) -> List[str]:
        """
        Get list of feature columns used for modeling
        
        Returns:
            List of feature column names
        """
        if self.feature_columns is None:
            logger.warning("Feature columns not set. Run normalize_features first.")
            return []
        return self.feature_columns
    
    def save_scaler(self, filepath: str) -> None:
        """
        Save the fitted scaler
        
        Args:
            filepath: Path to save the scaler
        """
        if self.scaler is None:
            raise ValueError("No scaler to save. Fit the scaler first.")
        
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'normalization_method': self.normalization_method
            }, f)
        
        logger.info(f"Scaler saved to {filepath}")
    
    def load_scaler(self, filepath: str) -> None:
        """
        Load a fitted scaler
        
        Args:
            filepath: Path to the saved scaler
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.scaler = data['scaler']
        self.feature_columns = data['feature_columns']
        self.normalization_method = data['normalization_method']
        
        logger.info(f"Scaler loaded from {filepath}")


def preprocess_data(train_df: pd.DataFrame, 
                   test_df: pd.DataFrame, 
                   rul_df: pd.DataFrame,
                   normalization_method: str = 'minmax',
                   val_split: float = 0.2) -> Dict:
    """
    Convenience function to preprocess all data
    
    Args:
        train_df: Raw training data
        test_df: Raw test data
        rul_df: RUL labels for test data
        normalization_method: Normalization method to use
        val_split: Validation split fraction
        
    Returns:
        Dictionary with preprocessed data and scaler
    """
    logger.info("Starting full preprocessing pipeline...")
    
    # Initialize preprocessor
    preprocessor = CMAPSSPreprocessor(normalization_method)
    
    # Prepare train and test data
    train_prepared = preprocessor.prepare_train_data(train_df)
    test_prepared, rul_labels = preprocessor.prepare_test_data(test_df, rul_df)
    
    # Normalize features
    train_normalized, test_normalized = preprocessor.normalize_features(
        train_prepared, test_prepared, fit=True
    )
    
    # Split train/validation
    train_set, val_set = preprocessor.split_train_validation(
        train_normalized, val_split=val_split
    )
    
    logger.info("Preprocessing pipeline complete")
    
    return {
        'train': train_set,
        'validation': val_set,
        'test': test_normalized,
        'test_rul': rul_labels,
        'preprocessor': preprocessor,
        'feature_columns': preprocessor.get_feature_columns()
    }


# Backward compatible alias
Preprocessor = CMAPSSPreprocessor


class DataPipelineValidator:
    """
    Validates the entire preprocessing pipeline
    Checks for data integrity, leakage, and correctness
    """
    
    def __init__(self):
        """Initialize pipeline validator"""
        self.validation_results = {}
        logger.info("Initialized DataPipelineValidator")
    
    def validate_data_flow(self,
                          raw_train: pd.DataFrame,
                          processed_train: pd.DataFrame) -> Dict:
        """
        Validate data flow from raw to processed
        
        Args:
            raw_train: Raw training data
            processed_train: Processed training data
            
        Returns:
            Dictionary with validation results
        """
        logger.info("Validating data flow integrity...")
        
        issues = []
        
        # Check row count consistency
        raw_count = len(raw_train)
        processed_count = len(processed_train)
        
        if processed_count > raw_count:
            issues.append(f"Data expansion detected: {raw_count} -> {processed_count} rows")
        
        # Check unit_id preservation
        raw_units = set(raw_train['unit_id'].unique())
        processed_units = set(processed_train['unit_id'].unique())
        
        if raw_units != processed_units:
            missing = raw_units - processed_units
            if missing:
                issues.append(f"Missing units after processing: {missing}")
        
        # Check for unexpected NaN introduction
        raw_nan_count = raw_train.isnull().sum().sum()
        processed_nan_count = processed_train.isnull().sum().sum()
        
        if processed_nan_count > raw_nan_count:
            new_nans = processed_nan_count - raw_nan_count
            issues.append(f"New NaN values introduced: {new_nans}")
        
        result = {
            'valid': len(issues) == 0,
            'raw_rows': raw_count,
            'processed_rows': processed_count,
            'issues': issues
        }
        
        self.validation_results['data_flow'] = result
        logger.info(f"Data flow validation: {'PASSED' if result['valid'] else 'FAILED'}")
        
        return result
    
    def check_data_leakage(self,
                          train_df: pd.DataFrame,
                          val_df: pd.DataFrame,
                          test_df: pd.DataFrame = None) -> Dict:
        """
        Check for data leakage between splits
        
        Args:
            train_df: Training data
            val_df: Validation data
            test_df: Test data (optional)
            
        Returns:
            Dictionary with leakage check results
        """
        logger.info("Checking for data leakage...")
        
        issues = []
        
        # Check engine overlap between train and validation
        train_units = set(train_df['unit_id'].unique())
        val_units = set(val_df['unit_id'].unique())
        
        overlap = train_units & val_units
        if overlap:
            issues.append(f"Train/Val engine overlap: {len(overlap)} engines")
        
        # Check for identical rows
        train_hashes = set(train_df.drop(columns=['unit_id'], errors='ignore').apply(tuple, axis=1))
        val_hashes = set(val_df.drop(columns=['unit_id'], errors='ignore').apply(tuple, axis=1))
        
        row_overlap = len(train_hashes & val_hashes)
        if row_overlap > 0:
            issues.append(f"Identical rows in train/val: {row_overlap}")
        
        # Check test data if provided
        if test_df is not None:
            test_units = set(test_df['unit_id'].unique()) if 'unit_id' in test_df.columns else set()
            test_train_overlap = train_units & test_units
            if test_train_overlap:
                issues.append(f"Train/Test engine overlap: {len(test_train_overlap)} engines")
        
        # Check for temporal leakage in RUL
        if 'RUL' in train_df.columns and 'time_cycles' in train_df.columns:
            # RUL should decrease with increasing time_cycles
            for unit_id in list(train_units)[:5]:  # Sample 5 engines
                unit_data = train_df[train_df['unit_id'] == unit_id].sort_values('time_cycles')
                rul_diffs = unit_data['RUL'].diff().dropna()
                
                if (rul_diffs > 0).any():
                    issues.append(f"RUL increases detected in unit {unit_id} (possible temporal leakage)")
                    break
        
        result = {
            'valid': len(issues) == 0,
            'train_engines': len(train_units),
            'val_engines': len(val_units),
            'issues': issues
        }
        
        self.validation_results['leakage'] = result
        logger.info(f"Leakage check: {'PASSED' if result['valid'] else 'FAILED'}")
        
        return result
    
    def validate_rul_labels(self, df: pd.DataFrame) -> Dict:
        """
        Validate RUL labels are correctly generated
        
        Args:
            df: DataFrame with RUL column
            
        Returns:
            Dictionary with RUL validation results
        """
        logger.info("Validating RUL labels...")
        
        issues = []
        
        if 'RUL' not in df.columns:
            return {'valid': False, 'issues': ['RUL column not found']}
        
        # Check for negative RUL
        negative_rul = (df['RUL'] < 0).sum()
        if negative_rul > 0:
            issues.append(f"Negative RUL values found: {negative_rul}")
        
        # Check RUL ends at 0 for each engine
        for unit_id in df['unit_id'].unique():
            unit_data = df[df['unit_id'] == unit_id].sort_values('time_cycles')
            final_rul = unit_data['RUL'].iloc[-1]
            
            if final_rul != 0:
                issues.append(f"Unit {unit_id}: Final RUL is {final_rul}, expected 0")
                if len(issues) > 5:  # Limit number of issues reported
                    issues.append("... and more units with incorrect final RUL")
                    break
        
        # Check RUL is monotonically decreasing
        sample_units = list(df['unit_id'].unique())[:10]
        for unit_id in sample_units:
            unit_data = df[df['unit_id'] == unit_id].sort_values('time_cycles')
            rul_diffs = unit_data['RUL'].diff().dropna()
            
            if (rul_diffs > 0).any():
                issues.append(f"Unit {unit_id}: RUL not monotonically decreasing")
                break
        
        # Check RUL range
        max_rul = df['RUL'].max()
        if max_rul > 500:  # Unusually high RUL
            issues.append(f"Unusually high max RUL: {max_rul}")
        
        result = {
            'valid': len(issues) == 0,
            'min_rul': int(df['RUL'].min()),
            'max_rul': int(df['RUL'].max()),
            'mean_rul': float(df['RUL'].mean()),
            'issues': issues
        }
        
        self.validation_results['rul_labels'] = result
        logger.info(f"RUL validation: {'PASSED' if result['valid'] else 'FAILED'}")
        
        return result
    
    def validate_scaling(self,
                        original_df: pd.DataFrame,
                        scaled_df: pd.DataFrame,
                        scaler,
                        feature_cols: List[str]) -> Dict:
        """
        Validate scaling is reversible
        
        Args:
            original_df: Original unscaled DataFrame
            scaled_df: Scaled DataFrame
            scaler: Fitted scaler object
            feature_cols: List of feature columns
            
        Returns:
            Dictionary with scaling validation results
        """
        logger.info("Validating scaling reversibility...")
        
        issues = []
        
        # Check scaled values are in expected range
        scaled_min = scaled_df[feature_cols].min().min()
        scaled_max = scaled_df[feature_cols].max().max()
        
        if hasattr(scaler, 'feature_range'):  # MinMaxScaler
            expected_min, expected_max = scaler.feature_range
            if scaled_min < expected_min - 0.01 or scaled_max > expected_max + 0.01:
                issues.append(f"Scaled values outside expected range: [{scaled_min:.2f}, {scaled_max:.2f}]")
        
        # Check reversibility
        try:
            unscaled = scaler.inverse_transform(scaled_df[feature_cols])
            original_values = original_df[feature_cols].values
            
            # Allow small floating point differences
            max_diff = np.max(np.abs(unscaled - original_values))
            if max_diff > 1e-6:
                issues.append(f"Scaling not perfectly reversible: max diff = {max_diff:.2e}")
        except Exception as e:
            issues.append(f"Could not verify reversibility: {e}")
        
        result = {
            'valid': len(issues) == 0,
            'scaled_range': (float(scaled_min), float(scaled_max)),
            'issues': issues
        }
        
        self.validation_results['scaling'] = result
        logger.info(f"Scaling validation: {'PASSED' if result['valid'] else 'FAILED'}")
        
        return result
    
    def run_full_validation(self,
                           raw_train: pd.DataFrame,
                           processed_train: pd.DataFrame,
                           train_split: pd.DataFrame,
                           val_split: pd.DataFrame,
                           scaler = None,
                           feature_cols: List[str] = None) -> Dict:
        """
        Run complete pipeline validation
        
        Args:
            raw_train: Raw training data
            processed_train: Processed training data (with RUL)
            train_split: Training split
            val_split: Validation split
            scaler: Fitted scaler (optional)
            feature_cols: Feature columns (optional)
            
        Returns:
            Complete validation report
        """
        logger.info("Running full pipeline validation...")
        
        # Data flow validation
        self.validate_data_flow(raw_train, processed_train)
        
        # Leakage check
        self.validate_data_flow(raw_train, processed_train)
        self.check_data_leakage(train_split, val_split)
        
        # RUL validation
        if 'RUL' in processed_train.columns:
            self.validate_rul_labels(processed_train)
        
        # Scaling validation
        if scaler is not None and feature_cols is not None:
            self.validate_scaling(processed_train, train_split, scaler, feature_cols)
        
        # Overall result
        all_valid = all(v.get('valid', True) for v in self.validation_results.values())
        
        return {
            'overall_valid': all_valid,
            'checks': self.validation_results
        }
    
    def print_report(self):
        """Print formatted validation report"""
        print("\n" + "="*60)
        print("DATA PIPELINE VALIDATION REPORT")
        print("="*60)
        
        for check_name, result in self.validation_results.items():
            status = "✅ PASSED" if result.get('valid', False) else "❌ FAILED"
            print(f"\n{check_name.upper()}: {status}")
            
            if result.get('issues'):
                for issue in result['issues']:
                    print(f"  ⚠️  {issue}")
        
        print("\n" + "="*60)


class DataAugmentor:
    """
    Data augmentation for time-series sensor data
    Includes noise injection, time warping, and oversampling
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize data augmentor
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
        self.augmentation_stats = {}
        logger.info(f"Initialized DataAugmentor (seed={random_seed})")
    
    def add_gaussian_noise(self,
                          data: np.ndarray,
                          noise_level: float = 0.01) -> np.ndarray:
        """
        Add Gaussian noise to sensor data
        
        Args:
            data: Input data array
            noise_level: Standard deviation of noise relative to data std
            
        Returns:
            Augmented data with noise
        """
        noise = np.random.normal(0, noise_level * np.std(data), data.shape)
        augmented = data + noise
        
        self.augmentation_stats['gaussian_noise'] = {
            'samples_augmented': len(data),
            'noise_level': noise_level
        }
        
        return augmented
    
    def time_warp(self,
                  sequences: np.ndarray,
                  warp_factor: float = 0.1) -> np.ndarray:
        """
        Apply time warping to sequences
        
        Args:
            sequences: Input sequences (n_samples, seq_len, features)
            warp_factor: Maximum warping factor
            
        Returns:
            Time-warped sequences
        """
        if len(sequences.shape) != 3:
            logger.warning("Time warp expects 3D sequences (samples, time, features)")
            return sequences
        
        n_samples, seq_len, n_features = sequences.shape
        warped = np.zeros_like(sequences)
        
        for i in range(n_samples):
            # Random warp factor per sample
            warp = 1 + np.random.uniform(-warp_factor, warp_factor)
            
            # Generate warped indices
            original_indices = np.arange(seq_len)
            warped_indices = np.linspace(0, seq_len - 1, int(seq_len * warp))
            
            # Interpolate for each feature
            for j in range(n_features):
                warped[i, :, j] = np.interp(
                    original_indices,
                    np.linspace(0, seq_len - 1, len(warped_indices)),
                    sequences[i, :int(len(warped_indices)), j] if warp > 1 else np.interp(
                        np.linspace(0, len(warped_indices) - 1, seq_len),
                        np.arange(len(warped_indices)),
                        sequences[i, :, j][:len(warped_indices)]
                    )
                )
        
        self.augmentation_stats['time_warp'] = {
            'samples_warped': n_samples,
            'warp_factor': warp_factor
        }
        
        return warped
    
    def magnitude_scale(self,
                        data: np.ndarray,
                        scale_range: Tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
        """
        Apply random magnitude scaling to data
        
        Args:
            data: Input data
            scale_range: Min and max scale factors
            
        Returns:
            Scaled data
        """
        scale_factors = np.random.uniform(scale_range[0], scale_range[1], data.shape)
        scaled = data * scale_factors
        
        self.augmentation_stats['magnitude_scale'] = {
            'samples_scaled': len(data),
            'scale_range': scale_range
        }
        
        return scaled
    
    def oversample_critical_rul(self,
                                df: pd.DataFrame,
                                rul_column: str = 'RUL',
                                threshold: int = 30,
                                factor: int = 2) -> pd.DataFrame:
        """
        Oversample data points with critical (low) RUL values
        
        Args:
            df: Input DataFrame
            rul_column: Name of RUL column
            threshold: RUL threshold for critical samples
            factor: Oversampling factor
            
        Returns:
            Augmented DataFrame with oversampled critical points
        """
        if rul_column not in df.columns:
            logger.warning(f"RUL column '{rul_column}' not found")
            return df
        
        critical_mask = df[rul_column] <= threshold
        critical_samples = df[critical_mask]
        n_critical = len(critical_samples)
        
        if n_critical == 0:
            logger.warning("No critical samples found")
            return df
        
        # Replicate critical samples
        augmented_critical = pd.concat([critical_samples] * factor, ignore_index=True)
        
        # Add slight noise to replicated samples
        numeric_cols = augmented_critical.select_dtypes(include=[np.number]).columns
        sensor_cols = [c for c in numeric_cols if 'sensor' in c]
        
        for col in sensor_cols:
            noise = np.random.normal(0, 0.01 * augmented_critical[col].std(), 
                                    len(augmented_critical))
            augmented_critical[col] = augmented_critical[col] + noise
        
        result = pd.concat([df, augmented_critical], ignore_index=True)
        
        self.augmentation_stats['oversample_critical'] = {
            'original_critical': n_critical,
            'added_samples': len(augmented_critical),
            'threshold': threshold
        }
        
        logger.info(f"Oversampled {n_critical} critical samples by {factor}x "
                   f"(added {len(augmented_critical)} samples)")
        
        return result
    
    def create_synthetic_degradation(self,
                                     df: pd.DataFrame,
                                     n_synthetic: int = 10) -> pd.DataFrame:
        """
        Create synthetic degradation sequences
        
        Args:
            df: Input DataFrame with engine data
            n_synthetic: Number of synthetic engines to generate
            
        Returns:
            DataFrame with synthetic data appended
        """
        if 'unit_id' not in df.columns:
            return df
        
        # Get statistics from existing engines
        engine_groups = df.groupby('unit_id')
        
        synthetic_dfs = []
        max_unit_id = df['unit_id'].max()
        
        for i in range(n_synthetic):
            # Select random engine as template
            template_id = np.random.choice(df['unit_id'].unique())
            template_data = engine_groups.get_group(template_id).copy()
            
            # Create synthetic version with modifications
            template_data['unit_id'] = max_unit_id + i + 1
            
            # Add noise to sensors
            sensor_cols = [c for c in template_data.columns if 'sensor' in c]
            for col in sensor_cols:
                noise = np.random.normal(0, 0.02 * template_data[col].std(),
                                        len(template_data))
                template_data[col] = template_data[col] + noise
            
            synthetic_dfs.append(template_data)
        
        result = pd.concat([df] + synthetic_dfs, ignore_index=True)
        
        self.augmentation_stats['synthetic_degradation'] = {
            'synthetic_engines': n_synthetic,
            'total_synthetic_samples': sum(len(s) for s in synthetic_dfs)
        }
        
        logger.info(f"Created {n_synthetic} synthetic engines")
        
        return result
    
    def augment_pipeline(self,
                         df: pd.DataFrame,
                         oversample: bool = True,
                         add_noise: bool = True,
                         synthetic: int = 0) -> pd.DataFrame:
        """
        Apply full augmentation pipeline
        
        Args:
            df: Input DataFrame
            oversample: Whether to oversample critical RUL
            add_noise: Whether to add Gaussian noise
            synthetic: Number of synthetic engines (0 to skip)
            
        Returns:
            Augmented DataFrame
        """
        result = df.copy()
        
        if oversample:
            result = self.oversample_critical_rul(result)
        
        if add_noise:
            sensor_cols = [c for c in result.columns if 'sensor' in c]
            for col in sensor_cols:
                result[col] = self.add_gaussian_noise(result[col].values)
        
        if synthetic > 0:
            result = self.create_synthetic_degradation(df, synthetic)
        
        return result
    
    def get_augmentation_summary(self) -> str:
        """Generate augmentation summary"""
        lines = [
            "=" * 60,
            "DATA AUGMENTATION SUMMARY",
            "=" * 60,
            ""
        ]
        
        for method, stats in self.augmentation_stats.items():
            lines.append(f"{method}:")
            for key, value in stats.items():
                lines.append(f"  {key}: {value}")
            lines.append("")
        
        lines.append("=" * 60)
        
        return '\n'.join(lines)


if __name__ == "__main__":
    # Test the preprocessor
    from data_loader import load_dataset
    
    print("="*60)
    print("Testing NASA C-MAPSS Data Preprocessor")
    print("="*60)
    
    # Load FD001 dataset
    train_df, test_df, rul_df = load_dataset('FD001')
    
    # Preprocess data
    preprocessed = preprocess_data(train_df, test_df, rul_df)
    
    print("\nPreprocessed Data Shapes:")
    print(f"  Train: {preprocessed['train'].shape}")
    print(f"  Validation: {preprocessed['validation'].shape}")
    print(f"  Test: {preprocessed['test'].shape}")
    print(f"  Test RUL: {preprocessed['test_rul'].shape}")
    
    print(f"\nNumber of features: {len(preprocessed['feature_columns'])}")
    print(f"Feature columns: {preprocessed['feature_columns'][:5]}... (showing first 5)")
    
    print("\nTrain Data Sample (with RUL):")
    print(preprocessed['train'][['unit_id', 'time_cycles', 'RUL', 'sensor_2', 'sensor_3']].head())
    
    print("\n" + "="*60)
    print("Preprocessor test complete!")
    print("="*60)

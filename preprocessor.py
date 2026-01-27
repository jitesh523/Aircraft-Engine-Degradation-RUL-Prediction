"""
Data Preprocessor for NASA C-MAPSS Dataset
Handles RUL label creation, normalization, and train/val splitting
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

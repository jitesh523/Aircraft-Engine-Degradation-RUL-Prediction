"""
Unit tests for preprocessor module
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessor import Preprocessor


class TestPreprocessor:
    """Test suite for Preprocessor class"""
    
    @pytest.fixture
    def preprocessor(self):
        """Create Preprocessor instance"""
        return Preprocessor()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        np.random.seed(42)
        return pd.DataFrame({
            'unit_id': np.repeat([1, 2, 3], 10),
            'time_cycles': np.tile(np.arange(1, 11), 3),
            'sensor_1': np.random.randn(30) * 10 + 100,
            'sensor_2': np.random.randn(30) * 5 + 50,
            'sensor_3': np.random.randn(30) * 2 + 20,
            'RUL': np.tile(np.arange(9, -1, -1), 3)
        })
    
    def test_initialization(self, preprocessor):
        """Test Preprocessor initialization"""
        assert preprocessor is not None
        assert hasattr(preprocessor, 'scaler')
        assert preprocessor.scaler is None  # Not fitted yet
    
    def test_normalize_features(self, preprocessor, sample_data):
        """Test feature normalization"""
        feature_cols = ['sensor_1', 'sensor_2', 'sensor_3']
        normalized = preprocessor.normalize_features(sample_data, feature_cols)
        
        # Check that scaler was fitted
        assert preprocessor.scaler is not None
        
        # Check normalized values are in [0, 1] range
        for col in feature_cols:
            assert normalized[col].min() >= 0
            assert normalized[col].max() <= 1
    
    def test_normalize_preserves_columns(self, preprocessor, sample_data):
        """Test that normalization preserves all columns"""
        feature_cols = ['sensor_1', 'sensor_2', 'sensor_3']
        original_cols = set(sample_data.columns)
        normalized = preprocessor.normalize_features(sample_data, feature_cols)
        
        assert set(normalized.columns) == original_cols
    
    def test_train_validation_split(self, preprocessor, sample_data):
        """Test train-validation split"""
        train, val = preprocessor.split_train_validation(sample_data, val_ratio=0.2)
        
        # Check split ratio
        total_engines = sample_data['unit_id'].nunique()
        val_engines = val['unit_id'].nunique()
        
        assert val_engines > 0
        assert val_engines < total_engines
    
    def test_split_maintains_temporal_order(self, preprocessor, sample_data):
        """Test that split maintains temporal order within engines"""
        train, val = preprocessor.split_train_validation(sample_data, val_ratio=0.3)
        
        # Check that time cycles are monotonic for each engine
        for unit_id in train['unit_id'].unique():
            unit_data = train[train['unit_id'] == unit_id]
            time_cycles = unit_data['time_cycles'].values
            assert np.all(time_cycles[:-1] <= time_cycles[1:])
    
    def test_create_sequences(self, preprocessor):
        """Test sequence creation for LSTM"""
        # Create simple sequential data
        data = pd.DataFrame({
            'feature_1': np.arange(20),
            'feature_2': np.arange(20) * 2
        })
        rul = np.arange(19, -1, -1)
        
        X, y = preprocessor.create_sequences(data.values, rul, sequence_length=5)
        
        # Check shapes
        assert X.shape[0] == len(data) - 5 + 1  # 16 sequences
        assert X.shape[1] == 5  # sequence length
        assert X.shape[2] == 2  # number of features
        assert y.shape[0] == X.shape[0]
    
    def test_sequences_correct_values(self, preprocessor):
        """Test that sequences contain correct values"""
        data = pd.DataFrame({
            'feature_1': np.arange(10)
        })
        rul = np.arange(9, -1, -1)
        
        X, y = preprocessor.create_sequences(data.values, rul, sequence_length=3)
        
        # First sequence should be [0, 1, 2]
        assert np.array_equal(X[0, :, 0], np.array([0, 1, 2]))
        # Second sequence should be [1, 2, 3]
        assert np.array_equal(X[1, :, 0], np.array([1, 2, 3]))
        
        # RUL should correspond to last time step of each sequence
        assert y[0] == 7  # RUL at time step 2
        assert y[1] == 6  # RUL at time step 3
    
    def test_remove_constant_features(self, preprocessor):
        """Test removal of constant features"""
        data = pd.DataFrame({
            'var_feature': np.random.randn(100),
            'const_feature_1': np.ones(100),
            'const_feature_2': np.zeros(100),
            'var_feature_2': np.random.randn(100)
        })
        
        cleaned = preprocessor.remove_constant_features(data)
        
        # Constant features should be removed
        assert 'const_feature_1' not in cleaned.columns
        assert 'const_feature_2' not in cleaned.columns
        # Variable features should remain
        assert 'var_feature' in cleaned.columns
        assert 'var_feature_2' in cleaned.columns
    
    def test_handle_missing_values(self, preprocessor, sample_data):
        """Test missing value handling"""
        # Introduce missing values
        data_with_nan = sample_data.copy()
        data_with_nan.loc[5, 'sensor_1'] = np.nan
        data_with_nan.loc[10, 'sensor_2'] = np.nan
        
        # Should handle missing values (forward fill or interpolate)
        if hasattr(preprocessor, 'handle_missing_values'):
            cleaned = preprocessor.handle_missing_values(data_with_nan)
            assert not cleaned.isnull().any().any()
    
    def test_normalize_new_data(self, preprocessor, sample_data):
        """Test normalizing new data with fitted scaler"""
        feature_cols = ['sensor_1', 'sensor_2', 'sensor_3']
        
        # Fit on training data
        _ = preprocessor.normalize_features(sample_data, feature_cols)
        
        # Create new test data
        test_data = pd.DataFrame({
            'unit_id': [4, 4],
            'time_cycles': [1, 2],
            'sensor_1': [105, 110],
            'sensor_2': [52, 53],
            'sensor_3': [21, 22],
            'RUL': [10, 9]
        })
        
        # Normalize test data (should use existing scaler)
        normalized_test = preprocessor.normalize_features(test_data, feature_cols, fit=False)
        
        # Check values are normalized
        for col in feature_cols:
            assert normalized_test[col].min() >= -0.5  # Allow some out-of-range
            assert normalized_test[col].max() <= 1.5


class TestPreprocessorEdgeCases:
    """Test edge cases for Preprocessor"""
    
    @pytest.fixture
    def preprocessor(self):
        return Preprocessor()
    
    def test_empty_data(self, preprocessor):
        """Test with empty dataset"""
        empty_df = pd.DataFrame()
        
        # Should handle gracefully
        with pytest.raises((ValueError, KeyError)):
            preprocessor.normalize_features(empty_df, [])
    
    def test_single_sample_sequence(self, preprocessor):
        """Test sequence creation with too few samples"""
        data = pd.DataFrame({
            'feature_1': [1, 2]
        })
        rul = np.array([1, 0])
        
        # Should handle sequences larger than data
        X, y = preprocessor.create_sequences(data.values, rul, sequence_length=5)
        assert len(X) == 0  # No sequences can be created
    
    def test_sequence_length_one(self, preprocessor):
        """Test sequence creation with length 1"""
        data = pd.DataFrame({
            'feature_1': np.arange(10)
        })
        rul = np.arange(9, -1, -1)
        
        X, y = preprocessor.create_sequences(data.values, rul, sequence_length=1)
        
        assert X.shape[0] == 10
        assert X.shape[1] == 1
        assert y.shape[0] == 10


if __name__ == "__main__":
    pytest.main([__file__, '-v'])

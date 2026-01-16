"""
Unit tests for feature_engineer module
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from feature_engineer import FeatureEngineer


class TestFeatureEngineer:
    """Test suite for FeatureEngineer class"""
    
    @pytest.fixture
    def feature_engineer(self):
        """Create FeatureEngineer instance"""
        return FeatureEngineer()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample sensor data"""
        np.random.seed(42)
        n_samples = 50
        return pd.DataFrame({
            'unit_id': np.repeat([1, 2], 25),
            'time_cycles': np.tile(np.arange(1, 26), 2),
            'sensor_1': np.random.randn(n_samples) * 10 + 100,
            'sensor_2': np.random.randn(n_samples) * 5 + 50,
            'sensor_3': np.random.randn(n_samples) * 2 + 20,
            'sensor_4': np.random.randn(n_samples) * 3 + 30,
        })
    
    def test_initialization(self, feature_engineer):
        """Test FeatureEngineer initialization"""
        assert feature_engineer is not None
        assert hasattr(feature_engineer, 'create_rolling_features')
        assert hasattr(feature_engineer, 'create_health_indicators')
    
    def test_rolling_mean(self, feature_engineer, sample_data):
        """Test rolling mean calculation"""
        sensor_cols = ['sensor_1', 'sensor_2']
        window_size = 5
        
        result = feature_engineer.create_rolling_features(
            sample_data, 
            sensor_cols, 
            window_sizes=[window_size],
            stats=['mean']
        )
        
        # Check new columns were created
        for col in sensor_cols:
            assert f'{col}_rolling_mean_{window_size}' in result.columns
    
    def test_rolling_statistics(self, feature_engineer, sample_data):
        """Test multiple rolling statistics"""
        sensor_cols = ['sensor_1']
        window_size = 5
        stats = ['mean', 'std', 'min', 'max']
        
        result = feature_engineer.create_rolling_features(
            sample_data,
            sensor_cols,
            window_sizes=[window_size],
            stats=stats
        )
        
        # All statistics should be calculated
        for stat in stats:
            col_name = f'sensor_1_rolling_{stat}_{window_size}'
            assert col_name in result.columns
    
    def test_rolling_multiple_windows(self, feature_engineer, sample_data):
        """Test rolling features with multiple window sizes"""
        sensor_cols = ['sensor_1']
        windows = [5, 10, 15]
        
        result = feature_engineer.create_rolling_features(
            sample_data,
            sensor_cols,
            window_sizes=windows,
            stats=['mean']
        )
        
        # Features for all windows should be created
        for window in windows:
            assert f'sensor_1_rolling_mean_{window}' in result.columns
    
    def test_rolling_groupby_engine(self, feature_engineer, sample_data):
        """Test that rolling features are calculated per engine"""
        result = feature_engineer.create_rolling_features(
            sample_data,
            ['sensor_1'],
            window_sizes=[5],
            stats=['mean']
        )
        
        # Check that engines are processed separately
        unit_1 = result[result['unit_id'] == 1]['sensor_1_rolling_mean_5']
        unit_2 = result[result['unit_id'] == 2]['sensor_1_rolling_mean_5']
        
        # First few values should be NaN due to insufficient window
        assert pd.isna(unit_1.iloc[0])
        assert pd.isna(unit_2.iloc[0])
    
    def test_rate_of_change(self, feature_engineer, sample_data):
        """Test rate of change calculation"""
        sensor_cols = ['sensor_1', 'sensor_2']
        
        result = feature_engineer.create_rate_of_change_features(
            sample_data,
            sensor_cols
        )
        
        # Check new columns
        for col in sensor_cols:
            assert f'{col}_roc' in result.columns
    
    def test_roc_values(self, feature_engineer):
        """Test that rate of change values are correct"""
        # Create simple increasing data
        data = pd.DataFrame({
            'unit_id': [1, 1, 1, 1, 1],
            'time_cycles': [1, 2, 3, 4, 5],
            'sensor_1': [100, 102, 104, 106, 108]  # Increases by 2 each step
        })
        
        result = feature_engineer.create_rate_of_change_features(data, ['sensor_1'])
        
        # First value should be NaN, rest should be 2
        assert pd.isna(result['sensor_1_roc'].iloc[0])
        assert np.allclose(result['sensor_1_roc'].iloc[1:].values, [2, 2, 2, 2])
    
    def test_health_indicators(self, feature_engineer):
        """Test health indicator creation"""
        # Create sample data with typical sensor columns
        data = pd.DataFrame({
            'unit_id': [1] * 10,
            'time_cycles': np.arange(1, 11),
            'sensor_11': np.random.randn(10) * 10 + 100,  # T2
            'sensor_13': np.random.randn(10) * 5 + 50,    # T24
            'sensor_4': np.random.randn(10) * 2 + 20,     # Nf
            'sensor_7': np.random.randn(10) * 1 + 10,     # P2
            'sensor_8': np.random.randn(10) * 2 + 15,     # P15
            'sensor_16': np.random.randn(10) * 0.5 + 5,   # W21
        })
        
        result = feature_engineer.create_health_indicators(data)
        
        # Check that health indicators were created
        assert 'temp_ratio' in result.columns or 'T24_T2_ratio' in result.columns
    
    def test_feature_engineering_pipeline(self, feature_engineer, sample_data):
        """Test complete feature engineering pipeline"""
        sensor_cols = ['sensor_1', 'sensor_2', 'sensor_3', 'sensor_4']
        
        # Apply multiple transformations
        result = feature_engineer.create_rolling_features(
            sample_data,
            sensor_cols,
            window_sizes=[5],
            stats=['mean', 'std']
        )
        
        result = feature_engineer.create_rate_of_change_features(
            result,
            sensor_cols
        )
        
        # Original columns should still exist
        for col in sensor_cols:
            assert col in result.columns
        
        # New features should be added
        assert 'sensor_1_rolling_mean_5' in result.columns
        assert 'sensor_1_roc' in result.columns
    
    def test_no_data_leakage(self, feature_engineer, sample_data):
        """Test that rolling features don't leak future information"""
        result = feature_engineer.create_rolling_features(
            sample_data,
            ['sensor_1'],
            window_sizes=[5],
            stats=['mean']
        )
        
        # At index 10 (0-indexed), rolling mean should only use indices 6-10
        # Not indices in the future
        unit_1_data = result[result['unit_id'] == 1]
        
        # The rolling mean at position 5 should use data from 1-5
        # We can verify this doesn't use future data by checking it's computed correctly
        if len(unit_1_data) >= 5:
            manual_mean = sample_data[sample_data['unit_id'] == 1]['sensor_1'].iloc[0:5].mean()
            computed_mean = unit_1_data['sensor_1_rolling_mean_5'].iloc[4]
            
            if not pd.isna(computed_mean):
                assert np.isclose(manual_mean, computed_mean, rtol=0.01)


class TestFeatureEngineerEdgeCases:
    """Test edge cases for FeatureEngineer"""
    
    @pytest.fixture
    def feature_engineer(self):
        return FeatureEngineer()
    
    def test_empty_sensor_list(self, feature_engineer):
        """Test with empty sensor list"""
        data = pd.DataFrame({
            'unit_id': [1, 1],
            'time_cycles': [1, 2],
            'sensor_1': [100, 101]
        })
        
        result = feature_engineer.create_rolling_features(data, [], [5], ['mean'])
        
        # Should return original data
        assert len(result.columns) == len(data.columns)
    
    def test_window_larger_than_data(self, feature_engineer):
        """Test window size larger than available data"""
        data = pd.DataFrame({
            'unit_id': [1, 1, 1],
            'time_cycles': [1, 2, 3],
            'sensor_1': [100, 101, 102]
        })
        
        result = feature_engineer.create_rolling_features(
            data,
            ['sensor_1'],
            window_sizes=[10],  # Larger than data
            stats=['mean']
        )
        
        # All rolling values should be NaN
        assert result['sensor_1_rolling_mean_10'].isna().all()


if __name__ == "__main__":
    pytest.main([__file__, '-v'])

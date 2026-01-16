"""
Unit tests for data_loader module
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_loader import DataLoader
import config


class TestDataLoader:
    """Test suite for DataLoader class"""
    
    @pytest.fixture
    def data_loader(self):
        """Create DataLoader instance"""
        return DataLoader()
    
    def test_initialization(self, data_loader):
        """Test DataLoader initialization"""
        assert data_loader is not None
        assert hasattr(data_loader, 'column_names')
        assert len(data_loader.column_names) == 26  # FD001 has 26 columns
    
    def test_column_names(self, data_loader):
        """Test column names are correctly defined"""
        expected_cols = ['unit_id', 'time_cycles']
        assert data_loader.column_names[0] == expected_cols[0]
        assert data_loader.column_names[1] == expected_cols[1]
        assert 'sensor_1' in data_loader.column_names
        assert 'sensor_21' in data_loader.column_names
    
    def test_add_rul_to_train(self, data_loader):
        """Test RUL calculation for training data"""
        # Create sample training data
        sample_data = pd.DataFrame({
            'unit_id': [1, 1, 1, 2, 2],
            'time_cycles': [1, 2, 3, 1, 2],
            'sensor_1': [100, 101, 102, 100, 101]
        })
        
        result = data_loader.add_rul_to_train(sample_data)
        
        # Check RUL column exists
        assert 'RUL' in result.columns
        
        # Check RUL values are correct (descending from max cycles)
        unit_1_rul = result[result['unit_id'] == 1]['RUL'].values
        assert list(unit_1_rul) == [2, 1, 0]
        
        unit_2_rul = result[result['unit_id'] == 2]['RUL'].values
        assert list(unit_2_rul) == [1, 0]
    
    def test_add_rul_with_empty_data(self, data_loader):
        """Test RUL calculation with empty data"""
        empty_df = pd.DataFrame(columns=['unit_id', 'time_cycles'])
        result = data_loader.add_rul_to_train(empty_df)
        assert 'RUL' in result.columns
        assert len(result) == 0
    
    def test_add_rul_single_engine(self, data_loader):
        """Test RUL calculation for single engine"""
        single_engine = pd.DataFrame({
            'unit_id': [1, 1, 1, 1],
            'time_cycles': [1, 2, 3, 4],
            'sensor_1': [100, 101, 102, 103]
        })
        
        result = data_loader.add_rul_to_train(single_engine)
        expected_rul = [3, 2, 1, 0]
        assert list(result['RUL'].values) == expected_rul
    
    def test_data_integrity(self, data_loader):
        """Test data integrity checks"""
        # Create data with missing values
        data_with_missing = pd.DataFrame({
            'unit_id': [1, 1, 1],
            'time_cycles': [1, 2, 3],
            'sensor_1': [100, np.nan, 102]
        })
        
        # Should handle missing values gracefully
        result = data_loader.add_rul_to_train(data_with_missing)
        assert 'RUL' in result.columns
    
    def test_rul_non_negative(self, data_loader):
        """Test that RUL values are never negative"""
        sample_data = pd.DataFrame({
            'unit_id': [1, 1, 2, 2, 2],
            'time_cycles': [1, 2, 1, 2, 3],
            'sensor_1': [100, 101, 100, 101, 102]
        })
        
        result = data_loader.add_rul_to_train(sample_data)
        assert (result['RUL'] >= 0).all()
    
    def test_rul_decreasing(self, data_loader):
        """Test that RUL decreases over time for each engine"""
        sample_data = pd.DataFrame({
            'unit_id': [1, 1, 1, 1, 1],
            'time_cycles': [1, 2, 3, 4, 5],
            'sensor_1': [100, 101, 102, 103, 104]
        })
        
        result = data_loader.add_rul_to_train(sample_data)
        rul_values = result['RUL'].values
        
        # Check that RUL is strictly decreasing
        for i in range(len(rul_values) - 1):
            assert rul_values[i] > rul_values[i + 1]


class TestDataLoaderIntegration:
    """Integration tests for DataLoader with actual data files (if available)"""
    
    @pytest.fixture
    def data_loader(self):
        """Create DataLoader instance"""
        return DataLoader()
    
    def test_load_train_data_if_exists(self, data_loader):
        """Test loading actual training data if file exists"""
        train_file = os.path.join(config.DATA_DIR, 'train_FD001.txt')
        
        if os.path.exists(train_file):
            train_data = data_loader.load_train_data('FD001')
            
            # Check data was loaded
            assert train_data is not None
            assert len(train_data) > 0
            
            # Check RUL column exists
            assert 'RUL' in train_data.columns
            
            # Check data types
            assert train_data['unit_id'].dtype in [np.int32, np.int64]
            assert train_data['time_cycles'].dtype in [np.int32, np.int64]
            assert train_data['RUL'].dtype in [np.int32, np.int64, np.float32, np.float64]
        else:
            pytest.skip("Training data file not found")
    
    def test_load_test_data_if_exists(self, data_loader):
        """Test loading actual test data if file exists"""
        test_file = os.path.join(config.DATA_DIR, 'test_FD001.txt')
        
        if os.path.exists(test_file):
            test_data, _ = data_loader.load_test_data('FD001')
            
            # Check data was loaded
            assert test_data is not None
            assert len(test_data) > 0
            
            # Check columns
            assert 'unit_id' in test_data.columns
            assert 'time_cycles' in test_data.columns
        else:
            pytest.skip("Test data file not found")


if __name__ == "__main__":
    pytest.main([__file__, '-v'])

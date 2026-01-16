"""
Integration tests for end-to-end pipeline
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_loader import DataLoader
from preprocessor import Preprocessor
from feature_engineer import FeatureEngineer
from models.baseline_model import BaselineModel
from models.lstm_model import LSTMModel


class TestEndToEndPipeline:
    """Integration tests for complete training and prediction pipeline"""
    
    @pytest.fixture
    def mock_data(self):
        """Create mock engine degradation data"""
        np.random.seed(42)
        
        # Simulate 5 engines with varying lifespans
        data_frames = []
        for unit_id in range(1, 6):
            n_cycles = np.random.randint(150, 250)
            
            # Simulate degrading sensors
            time_cycles = np.arange(1, n_cycles + 1)
            degradation = time_cycles / n_cycles  # 0 to 1
            
            df = pd.DataFrame({
                'unit_id': unit_id,
                'time_cycles': time_cycles,
                'sensor_1': 100 + degradation * 20 + np.random.randn(n_cycles) * 2,
                'sensor_2': 50 + degradation * 10 + np.random.randn(n_cycles) * 1,
                'sensor_3': 20 + degradation * 5 + np.random.randn(n_cycles) * 0.5,
                'sensor_4': 30 + degradation * 8 + np.random.randn(n_cycles) * 1,
                'sensor_5': 10 + degradation * 3 + np.random.randn(n_cycles) * 0.3,
            })
            data_frames.append(df)
        
        return pd.concat(data_frames, ignore_index=True)
    
    def test_data_loading_and_rul_calculation(self, mock_data):
        """Test data loading and RUL calculation"""
        loader = DataLoader()
        
        # Add RUL
        data_with_rul = loader.add_rul_to_train(mock_data)
        
        # Verify RUL exists and is correct
        assert 'RUL' in data_with_rul.columns
        
        # For each engine, RUL should start at max and decrease to 0
        for unit_id in data_with_rul['unit_id'].unique():
            unit_data = data_with_rul[data_with_rul['unit_id'] == unit_id]
            rul_values = unit_data['RUL'].values
            
            # Should be monotonically decreasing
            assert np.all(rul_values[:-1] >= rul_values[1:])
            # Should end at 0
            assert rul_values[-1] == 0
    
    def test_preprocessing_pipeline(self, mock_data):
        """Test complete preprocessing pipeline"""
        loader = DataLoader()
        preprocessor = Preprocessor()
        
        # Add RUL
        data = loader.add_rul_to_train(mock_data)
        
        # Normalize
        sensor_cols = ['sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5']
        normalized = preprocessor.normalize_features(data, sensor_cols)
        
        # Verify normalization
        for col in sensor_cols:
            assert normalized[col].min() >= 0
            assert normalized[col].max() <= 1
        
        # Split train/val
        train, val = preprocessor.split_train_validation(normalized, val_ratio=0.2)
        
        # Verify split
        assert len(train) > 0
        assert len(val) > 0
        assert len(train) + len(val) == len(normalized)
    
    def test_feature_engineering_pipeline(self, mock_data):
        """Test feature engineering pipeline"""
        loader = DataLoader()
        feature_engineer = FeatureEngineer()
        
        data = loader.add_rul_to_train(mock_data)
        
        sensor_cols = ['sensor_1', 'sensor_2', 'sensor_3']
        
        # Add rolling features
        data_fe = feature_engineer.create_rolling_features(
            data,
            sensor_cols,
            window_sizes=[5, 10],
            stats=['mean', 'std']
        )
        
        # Add rate of change
        data_fe = feature_engineer.create_rate_of_change_features(data_fe, sensor_cols)
        
        # Verify new features exist
        assert 'sensor_1_rolling_mean_5' in data_fe.columns
        assert 'sensor_1_rolling_std_10' in data_fe.columns
        assert 'sensor_1_roc' in data_fe.columns
    
    def test_baseline_model_training_pipeline(self, mock_data):
        """Test complete baseline model training pipeline"""
        loader = DataLoader()
        preprocessor = Preprocessor()
        feature_engineer = FeatureEngineer()
        
        # Prepare data
        data = loader.add_rul_to_train(mock_data)
        sensor_cols = ['sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5']
        
        # Feature engineering
        data_fe = feature_engineer.create_rolling_features(
            data,
            sensor_cols,
            window_sizes=[5],
            stats=['mean']
        )
        
        # Fill NaNs from rolling windows
        data_fe = data_fe.fillna(method='bfill')
        
        # Normalize
        all_feature_cols = [col for col in data_fe.columns 
                           if col not in ['unit_id', 'time_cycles', 'RUL']]
        data_normalized = preprocessor.normalize_features(data_fe, all_feature_cols)
        
        # Prepare X, y
        X = data_normalized[all_feature_cols].values
        y = data_normalized['RUL'].values
        
        # Train model
        model = BaselineModel('random_forest')
        model.train(X, y)
        
        # Make predictions
        predictions = model.predict(X[:10])
        
        assert len(predictions) == 10
        assert np.all(predictions >= 0)
    
    def test_lstm_training_pipeline(self, mock_data):
        """Test complete LSTM training pipeline"""
        loader = DataLoader()
        preprocessor = Preprocessor()
        
        # Prepare data
        data = loader.add_rul_to_train(mock_data)
        sensor_cols = ['sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5']
        
        # Normalize
        data_normalized = preprocessor.normalize_features(data, sensor_cols)
        
        # Get one engine for testing
        unit_1 = data_normalized[data_normalized['unit_id'] == 1]
        X_flat = unit_1[sensor_cols].values
        y_flat = unit_1['RUL'].values
        
        # Create sequences
        sequence_length = 10
        X_seq, y_seq = preprocessor.create_sequences(X_flat, y_flat, sequence_length)
        
        # Verify sequences
        assert X_seq.shape[1] == sequence_length
        assert X_seq.shape[2] == len(sensor_cols)
        
        # Train LSTM
        if len(X_seq) > 0:
            model = LSTMModel(
                sequence_length=sequence_length,
                n_features=len(sensor_cols),
                lstm_units=[25]
            )
            model.build_model()
            model.train(X_seq, y_seq, epochs=2, verbose=0)
            
            # Make predictions
            predictions = model.predict(X_seq[:5])
            
            assert len(predictions) == 5
    
    def test_full_train_predict_workflow(self, mock_data):
        """Test complete workflow from raw data to predictions"""
        # Initialize components
        loader = DataLoader()
        preprocessor = Preprocessor()
        feature_engineer = FeatureEngineer()
        
        # Split into train and test engines
        all_units = mock_data['unit_id'].unique()
        train_units = all_units[:4]
        test_units = all_units[4:]
        
        train_data = mock_data[mock_data['unit_id'].isin(train_units)].copy()
        test_data = mock_data[mock_data['unit_id'].isin(test_units)].copy()
        
        # Process training data
        train_data = loader.add_rul_to_train(train_data)
        sensor_cols = ['sensor_1', 'sensor_2', 'sensor_3']
        
        train_data = feature_engineer.create_rolling_features(
            train_data,
            sensor_cols,
            window_sizes=[5],
            stats=['mean']
        )
        train_data = train_data.fillna(method='bfill')
        
        feature_cols = [col for col in train_data.columns 
                       if col not in ['unit_id', 'time_cycles', 'RUL']]
        
        train_normalized = preprocessor.normalize_features(train_data, feature_cols)
        
        X_train = train_normalized[feature_cols].values
        y_train = train_normalized['RUL'].values
        
        # Train model
        model = BaselineModel('random_forest')
        model.train(X_train, y_train)
        
        # Process test data
        test_data = loader.add_rul_to_train(test_data)
        test_data = feature_engineer.create_rolling_features(
            test_data,
            sensor_cols,
            window_sizes=[5],
            stats=['mean']
        )
        test_data = test_data.fillna(method='bfill')
        
        test_normalized = preprocessor.normalize_features(test_data, feature_cols, fit=False)
        
        X_test = test_normalized[feature_cols].values
        y_test = test_normalized['RUL'].values
        
        # Make predictions
        predictions = model.predict(X_test)
        
        # Validate predictions
        assert len(predictions) == len(y_test)
        assert np.all(predictions >= 0)
        
        # Calculate error (informational)
        mae = np.mean(np.abs(predictions - y_test))
        print(f"\nMean Absolute Error on test set: {mae:.2f} cycles")


if __name__ == "__main__":
    pytest.main([__file__, '-v', '-s'])

"""
Unit tests for model modules
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.baseline_model import BaselineModel
from models.lstm_model import LSTMModel


class TestBaselineModel:
    """Test suite for BaselineModel"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data"""
        np.random.seed(42)
        X = np.random.randn(100, 10)  # 100 samples, 10 features
        y = np.random.randint(0, 200, 100)  # RUL values
        return X, y
    
    def test_random_forest_initialization(self):
        """Test Random Forest model initialization"""
        model = BaselineModel(model_type='random_forest')
        assert model is not None
        assert model.model_type == 'random_forest'
    
    def test_linear_regression_initialization(self):
        """Test Linear Regression model initialization"""
        model = BaselineModel(model_type='linear_regression')
        assert model is not None
        assert model.model_type == 'linear_regression'
    
    def test_train_random_forest(self, sample_data):
        """Test training Random Forest"""
        X, y = sample_data
        model = BaselineModel(model_type='random_forest')
        model.train(X, y)
        
        assert model.model is not None
    
    def test_train_linear_regression(self, sample_data):
        """Test training Linear Regression"""
        X, y = sample_data
        model = BaselineModel(model_type='linear_regression')
        model.train(X, y)
        
        assert model.model is not None
    
    def test_predict_shape(self, sample_data):
        """Test prediction output shape"""
        X_train, y_train = sample_data
        X_test = np.random.randn(20, 10)
        
        model = BaselineModel(model_type='random_forest')
        model.train(X_train, y_train)
        predictions = model.predict(X_test)
        
        assert predictions.shape == (20,)
    
    def test_predictions_non_negative(self, sample_data):
        """Test that predictions are non-negative"""
        X_train, y_train = sample_data
        X_test = np.random.randn(20, 10)
        
        model = BaselineModel(model_type='random_forest')
        model.train(X_train, y_train)
        predictions = model.predict(X_test)
        
        # RUL should not be negative
        assert np.all(predictions >= 0)


class TestLSTMModel:
    """Test suite for LSTMModel"""
    
    @pytest.fixture
    def sample_sequences(self):
        """Create sample sequence data for LSTM"""
        np.random.seed(42)
        # Shape: (samples, sequence_length, features)
        X = np.random.randn(100, 30, 10)
        y = np.random.randint(0, 200, 100)
        return X, y
    
    def test_initialization(self):
        """Test LSTM model initialization"""
        model = LSTMModel(
            sequence_length=30,
            n_features=10,
            lstm_units=[50, 25]
        )
        assert model is not None
    
    def test_build_model(self):
        """Test model building"""
        model = LSTMModel(
            sequence_length=30,
            n_features=10,
            lstm_units=[50, 25]
        )
        model.build_model()
        
        assert model.model is not None
        
        # Check input shape
        input_shape = model.model.input_shape
        assert input_shape[1] == 30  # sequence_length
        assert input_shape[2] == 10  # n_features
    
    def test_train(self, sample_sequences):
        """Test LSTM training"""
        X, y = sample_sequences
        
        model = LSTMModel(
            sequence_length=30,
            n_features=10,
            lstm_units=[25]  # Smaller for faster testing
        )
        model.build_model()
        
        # Train for just 2 epochs to test
        history = model.train(
            X, y,
            validation_split=0.2,
            epochs=2,
            batch_size=32,
            verbose=0
        )
        
        assert history is not None
        assert 'loss' in history.history
    
    def test_predict_shape(self, sample_sequences):
        """Test prediction output shape"""
        X_train, y_train = sample_sequences
        X_test = np.random.randn(20, 30, 10)
        
        model = LSTMModel(
            sequence_length=30,
            n_features=10,
            lstm_units=[25]
        )
        model.build_model()
        model.train(X_train, y_train, epochs=1, verbose=0)
        
        predictions = model.predict(X_test)
        
        assert predictions.shape == (20,)
    
    def test_output_activation(self):
        """Test that output layer has no activation (linear)"""
        model = LSTMModel(
            sequence_length=30,
            n_features=10,
            lstm_units=[25]
        )
        model.build_model()
        
        # Output layer should be linear (no activation)
        output_layer = model.model.layers[-1]
        assert output_layer.activation.__name__ == 'linear'
    
    def test_dropout_layers(self):
        """Test that dropout layers are present"""
        model = LSTMModel(
            sequence_length=30,
            n_features=10,
            lstm_units=[50, 25],
            dropout_rate=0.2
        )
        model.build_model()
        
        # Check for dropout in layer names
        layer_names = [layer.name for layer in model.model.layers]
        has_dropout = any('dropout' in name for name in layer_names)
        
        assert has_dropout


class TestModelIntegration:
    """Integration tests for models"""
    
    def test_baseline_vs_lstm_compatibility(self):
        """Test that baseline and LSTM models can work with same data format"""
        np.random.seed(42)
        
        # Prepare data
        X_baseline = np.random.randn(100, 10)
        X_lstm = np.random.randn(100, 30, 10)  # sequences
        y = np.random.randint(0, 200, 100)
        
        # Train baseline
        baseline = BaselineModel('random_forest')
        baseline.train(X_baseline, y)
        baseline_pred = baseline.predict(X_baseline[:10])
        
        # Train LSTM
        lstm = LSTMModel(sequence_length=30, n_features=10, lstm_units=[25])
        lstm.build_model()
        lstm.train(X_lstm, y, epochs=1, verbose=0)
        lstm_pred = lstm.predict(X_lstm[:10])
        
        # Both should produce predictions of same length
        assert len(baseline_pred) == len(lstm_pred)
        
        # Both should produce non-negative predictions
        assert np.all(baseline_pred >= 0)
        assert np.all(lstm_pred >= 0)


class TestModelEdgeCases:
    """Test edge cases for models"""
    
    def test_single_sample_prediction(self):
        """Test prediction with single sample"""
        np.random.seed(42)
        X_train = np.random.randn(50, 10)
        y_train = np.random.randint(0, 200, 50)
        
        model = BaselineModel('random_forest')
        model.train(X_train, y_train)
        
        # Predict single sample
        X_single = np.random.randn(1, 10)
        pred = model.predict(X_single)
        
        assert pred.shape == (1,)
    
    def test_lstm_minimum_sequence_length(self):
        """Test LSTM with minimal sequence length"""
        model = LSTMModel(
            sequence_length=1,  # Minimum
            n_features=5,
            lstm_units=[10]
        )
        model.build_model()
        
        assert model.model is not None
        assert model.model.input_shape[1] == 1


if __name__ == "__main__":
    pytest.main([__file__, '-v'])

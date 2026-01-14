"""
LSTM Neural Network for RUL Prediction
Deep learning model for time-series RUL prediction
"""

import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Tuple
import config
from utils import setup_logging

logger = setup_logging(__name__)


class LSTMModel:
    """
    LSTM neural network for RUL prediction
    """
    
    def __init__(self, 
                 sequence_length: int = None,
                 num_features: int = None,
                 lstm_units: list = None,
                 dropout_rate: float = None,
                 learning_rate: float = None):
        """
        Initialize LSTM model
        
        Args:
            sequence_length: Number of time steps in input sequences
            num_features: Number of features per time step
            lstm_units: List of units for each LSTM layer
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for Adam optimizer
        """
        # Use config defaults if not provided
        self.sequence_length = sequence_length or config.LSTM_CONFIG['sequence_length']
        self.num_features = num_features
        self.lstm_units = lstm_units or config.LSTM_CONFIG['lstm_units']
        self.dropout_rate = dropout_rate or config.LSTM_CONFIG['dropout_rate']
        self.learning_rate = learning_rate or config.LSTM_CONFIG['learning_rate']
        
        self.model = None
        self.history = None
        
        logger.info(f"Initialized LSTM model with sequence_length={self.sequence_length}")
    
    def build_model(self, num_features: int = None) -> None:
        """
        Build the LSTM architecture
        
        Args:
            num_features: Number of features per time step
        """
        if num_features is not None:
            self.num_features = num_features
        
        if self.num_features is None:
            raise ValueError("num_features must be specified")
        
        logger.info(f"Building LSTM model with {len(self.lstm_units)} layers...")
        
        self.model = Sequential()
        
        # First LSTM layer
        self.model.add(LSTM(
            units=self.lstm_units[0],
            input_shape=(self.sequence_length, self.num_features),
            return_sequences=len(self.lstm_units) > 1
        ))
        self.model.add(Dropout(self.dropout_rate))
        
        # Additional LSTM layers
        for i, units in enumerate(self.lstm_units[1:], 1):
            # Return sequences for all but the last LSTM layer
            return_seq = i < len(self.lstm_units) - 1
            self.model.add(LSTM(units=units, return_sequences=return_seq))
            self.model.add(Dropout(self.dropout_rate))
        
        # Output layer (single RUL value)
        self.model.add(Dense(1, activation='linear'))
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        logger.info("LSTM model built successfully")
        logger.info(f"Model parameters: {self.model.count_params():,}")
    
    def train(self, 
              X_train: np.ndarray, 
              y_train: np.ndarray,
              X_val: np.ndarray = None,
              y_val: np.ndarray = None,
              epochs: int = None,
              batch_size: int = None,
              patience: int = None,
              verbose: int = 1) -> None:
        """
        Train the LSTM model
        
        Args:
            X_train: Training sequences, shape (samples, sequence_length, features)
            y_train: Training targets (RUL values)
            X_val: Validation sequences (optional)
            y_val: Validation targets (optional)
            epochs: Number of training epochs
            batch_size: Training batch size
            patience: Early stopping patience
            verbose: Verbosity level
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Use config defaults if not provided
        epochs = epochs or config.LSTM_CONFIG['epochs']
        batch_size = batch_size or config.LSTM_CONFIG['batch_size']
        patience = patience or config.LSTM_CONFIG['patience']
        
        logger.info(f"Training LSTM model for up to {epochs} epochs...")
        logger.info(f"Training samples: {len(X_train)}, Batch size: {batch_size}")
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
            logger.info(f"Validation samples: {len(X_val)}")
        
        # Callbacks
        callback_list = []
        
        # Early stopping
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss' if validation_data else 'loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        callback_list.append(early_stop)
        
        # Model checkpoint
        # checkpoint = callbacks.ModelCheckpoint(
        #     filepath=config.MODELS_DIR + '/lstm_checkpoint.h5',
        #     monitor='val_loss' if validation_data else 'loss',
        #     save_best_only=True,
        #     verbose=0
        # )
        # callback_list.append(checkpoint)
        
        # Train the model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callback_list,
            verbose=verbose
        )
        
        logger.info("Training complete")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make RUL predictions
        
        Args:
            X: Input sequences, shape (samples, sequence_length, features)
            
        Returns:
            Predicted RUL values, shape (samples,)
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        predictions = self.model.predict(X, verbose=0)
        predictions = predictions.flatten()
        
        # Clip negative predictions to 0
        predictions = np.maximum(predictions, 0)
        
        return predictions
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate model performance
        
        Args:
            X_test: Test sequences
            y_test: True RUL values
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating LSTM model...")
        
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Model evaluation (loss and mae from Keras)
        test_loss, test_mae = self.model.evaluate(X_test, y_test, verbose=0)
        
        metrics = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'test_loss': test_loss,
            'test_mae': test_mae
        }
        
        logger.info(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.4f}")
        
        return metrics
    
    def get_training_history(self) -> Dict:
        """
        Get training history
        
        Returns:
            Dictionary with training and validation metrics
        """
        if self.history is None:
            logger.warning("Model not trained yet")
            return None
        
        return {
            'loss': self.history.history['loss'],
            'mae': self.history.history['mae'],
            'val_loss': self.history.history.get('val_loss', []),
            'val_mae': self.history.history.get('val_mae', [])
        }
    
    def save(self, filepath: str) -> None:
        """
        Save the trained model
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        Load a trained model
        
        Args:
            filepath: Path to the saved model
        """
        self.model = keras.models.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")
    
    def summary(self) -> None:
        """Print model summary"""
        if self.model is None:
            print("Model not built yet")
        else:
            self.model.summary()


if __name__ == "__main__":
    # Test LSTM model
    print("="*60)
    print("Testing LSTM Model")
    print("="*60)
    
    # Generate synthetic data for testing
    np.random.seed(42)
    sequence_length = 30
    num_features = 20
    num_samples = 1000
    
    X_train = np.random.randn(num_samples, sequence_length, num_features)
    y_train = np.random.randint(0, 150, size=num_samples).astype(float)
    X_val = np.random.randn(200, sequence_length, num_features)
    y_val = np.random.randint(0, 150, size=200).astype(float)
    X_test = np.random.randn(100, sequence_length, num_features)
    y_test = np.random.randint(0, 150, size=100).astype(float)
    
    # Build and train model
    lstm_model = LSTMModel(sequence_length=sequence_length)
    lstm_model.build_model(num_features=num_features)
    
    print("\nModel Summary:")
    lstm_model.summary()
    
    print("\nTraining model...")
    lstm_model.train(X_train, y_train, X_val, y_val, epochs=10, verbose=0)
    
    # Evaluate
    print("\nEvaluating model...")
    metrics = lstm_model.evaluate(X_test, y_test)
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"R2: {metrics['r2']:.4f}")
    
    # Get training history
    history = lstm_model.get_training_history()
    print(f"\nTraining completed in {len(history['loss'])} epochs")
    print(f"Final training loss: {history['loss'][-1]:.4f}")
    if history['val_loss']:
        print(f"Final validation loss: {history['val_loss'][-1]:.4f}")
    
    print("\n" + "="*60)
    print("LSTM model test complete!")
    print("="*60)

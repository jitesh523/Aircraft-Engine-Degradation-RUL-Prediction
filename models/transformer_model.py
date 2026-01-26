"""
Transformer Neural Network for RUL Prediction
Advanced deep learning model using Multi-Head Attention for time-series RUL prediction
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Tuple, List
import config
from utils import setup_logging

logger = setup_logging(__name__)


class TransformerModel:
    """
    Transformer neural network for RUL prediction
    Uses Multi-Head Attention to capture long-range dependencies in sensor data
    """
    
    def __init__(self, 
                 sequence_length: int = None,
                 num_features: int = None,
                 head_size: int = None,
                 num_heads: int = None,
                 ff_dim: int = None,
                 num_transformer_blocks: int = None,
                 mlp_units: list = None,
                 dropout_rate: float = None,
                 mlp_dropout: float = None):
        """
        Initialize Transformer model
        
        Args:
            sequence_length: Number of time steps in input sequences
            num_features: Number of features per time step
            head_size: Dimensionality of attention heads
            num_heads: Number of attention heads
            ff_dim: Dimensionality of feed-forward network
            num_transformer_blocks: Number of transformer blocks
            mlp_units: List of units for final MLP layers
            dropout_rate: Dropout rate for attention layers
            mlp_dropout: Dropout rate for MLP layers
        """
        # Use config defaults if not provided
        self.sequence_length = sequence_length or config.LSTM_CONFIG['sequence_length']
        self.num_features = num_features
        
        # Transformer specific config
        t_config = config.TRANSFORMER_CONFIG
        self.head_size = head_size or t_config['head_size']
        self.num_heads = num_heads or t_config['num_heads']
        self.ff_dim = ff_dim or t_config['ff_dim']
        self.num_transformer_blocks = num_transformer_blocks or t_config['num_transformer_blocks']
        self.mlp_units = mlp_units or t_config['mlp_units']
        self.dropout_rate = dropout_rate or t_config['dropout_rate']
        self.mlp_dropout = mlp_dropout or t_config['mlp_dropout']
        self.learning_rate = t_config['learning_rate']
        
        self.model = None
        self.history = None
        
        logger.info(f"Initialized Transformer model with {self.num_transformer_blocks} blocks, {self.num_heads} heads")
    
    def transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0):
        """
        Single Transformer Encoder block
        
        Args:
            inputs: Input tensor
            head_size: Dimensionality of attention heads
            num_heads: Number of attention heads
            ff_dim: Dimensionality of feed-forward network
            dropout: Dropout rate
            
        Returns:
            Output tensor
        """
        # Attention and Normalization
        x = LayerNormalization(epsilon=1e-6)(inputs)
        x = MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(x, x)
        x = Dropout(dropout)(x)
        res = x + inputs
        
        # Feed Forward Part
        x = LayerNormalization(epsilon=1e-6)(res)
        x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
        x = Dropout(dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        return x + res
    
    def build_model(self, num_features: int = None) -> None:
        """
        Build the Transformer architecture
        
        Args:
            num_features: Number of features per time step
        """
        if num_features is not None:
            self.num_features = num_features
        
        if self.num_features is None:
            raise ValueError("num_features must be specified")
        
        logger.info("Building Transformer model...")
        
        inputs = Input(shape=(self.sequence_length, self.num_features))
        x = inputs
        
        # Transformer Blocks
        for _ in range(self.num_transformer_blocks):
            x = self.transformer_encoder(
                x, 
                self.head_size, 
                self.num_heads, 
                self.ff_dim, 
                self.dropout_rate
            )
        
        # Pooling and MLP
        x = GlobalAveragePooling1D()(x)
        
        for dim in self.mlp_units:
            x = Dense(dim, activation="relu")(x)
            x = Dropout(self.mlp_dropout)(x)
            
        outputs = Dense(1, activation="linear")(x)
        
        self.model = Model(inputs, outputs)
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        logger.info("Transformer model built successfully")
        # logger.info(f"Model parameters: {self.model.count_params():,}")
    
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
        Train the Transformer model
        
        Args:
            X_train: Training sequences
            y_train: Training targets
            X_val: Validation sequences
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Training batch size
            patience: Early stopping patience
            verbose: Verbosity level
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Use config defaults if not provided
        t_config = config.TRANSFORMER_CONFIG
        epochs = epochs or t_config['epochs']
        batch_size = batch_size or t_config['batch_size']
        patience = patience or t_config['patience']
        
        logger.info(f"Training Transformer model for up to {epochs} epochs...")
        
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        # Callbacks
        callback_list = []
        
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss' if validation_data else 'loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        callback_list.append(early_stop)
        
        # Train
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
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not built")
        
        predictions = self.model.predict(X, verbose=0)
        predictions = predictions.flatten()
        return np.maximum(predictions, 0)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate model performance"""
        logger.info("Evaluating Transformer model...")
        y_pred = self.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        logger.info(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.4f}")
        
        return {'rmse': rmse, 'mae': mae, 'r2': r2}
    
    def save(self, filepath: str) -> None:
        """Save model"""
        if self.model is None:
            raise ValueError("No model to save")
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load model"""
        self.model = keras.models.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")

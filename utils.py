"""
Utility functions for Aircraft Engine RUL Prediction System
Helper functions for data manipulation, sequence generation, and model I/O
"""

import numpy as np
import pandas as pd
import pickle
import json
import logging
from typing import Tuple, List, Dict, Any
import config

# ==================== Logging Setup ====================
def setup_logging(name: str = __name__) -> logging.Logger:
    """
    Set up logging configuration
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=getattr(logging, config.LOGGING_CONFIG['level']),
        format=config.LOGGING_CONFIG['format'],
        datefmt=config.LOGGING_CONFIG['datefmt']
    )
    return logging.getLogger(name)

logger = setup_logging()

# ==================== Sequence Generation ====================
def generate_sequences(data: pd.DataFrame, 
                       sequence_length: int,
                       feature_cols: List[str],
                       target_col: str = 'RUL') -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate time-series sequences for LSTM input
    
    For each engine unit, creates overlapping sequences of length 'sequence_length'
    from the available time cycles. Each sequence has shape (sequence_length, num_features).
    
    Args:
        data: DataFrame with columns [unit_id, time_cycles, features..., RUL]
        sequence_length: Number of time steps in each sequence
        feature_cols: List of feature column names to include
        target_col: Name of target column (RUL)
        
    Returns:
        X: Array of sequences, shape (num_sequences, sequence_length, num_features)
        y: Array of targets, shape (num_sequences,)
    """
    sequences = []
    targets = []
    
    # Group by unit_id to process each engine separately
    for unit_id, unit_data in data.groupby('unit_id'):
        # Sort by time to ensure chronological order
        unit_data = unit_data.sort_values('time_cycles')
        
        # Extract features and target
        features = unit_data[feature_cols].values
        rul_values = unit_data[target_col].values
        
        # Generate sequences
        num_cycles = len(unit_data)
        for i in range(sequence_length, num_cycles + 1):
            # Sequence: from (i - sequence_length) to i
            seq = features[i - sequence_length:i, :]
            target = rul_values[i - 1]  # RUL at the end of sequence
            
            sequences.append(seq)
            targets.append(target)
    
    X = np.array(sequences)
    y = np.array(targets)
    
    logger.info(f"Generated {len(sequences)} sequences with shape {X.shape}")
    return X, y


def generate_sequences_for_prediction(data: pd.DataFrame,
                                      sequence_length: int,
                                      feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate sequences for test data prediction
    
    For each test engine, creates ONE sequence from the last 'sequence_length' cycles
    to predict the final RUL.
    
    Args:
        data: Test DataFrame with columns [unit_id, time_cycles, features...]
        sequence_length: Number of time steps in each sequence
        feature_cols: List of feature column names
        
    Returns:
        X: Array of sequences, shape (num_engines, sequence_length, num_features)
        unit_ids: Array of corresponding unit IDs
    """
    sequences = []
    unit_ids = []
    
    for unit_id, unit_data in data.groupby('unit_id'):
        unit_data = unit_data.sort_values('time_cycles')
        features = unit_data[feature_cols].values
        
        # Take only the last 'sequence_length' cycles
        if len(unit_data) >= sequence_length:
            seq = features[-sequence_length:, :]
        else:
            # Pad with zeros if not enough cycles
            padding = np.zeros((sequence_length - len(features), len(feature_cols)))
            seq = np.vstack([padding, features])
        
        sequences.append(seq)
        unit_ids.append(unit_id)
    
    X = np.array(sequences)
    unit_ids = np.array(unit_ids)
    
    logger.info(f"Generated {len(sequences)} test sequences with shape {X.shape}")
    return X, unit_ids


# ==================== Model I/O ====================
def save_model(model: Any, filepath: str) -> None:
    """
    Save trained model to disk
    
    Args:
        model: Trained model object
        filepath: Path to save the model
    """
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # For Keras/TensorFlow models
    if hasattr(model, 'save'):
        model.save(filepath)
        logger.info(f"Keras model saved to {filepath}")
    # For sklearn models
    else:
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Sklearn model saved to {filepath}")


def load_model(filepath: str, is_keras: bool = True) -> Any:
    """
    Load trained model from disk
    
    Args:
        filepath: Path to the saved model
        is_keras: Whether the model is a Keras model
        
    Returns:
        Loaded model object
    """
    if is_keras:
        from tensorflow import keras
        model = keras.models.load_model(filepath)
        logger.info(f"Keras model loaded from {filepath}")
    else:
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Sklearn model loaded from {filepath}")
    
    return model


def save_scaler(scaler: Any, filepath: str) -> None:
    """
    Save trained scaler to disk
    
    Args:
        scaler: Fitted scaler object
        filepath: Path to save the scaler
    """
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f"Scaler saved to {filepath}")


def load_scaler(filepath: str) -> Any:
    """
    Load trained scaler from disk
    
    Args:
        filepath: Path to the saved scaler
        
    Returns:
        Loaded scaler object
    """
    with open(filepath, 'rb') as f:
        scaler = pickle.load(f)
    logger.info(f"Scaler loaded from {filepath}")
    return scaler


# ==================== Results I/O ====================
def save_results(results: Dict[str, Any], filepath: str) -> None:
    """
    Save evaluation results to JSON file
    
    Args:
        results: Dictionary of results
        filepath: Path to save the results
    """
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Convert numpy types to Python native types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    results_serializable = {k: convert_numpy(v) for k, v in results.items()}
    
    with open(filepath, 'w') as f:
        json.dump(results_serializable, f, indent=4)
    logger.info(f"Results saved to {filepath}")


def load_results(filepath: str) -> Dict[str, Any]:
    """
    Load evaluation results from JSON file
    
    Args:
        filepath: Path to the results file
        
    Returns:
        Dictionary of results
    """
    with open(filepath, 'r') as f:
        results = json.load(f)
    logger.info(f"Results loaded from {filepath}")
    return results


# ==================== Data Manipulation ====================
def add_remaining_useful_life(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate and add RUL column to training data
    
    RUL = max_cycle - current_cycle for each engine unit
    
    Args:
        df: DataFrame with unit_id and time_cycles columns
        
    Returns:
        DataFrame with added RUL column
    """
    # Get max cycle for each engine
    max_cycles = df.groupby('unit_id')['time_cycles'].max().reset_index()
    max_cycles.columns = ['unit_id', 'max_cycle']
    
    # Merge and calculate RUL
    df = df.merge(max_cycles, on='unit_id', how='left')
    df['RUL'] = df['max_cycle'] - df['time_cycles']
    df = df.drop('max_cycle', axis=1)
    
    # Apply RUL clipping if configured
    if config.USE_RUL_CLIPPING:
        df['RUL'] = df['RUL'].clip(upper=config.MAX_RUL)
    
    logger.info(f"Added RUL column. Range: {df['RUL'].min()} to {df['RUL'].max()}")
    return df


def calculate_rate_of_change(df: pd.DataFrame, columns: List[str], periods: int = 1) -> pd.DataFrame:
    """
    Calculate rate of change for specified columns
    
    Args:
        df: DataFrame with unit_id and time_cycles
        columns: List of column names to calculate rate of change
        periods: Number of periods to shift for rate calculation
        
    Returns:
        DataFrame with additional rate-of-change columns
    """
    for col in columns:
        rate_col = f'{col}_roc_{periods}'
        df[rate_col] = df.groupby('unit_id')[col].diff(periods)
        # Fill NaN values with 0 (first few cycles)
        df[rate_col] = df[rate_col].fillna(0)
    
    logger.info(f"Calculated rate of change for {len(columns)} columns")
    return df


def add_rolling_statistics(df: pd.DataFrame, 
                          columns: List[str], 
                          window_sizes: List[int],
                          stats: List[str] = ['mean', 'std']) -> pd.DataFrame:
    """
    Add rolling window statistics for specified columns
    
    Args:
        df: DataFrame with unit_id and time_cycles
        columns: List of column names to calculate statistics
        window_sizes: List of window sizes for rolling statistics
        stats: List of statistics to calculate ('mean', 'std', 'min', 'max')
        
    Returns:
        DataFrame with additional rolling statistic columns
    """
    for col in columns:
        for window in window_sizes:
            for stat in stats:
                new_col = f'{col}_rolling_{stat}_{window}'
                if stat == 'mean':
                    df[new_col] = df.groupby('unit_id')[col].rolling(window=window, min_periods=1).mean().reset_index(0, drop=True)
                elif stat == 'std':
                    df[new_col] = df.groupby('unit_id')[col].rolling(window=window, min_periods=1).std().reset_index(0, drop=True)
                elif stat == 'min':
                    df[new_col] = df.groupby('unit_id')[col].rolling(window=window, min_periods=1).min().reset_index(0, drop=True)
                elif stat == 'max':
                    df[new_col] = df.groupby('unit_id')[col].rolling(window=window, min_periods=1).max().reset_index(0, drop=True)
                # Fill NaN with original value
                df[new_col] = df[new_col].fillna(df[col])
    
    logger.info(f"Added rolling statistics for {len(columns)} columns with {len(window_sizes)} window sizes")
    return df


# ==================== Evaluation Helpers ====================
def asymmetric_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate asymmetric scoring function (NASA scoring metric)
    
    Penalizes late predictions (under-predicting RUL) more heavily than early predictions
    
    Args:
        y_true: True RUL values
        y_pred: Predicted RUL values
        
    Returns:
        Asymmetric score (lower is better)
    """
    diff = y_pred - y_true
    score = np.sum(
        np.where(diff < 0, 
                np.exp(-diff / config.ASYMMETRIC_LOSS_WEIGHTS['late_prediction_penalty']) - 1,
                np.exp(diff / config.ASYMMETRIC_LOSS_WEIGHTS['early_prediction_penalty']) - 1)
    )
    return score


if __name__ == "__main__":
    logger.info("Utility functions loaded successfully")
    print("Available utility functions:")
    print("- setup_logging()")
    print("- generate_sequences()")
    print("- generate_sequences_for_prediction()")
    print("- save_model() / load_model()")
    print("- save_scaler() / load_scaler()")
    print("- save_results() / load_results()")
    print("- add_remaining_useful_life()")
    print("- calculate_rate_of_change()")
    print("- add_rolling_statistics()")
    print("- asymmetric_score()")

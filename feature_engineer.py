"""
Feature Engineering for NASA C-MAPSS Dataset
Creates rolling statistics, rate-of-change features, and health indicators
"""

import pandas as pd
import numpy as np
from typing import List, Dict
import config
from utils import setup_logging, add_rolling_statistics, calculate_rate_of_change

logger = setup_logging(__name__)


class FeatureEngineer:
    """
    Feature engineering for turbofan engine degradation prediction
    Creates time-series features from sensor data
    """
    
    def __init__(self):
        """Initialize feature engineer"""
        self.engineered_features = []
        logger.info("Initialized Feature Engineer")
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all engineered features
        
        Args:
            df: DataFrame with sensor data
            
        Returns:
            DataFrame with added engineered features
        """
        logger.info("Creating all engineered features...")
        
        # Create a copy
        df_engineered = df.copy()
        
        # Get sensor columns
        sensor_cols = [col for col in df.columns if col.startswith('sensor_')]
        logger.info(f"Engineering features for {len(sensor_cols)} sensors")
        
        # 1. Rolling window statistics
        if config.ROLLING_WINDOW_SIZES:
            df_engineered = self.add_rolling_features(
                df_engineered, 
                sensor_cols,
                config.ROLLING_WINDOW_SIZES,
                config.ROLLING_FEATURES
            )
        
        # 2. Rate of change features
        if config.CREATE_RATE_OF_CHANGE:
            df_engineered = self.add_rate_of_change_features(df_engineered, sensor_cols)
        
        # 3. Health indicators (domain-specific combinations)
        df_engineered = self.add_health_indicators(df_engineered)
        
        # Track engineered features
        original_cols = set(df.columns)
        new_cols = set(df_engineered.columns) - original_cols
        self.engineered_features = list(new_cols)
        
        logger.info(f"Feature engineering complete. Added {len(self.engineered_features)} new features")
        logger.info(f"Final feature count: {len(df_engineered.columns)}")
        
        return df_engineered
    
    def add_rolling_features(self, 
                            df: pd.DataFrame, 
                            columns: List[str],
                            window_sizes: List[int],
                            stats: List[str]) -> pd.DataFrame:
        """
        Add rolling window statistics
        
        Args:
            df: DataFrame
            columns: Columns to compute rolling statistics
            window_sizes: List of window sizes
            stats: List of statistics ('mean', 'std', 'min', 'max')
            
        Returns:
            DataFrame with rolling features
        """
        logger.info(f"Adding rolling statistics for {len(columns)} columns...")
        
        df_rolled = add_rolling_statistics(df, columns, window_sizes, stats)
        
        return df_rolled
    
    def add_rate_of_change_features(self, 
                                    df: pd.DataFrame, 
                                    columns: List[str],
                                    periods: int = 1) -> pd.DataFrame:
        """
        Add rate of change (degradation rate) features
        
        Args:
            df: DataFrame
            columns: Columns to compute rate of change
            periods: Number of periods for difference calculation
            
        Returns:
            DataFrame with rate-of-change features
        """
        logger.info(f"Adding rate-of-change features for {len(columns)} columns...")
        
        df_roc = calculate_rate_of_change(df, columns, periods)
        
        return df_roc
    
    def add_health_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add domain-specific health indicators
        
        Combines multiple sensors to create meaningful health metrics
        based on turbofan engine physics
        
        Args:
            df: DataFrame with sensor data
            
        Returns:
            DataFrame with health indicator features
        """
        logger.info("Adding health indicators...")
        
        df_health = df.copy()
        
        # Health Indicator 1: Temperature Ratio
        # Ratio of HPC outlet temp to fan inlet temp (thermal efficiency indicator)
        if 'sensor_3' in df.columns and 'sensor_1' in df.columns:
            df_health['health_temp_ratio'] = df_health['sensor_3'] / (df_health['sensor_1'] + 1e-6)
        
        # Health Indicator 2: Pressure Ratio
        # Ratio of HPC outlet pressure to fan inlet pressure (compression efficiency)
        if 'sensor_7' in df.columns and 'sensor_5' in df.columns:
            df_health['health_pressure_ratio'] = df_health['sensor_7'] / (df_health['sensor_5'] + 1e-6)
        
        # Health Indicator 3: Speed Ratio
        # Ratio of physical core speed to fan speed (mechanical health)
        if 'sensor_9' in df.columns and 'sensor_8' in df.columns:
            df_health['health_speed_ratio'] = df_health['sensor_9'] / (df_health['sensor_8'] + 1e-6)
        
        # Health Indicator 4: Corrected Speed Average
        # Average of corrected fan and core speeds (overall rotational health)
        if 'sensor_13' in df.columns and 'sensor_14' in df.columns:
            df_health['health_corrected_speed_avg'] = (df_health['sensor_13'] + df_health['sensor_14']) / 2
        
        # Health Indicator 5: Temperature Spread
        # Difference between LPT outlet and fan inlet temperature (thermal stress)
        if 'sensor_4' in df.columns and 'sensor_1' in df.columns:
            df_health['health_temp_spread'] = df_health['sensor_4'] - df_health['sensor_1']
        
        # Health Indicator 6: Coolant Bleed Total
        # Sum of HPT and LPT coolant bleed (cooling demand indicator)
        if 'sensor_20' in df.columns and 'sensor_21' in df.columns:
            df_health['health_coolant_total'] = df_health['sensor_20'] + df_health['sensor_21']
        
        logger.info("Health indicators added")
        
        return df_health
    
    def get_engineered_feature_names(self) -> List[str]:
        """
        Get list of engineered feature names
        
        Returns:
            List of engineered feature names
        """
        return self.engineered_features
    
    def select_top_features(self, 
                           df: pd.DataFrame, 
                           target_col: str = 'RUL',
                           top_k: int = 50) -> List[str]:
        """
        Select top K most correlated features with target
        
        Args:
            df: DataFrame with features and target
            target_col: Target column name
            top_k: Number of top features to select
            
        Returns:
            List of selected feature names
        """
        logger.info(f"Selecting top {top_k} features based on correlation with {target_col}...")
        
        # Get feature columns (exclude meta columns)
        feature_cols = [col for col in df.columns 
                       if col not in ['unit_id', 'time_cycles', target_col]]
        
        # Calculate absolute correlation with target
        correlations = df[feature_cols + [target_col]].corr()[target_col].abs()
        correlations = correlations.drop(target_col).sort_values(ascending=False)
        
        # Select top K
        selected_features = correlations.head(top_k).index.tolist()
        
        logger.info(f"Selected {len(selected_features)} top features")
        logger.info(f"Top 5 features: {selected_features[:5]}")
        
        return selected_features
    
    # Backward compatible method aliases
    def create_rolling_features(self, df, columns, window_sizes, stats):
        """Alias for add_rolling_features"""
        return self.add_rolling_features(df, columns, window_sizes, stats)
    
    def create_rate_of_change_features(self, df, columns, periods=1):
        """Alias for add_rate_of_change_features"""
        return self.add_rate_of_change_features(df, columns, periods)
    
    def create_health_indicators(self, df):
        """Alias for add_health_indicators"""
        return self.add_health_indicators(df)


def engineer_features(train_df: pd.DataFrame, 
                     test_df: pd.DataFrame = None) -> Dict:
    """
    Convenience function to engineer features for train and test data
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame (optional)
        
    Returns:
        Dictionary with engineered DataFrames
    """
    logger.info("Starting feature engineering pipeline...")
    
    engineer = FeatureEngineer()
    
    # Engineer train features
    train_engineered = engineer.create_all_features(train_df)
    
    # Engineer test features if provided
    test_engineered = None
    if test_df is not None:
        test_engineered = engineer.create_all_features(test_df)
    
    logger.info("Feature engineering pipeline complete")
    
    return {
        'train': train_engineered,
        'test': test_engineered,
        'engineer': engineer,
        'feature_names': engineer.get_engineered_feature_names()
    }


if __name__ == "__main__":
    # Test the feature engineer
    from data_loader import load_dataset
    from preprocessor import preprocess_data
    
    print("="*60)
    print("Testing Feature Engineering")
    print("="*60)
    
    # Load and preprocess FD001 dataset
    train_df, test_df, rul_df = load_dataset('FD001')
    preprocessed = preprocess_data(train_df, test_df, rul_df)
    
    # Engineer features
    engineered = engineer_features(
        preprocessed['train'], 
        preprocessed['test']
    )
    
    print("\nEngineered Data Shapes:")
    print(f"  Train: {engineered['train'].shape}")
    if engineered['test'] is not None:
        print(f"  Test: {engineered['test'].shape}")
    
    print(f"\nNumber of engineered features: {len(engineered['feature_names'])}")
    print(f"Sample engineered features: {engineered['feature_names'][:10]}")
    
    print("\nTrain Data Sample (with engineered features):")
    sample_cols = ['unit_id', 'time_cycles', 'RUL'] + engineered['feature_names'][:3]
    print(engineered['train'][sample_cols].head())
    
    # Test feature selection
    top_features = engineered['engineer'].select_top_features(
        engineered['train'], 
        target_col='RUL', 
        top_k=20
    )
    print(f"\nTop 10 features by correlation with RUL:")
    for i, feat in enumerate(top_features[:10], 1):
        print(f"  {i}. {feat}")
    
    print("\n" + "="*60)
    print("Feature engineering test complete!")
    print("="*60)

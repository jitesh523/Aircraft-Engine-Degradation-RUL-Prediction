"""
Automated Model Selection Pipeline for RUL Prediction
Intelligently selects and trains the best model based on dataset characteristics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time

import config
from utils import setup_logging

logger = setup_logging(__name__)


class AutoModelSelector:
    """
    Automated model selection based on dataset characteristics
    Analyzes data properties and recommends/trains optimal models
    """
    
    def __init__(self):
        """Initialize auto model selector"""
        self.dataset_profile = {}
        self.model_registry = {
            'lstm': {
                'name': 'LSTM Neural Network',
                'best_for': ['large_dataset', 'long_sequences', 'temporal_patterns'],
                'min_samples': 5000,
                'complexity': 'high'
            },
            'transformer': {
                'name': 'Transformer',
                'best_for': ['very_large_dataset', 'complex_patterns', 'multi_head_attention'],
                'min_samples': 10000,
                'complexity': 'very_high'
            },
            'xgboost': {
                'name': 'XGBoost',
                'best_for': ['medium_dataset', 'tabular_features', 'fast_training'],
                'min_samples': 1000,
                'complexity': 'medium'
            },
            'lightgbm': {
                'name': 'LightGBM',
                'best_for': ['large_dataset', 'high_cardinality', 'speed'],
                'min_samples': 1000,
                'complexity': 'medium'
            },
            'random_forest': {
                'name': 'Random Forest',
                'best_for': ['small_dataset', 'interpretability', 'robustness'],
                'min_samples': 500,
                'complexity': 'low'
            },
            'ensemble': {
                'name': 'Stacking Ensemble',
                'best_for': ['production', 'best_accuracy', 'diverse_patterns'],
                'min_samples': 3000,
                'complexity': 'very_high'
            }
        }
        
        logger.info("Initialized AutoModelSelector")
    
    def analyze_dataset(self, 
                       df: pd.DataFrame,
                       feature_cols: List[str],
                       target_col: str = 'RUL') -> Dict:
        """
        Analyze dataset characteristics to inform model selection
        
        Args:
            df: Dataset DataFrame
            feature_cols: List of feature columns
            target_col: Target column name
            
        Returns:
            Dictionary with dataset analysis
        """
        logger.info("Analyzing dataset characteristics...")
        
        n_samples = len(df)
        n_features = len(feature_cols)
        n_engines = df['unit_id'].nunique() if 'unit_id' in df.columns else 1
        
        # Calculate per-engine sequence length
        if 'unit_id' in df.columns and 'time_cycles' in df.columns:
            avg_sequence_length = df.groupby('unit_id').size().mean()
            max_sequence_length = df.groupby('unit_id').size().max()
        else:
            avg_sequence_length = n_samples
            max_sequence_length = n_samples
        
        # Feature statistics
        feature_data = df[feature_cols]
        
        # Check for missing values
        missing_ratio = feature_data.isnull().sum().sum() / (n_samples * n_features)
        
        # Check feature variance
        low_variance_features = sum(feature_data.std() < 0.01)
        
        # Check correlations
        corr_matrix = feature_data.corr().abs()
        np.fill_diagonal(corr_matrix.values, 0)
        high_corr_pairs = (corr_matrix > 0.9).sum().sum() // 2
        
        # Target distribution
        if target_col in df.columns:
            target_mean = df[target_col].mean()
            target_std = df[target_col].std()
            target_skewness = df[target_col].skew()
        else:
            target_mean = target_std = target_skewness = None
        
        # Noise estimation (based on rolling std)
        sensor_cols = [c for c in feature_cols if c.startswith('sensor_')]
        if sensor_cols and 'unit_id' in df.columns:
            noise_levels = []
            for col in sensor_cols[:5]:  # Sample first 5 sensors
                rolling_std = df.groupby('unit_id')[col].rolling(5).std().mean()
                noise_levels.append(rolling_std)
            avg_noise = np.mean([n for n in noise_levels if not np.isnan(n)])
        else:
            avg_noise = 0
        
        # Determine dataset size category
        if n_samples >= 50000:
            size_category = 'very_large'
        elif n_samples >= 10000:
            size_category = 'large'
        elif n_samples >= 3000:
            size_category = 'medium'
        else:
            size_category = 'small'
        
        # Complexity score (0-10)
        complexity_score = min(10, (
            (n_features / 20) * 2 +  # Feature count factor
            (avg_sequence_length / 100) * 2 +  # Sequence length factor
            (1 - missing_ratio) * 2 +  # Data quality factor
            (high_corr_pairs / 10) +  # Correlation complexity
            (avg_noise * 5)  # Noise factor
        ))
        
        self.dataset_profile = {
            'n_samples': n_samples,
            'n_features': n_features,
            'n_engines': n_engines,
            'avg_sequence_length': float(avg_sequence_length),
            'max_sequence_length': int(max_sequence_length),
            'size_category': size_category,
            'missing_ratio': float(missing_ratio),
            'low_variance_features': low_variance_features,
            'high_correlation_pairs': high_corr_pairs,
            'target_mean': float(target_mean) if target_mean else None,
            'target_std': float(target_std) if target_std else None,
            'target_skewness': float(target_skewness) if target_skewness else None,
            'avg_noise_level': float(avg_noise) if not np.isnan(avg_noise) else 0,
            'complexity_score': round(complexity_score, 2)
        }
        
        logger.info(f"Dataset profile: {n_samples} samples, {n_features} features, "
                   f"{n_engines} engines, {size_category} dataset")
        
        return self.dataset_profile
    
    def recommend_model(self, 
                       priority: str = 'accuracy',
                       constraints: Dict = None) -> List[Dict]:
        """
        Recommend models based on dataset analysis and priorities
        
        Args:
            priority: 'accuracy', 'speed', 'interpretability', or 'balanced'
            constraints: Dict with constraints like {'max_training_time': 3600}
            
        Returns:
            List of recommended models with scores and rationale
        """
        if not self.dataset_profile:
            raise ValueError("Must call analyze_dataset() first")
        
        logger.info(f"Recommending models with priority: {priority}")
        
        recommendations = []
        profile = self.dataset_profile
        
        for model_id, model_info in self.model_registry.items():
            # Check minimum sample constraint
            if profile['n_samples'] < model_info['min_samples']:
                continue
            
            # Calculate suitability score
            score = 0
            rationale = []
            
            # Size matching
            if profile['size_category'] == 'very_large':
                if model_id in ['transformer', 'lstm', 'lightgbm']:
                    score += 3
                    rationale.append(f"Well-suited for {profile['size_category']} datasets")
            elif profile['size_category'] == 'large':
                if model_id in ['lstm', 'xgboost', 'lightgbm', 'ensemble']:
                    score += 3
                    rationale.append(f"Good for {profile['size_category']} datasets")
            elif profile['size_category'] == 'medium':
                if model_id in ['xgboost', 'lightgbm', 'random_forest']:
                    score += 3
                    rationale.append(f"Efficient for {profile['size_category']} datasets")
            else:  # small
                if model_id in ['random_forest', 'xgboost']:
                    score += 3
                    rationale.append("Works well with limited data")
            
            # Sequence handling
            if profile['avg_sequence_length'] > 50:
                if model_id in ['lstm', 'transformer']:
                    score += 2
                    rationale.append("Handles long sequences well")
            
            # Priority matching
            if priority == 'accuracy':
                if model_id in ['ensemble', 'transformer', 'lstm']:
                    score += 2
                    rationale.append("Optimized for accuracy")
            elif priority == 'speed':
                if model_id in ['lightgbm', 'random_forest', 'xgboost']:
                    score += 2
                    rationale.append("Fast training and inference")
            elif priority == 'interpretability':
                if model_id in ['random_forest', 'xgboost']:
                    score += 2
                    rationale.append("Provides feature importance")
            
            # Complexity handling
            if profile['complexity_score'] > 7:
                if model_id in ['transformer', 'ensemble']:
                    score += 1
                    rationale.append("Handles high complexity")
            
            # Noise robustness
            if profile['avg_noise_level'] > 0.5:
                if model_id in ['ensemble', 'random_forest']:
                    score += 1
                    rationale.append("Robust to noisy data")
            
            recommendations.append({
                'model_id': model_id,
                'model_name': model_info['name'],
                'score': score,
                'complexity': model_info['complexity'],
                'rationale': rationale
            })
        
        # Sort by score
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        # Log top recommendations
        for i, rec in enumerate(recommendations[:3], 1):
            logger.info(f"  #{i}: {rec['model_name']} (score: {rec['score']})")
        
        return recommendations
    
    def get_training_config(self, model_id: str) -> Dict:
        """
        Get recommended training configuration for a model
        
        Args:
            model_id: Model identifier
            
        Returns:
            Training configuration dictionary
        """
        profile = self.dataset_profile
        
        base_configs = {
            'lstm': {
                'epochs': 100 if profile.get('n_samples', 0) > 10000 else 50,
                'batch_size': 64 if profile.get('n_samples', 0) > 10000 else 32,
                'lstm_units': [128, 64] if profile.get('complexity_score', 0) > 5 else [64, 32],
                'dropout': 0.3 if profile.get('avg_noise_level', 0) > 0.3 else 0.2,
                'learning_rate': 0.001,
                'early_stopping_patience': 15
            },
            'transformer': {
                'epochs': 100,
                'batch_size': 32,
                'd_model': 128,
                'n_heads': 8 if profile.get('complexity_score', 0) > 5 else 4,
                'n_layers': 3,
                'dropout': 0.2,
                'learning_rate': 0.0001
            },
            'xgboost': {
                'n_estimators': 500,
                'max_depth': 8 if profile.get('complexity_score', 0) > 5 else 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'early_stopping_rounds': 50
            },
            'lightgbm': {
                'n_estimators': 500,
                'max_depth': 10,
                'learning_rate': 0.05,
                'num_leaves': 31,
                'feature_fraction': 0.8,
                'early_stopping_rounds': 50
            },
            'random_forest': {
                'n_estimators': 200,
                'max_depth': 15,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'n_jobs': -1
            },
            'ensemble': {
                'base_models': ['xgboost', 'lightgbm', 'random_forest'],
                'meta_learner': 'ridge',
                'cv_folds': 5
            }
        }
        
        return base_configs.get(model_id, {})
    
    def quick_benchmark(self,
                       df: pd.DataFrame,
                       feature_cols: List[str],
                       target_col: str = 'RUL',
                       test_size: float = 0.2,
                       models_to_test: List[str] = None) -> pd.DataFrame:
        """
        Quick benchmark of multiple models on the dataset
        
        Args:
            df: Dataset DataFrame
            feature_cols: Feature column names
            target_col: Target column name
            test_size: Test set ratio
            models_to_test: List of model IDs to test (default: quick models)
            
        Returns:
            DataFrame with benchmark results
        """
        if models_to_test is None:
            # Default to quick models for benchmarking
            models_to_test = ['random_forest', 'xgboost']
        
        logger.info(f"Starting quick benchmark with models: {models_to_test}")
        
        # Prepare data
        X = df[feature_cols].values
        y = df[target_col].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        results = []
        
        for model_id in models_to_test:
            try:
                logger.info(f"Benchmarking {model_id}...")
                
                # Create model
                if model_id == 'random_forest':
                    from sklearn.ensemble import RandomForestRegressor
                    model = RandomForestRegressor(
                        n_estimators=100, max_depth=10, n_jobs=-1, random_state=42
                    )
                elif model_id == 'xgboost':
                    from xgboost import XGBRegressor
                    model = XGBRegressor(
                        n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
                    )
                else:
                    continue
                
                # Train
                start_time = time.time()
                model.fit(X_train, y_train)
                train_time = time.time() - start_time
                
                # Predict
                start_time = time.time()
                y_pred = model.predict(X_test)
                inference_time = (time.time() - start_time) / len(X_test) * 1000  # ms per sample
                
                # Metrics
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                results.append({
                    'model': model_id,
                    'rmse': round(rmse, 2),
                    'mae': round(mae, 2),
                    'r2': round(r2, 4),
                    'train_time_s': round(train_time, 2),
                    'inference_ms': round(inference_time, 3)
                })
                
                logger.info(f"  {model_id}: RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.4f}")
                
            except Exception as e:
                logger.warning(f"Error benchmarking {model_id}: {e}")
        
        return pd.DataFrame(results).sort_values('rmse')
    
    def generate_selection_report(self) -> str:
        """Generate model selection report"""
        if not self.dataset_profile:
            return "No dataset analyzed yet. Call analyze_dataset() first."
        
        recommendations = self.recommend_model()
        
        report_lines = [
            "=" * 60,
            "AUTO MODEL SELECTION REPORT",
            "=" * 60,
            "",
            "DATASET PROFILE:",
            f"  Samples: {self.dataset_profile['n_samples']:,}",
            f"  Features: {self.dataset_profile['n_features']}",
            f"  Engines: {self.dataset_profile['n_engines']}",
            f"  Size Category: {self.dataset_profile['size_category'].upper()}",
            f"  Complexity Score: {self.dataset_profile['complexity_score']}/10",
            "",
            "TOP RECOMMENDATIONS:",
        ]
        
        for i, rec in enumerate(recommendations[:3], 1):
            report_lines.append(f"  {i}. {rec['model_name']} (Score: {rec['score']})")
            for r in rec['rationale']:
                report_lines.append(f"     • {r}")
            report_lines.append("")
        
        report_lines.append("=" * 60)
        
        return '\n'.join(report_lines)


if __name__ == "__main__":
    print("="*60)
    print("Testing Auto Model Selector")
    print("="*60)
    
    # Generate synthetic dataset
    np.random.seed(42)
    n_samples = 5000
    
    df = pd.DataFrame({
        'unit_id': np.repeat(range(50), 100),
        'time_cycles': np.tile(range(100), 50),
        'sensor_1': np.random.randn(n_samples),
        'sensor_2': np.random.randn(n_samples) * 2,
        'sensor_3': np.random.randn(n_samples) * 1.5,
        'sensor_4': np.random.randn(n_samples) * 0.5,
        'RUL': np.maximum(0, np.tile(range(99, -1, -1), 50) + np.random.randn(n_samples) * 5)
    })
    
    feature_cols = ['sensor_1', 'sensor_2', 'sensor_3', 'sensor_4']
    
    # Create selector
    selector = AutoModelSelector()
    
    # Analyze dataset
    print("\n1. Analyzing dataset...")
    profile = selector.analyze_dataset(df, feature_cols)
    
    # Get recommendations
    print("\n2. Getting model recommendations...")
    recommendations = selector.recommend_model(priority='accuracy')
    
    # Generate report
    print("\n3. Selection Report:")
    print(selector.generate_selection_report())
    
    # Quick benchmark
    print("\n4. Quick benchmark:")
    benchmark = selector.quick_benchmark(df, feature_cols)
    print(benchmark.to_string(index=False))
    
    print("\n" + "="*60)
    print("Auto model selector test complete!")
    print("="*60)

"""
Configuration file for Aircraft Engine RUL Prediction System
Contains dataset paths, model hyperparameters, and system settings
"""

import os

# ==================== Dataset Configuration ====================
DATA_DIR = "/Users/neha/Downloads/CMAPSSData"

# Dataset files
DATASETS = {
    'FD001': {
        'train': os.path.join(DATA_DIR, 'train_FD001.txt'),
        'test': os.path.join(DATA_DIR, 'test_FD001.txt'),
        'rul': os.path.join(DATA_DIR, 'RUL_FD001.txt'),
        'description': 'ONE operating condition, ONE fault mode (HPC Degradation)',
        'train_engines': 100,
        'test_engines': 100
    },
    'FD002': {
        'train': os.path.join(DATA_DIR, 'train_FD002.txt'),
        'test': os.path.join(DATA_DIR, 'test_FD002.txt'),
        'rul': os.path.join(DATA_DIR, 'RUL_FD002.txt'),
        'description': 'SIX operating conditions, ONE fault mode (HPC Degradation)',
        'train_engines': 260,
        'test_engines': 259
    },
    'FD003': {
        'train': os.path.join(DATA_DIR, 'train_FD003.txt'),
        'test': os.path.join(DATA_DIR, 'test_FD003.txt'),
        'rul': os.path.join(DATA_DIR, 'RUL_FD003.txt'),
        'description': 'ONE operating condition, TWO fault modes (HPC & Fan Degradation)',
        'train_engines': 100,
        'test_engines': 100
    },
    'FD004': {
        'train': os.path.join(DATA_DIR, 'train_FD004.txt'),
        'test': os.path.join(DATA_DIR, 'test_FD004.txt'),
        'rul': os.path.join(DATA_DIR, 'RUL_FD004.txt'),
        'description': 'SIX operating conditions, TWO fault modes (HPC & Fan Degradation)',
        'train_engines': 248,
        'test_engines': 249
    }
}

# Column names for the dataset (26 columns total)
COLUMN_NAMES = [
    'unit_id',           # Engine unit identifier
    'time_cycles',       # Time in cycles
    'setting_1',         # Operational setting 1
    'setting_2',         # Operational setting 2
    'setting_3',         # Operational setting 3
    'sensor_1',          # Sensor measurement 1 (T2 - Total temperature at fan inlet)
    'sensor_2',          # Sensor measurement 2 (T24 - Total temperature at LPC outlet)
    'sensor_3',          # Sensor measurement 3 (T30 - Total temperature at HPC outlet)
    'sensor_4',          # Sensor measurement 4 (T50 - Total temperature at LPT outlet)
    'sensor_5',          # Sensor measurement 5 (P2 - Pressure at fan inlet)
    'sensor_6',          # Sensor measurement 6 (P15 - Total pressure in bypass-duct)
    'sensor_7',          # Sensor measurement 7 (P30 - Total pressure at HPC outlet)
    'sensor_8',          # Sensor measurement 8 (Nf - Physical fan speed)
    'sensor_9',          # Sensor measurement 9 (Nc - Physical core speed)
    'sensor_10',         # Sensor measurement 10 (epr - Engine pressure ratio)
    'sensor_11',         # Sensor measurement 11 (Ps30 - Static pressure at HPC outlet)
    'sensor_12',         # Sensor measurement 12 (phi - Ratio of fuel flow to Ps30)
    'sensor_13',         # Sensor measurement 13 (NRf - Corrected fan speed)
    'sensor_14',         # Sensor measurement 14 (NRc - Corrected core speed)
    'sensor_15',         # Sensor measurement 15 (BPR - Bypass ratio)
    'sensor_16',         # Sensor measurement 16 (farB - Burner fuel-air ratio)
    'sensor_17',         # Sensor measurement 17 (htBleed - Bleed enthalpy)
    'sensor_18',         # Sensor measurement 18 (Nf_dmd - Demanded fan speed)
    'sensor_19',         # Sensor measurement 19 (PCNfR_dmd - Demanded corrected fan speed)
    'sensor_20',         # Sensor measurement 20 (W31 - HPT coolant bleed)
    'sensor_21'          # Sensor measurement 21 (W32 - LPT coolant bleed)
]

# Sensors to drop (low variance, not predictive)
# These will be identified during EDA, but commonly:
SENSORS_TO_DROP = ['sensor_1', 'sensor_5', 'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19']

# ==================== Preprocessing Configuration ====================
# RUL clipping - use piecewise linear labeling
# Early cycles have less predictive signal, so cap RUL at max value
USE_RUL_CLIPPING = True
MAX_RUL = 125  # Cap RUL at this value for early cycles

# Normalization method: 'minmax' or 'standard'
NORMALIZATION_METHOD = 'minmax'

# Train/validation split ratio
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42

# ==================== Feature Engineering ====================
# Rolling window statistics
ROLLING_WINDOW_SIZES = [5, 10, 15]  # cycles
ROLLING_FEATURES = ['mean', 'std', 'min', 'max']

# Create rate-of-change features
CREATE_RATE_OF_CHANGE = True

# ==================== Model Configuration ====================

# LSTM Model Hyperparameters
LSTM_CONFIG = {
    'sequence_length': 30,      # Number of time steps to look back
    'lstm_units': [100, 50],    # Units in each LSTM layer
    'dropout_rate': 0.2,        # Dropout for regularization
    'learning_rate': 0.001,     # Adam optimizer learning rate
    'batch_size': 256,          # Training batch size
    'epochs': 100,              # Maximum training epochs
    'patience': 15,             # Early stopping patience
    'validation_split': 0.2     # Validation split during training
}

# Transformer Model Hyperparameters
TRANSFORMER_CONFIG = {
    'head_size': 256,
    'num_heads': 4,
    'ff_dim': 4,
    'num_transformer_blocks': 4,
    'mlp_units': [128],
    'dropout_rate': 0.4,
    'mlp_dropout': 0.4,
    'learning_rate': 0.001,
    'batch_size': 64,
    'epochs': 100,
    'patience': 20
}

# Baseline Model Configuration
BASELINE_CONFIG = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 20,
        'min_samples_split': 5,
        'random_state': RANDOM_SEED
    },
    'linear_regression': {
        'fit_intercept': True
    }
}

# Anomaly Detection Configuration
ANOMALY_CONFIG = {
    'method': 'isolation_forest',  # 'isolation_forest' or 'autoencoder'
    'contamination': 0.1,           # Expected proportion of outliers
    'random_state': RANDOM_SEED
}

# ==================== Phase 1: Advanced Ensemble Methods ====================
# XGBoost Configuration
XGBOOST_CONFIG = {
    'objective': 'reg:squarederror',
    'n_estimators': 500,
    'max_depth': 7,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': RANDOM_SEED
}

# LightGBM Configuration
LIGHTGBM_CONFIG = {
    'objective': 'regression',
    'boosting_type': 'gbdt',
    'n_estimators': 500,
    'num_leaves': 63,
    'max_depth': 8,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': RANDOM_SEED
}

# CatBoost Configuration
CATBOOST_CONFIG = {
    'loss_function': 'RMSE',
    'iterations': 500,
    'depth': 7,
    'learning_rate': 0.05,
    'l2_leaf_reg': 3.0,
    'random_seed': RANDOM_SEED,
    'verbose': False
}

# Stacking Ensemble Configuration
STACKING_CONFIG = {
    'n_folds': 5,
    'meta_learner': 'ridge',  # 'ridge' or 'mlp'
    'meta_learner_params': {
        'ridge': {'alpha': 10.0},
        'mlp': {
            'hidden_layer_sizes': (50, 25),
            'activation': 'relu',
            'max_iter': 1000,
            'random_state': RANDOM_SEED
        }
    }
}

# MLflow Configuration
MLFLOW_CONFIG = {
    'tracking_uri': 'file:./mlruns',  # or http://localhost:5000
    'experiment_name': 'RUL_Prediction',
    'artifact_location': None,
    'register_models': True
}

# A/B Testing Configuration
AB_TESTING_CONFIG = {
    'significance_level': 0.05,
    'statistical_power': 0.8,
    'min_sample_size': 100,
    'bootstrap_iterations': 1000,
    'auto_promote_threshold': 0.10  # Promote if >10% improvement
}

# ==================== Phase 2: Streaming & Edge Deployment ====================
# Redis Streaming Configuration
REDIS_CONFIG = {
    'host': 'localhost',
    'port': 6379,
    'db': 0,
    'input_stream': 'engine:sensors',
    'output_stream': 'predictions:rul',
    'consumer_group': 'rul_processors',
    'max_stream_length': 100000
}

# Stream Processing Configuration
STREAM_PROCESSING_CONFIG = {
    'batch_size': 10,
    'block_time_ms': 1000,
    'window_size': 15,  # For rolling features
    'processing_delay_ms': 10
}

# Model Quantization Configuration
QUANTIZATION_CONFIG = {
    'default_type': 'dynamic',  # 'dynamic', 'float16', 'int8'
    'int8_representative_samples': 1000,
    'target_size_reduction_percent': 75.0
}

# ONNX Export Configuration
ONNX_CONFIG = {
    'opset_version': 13,
    'optimize': True,
    'model_types': ['tensorflow', 'sklearn']
}

# Edge Deployment Configuration
EDGE_DEPLOYMENT_CONFIG = {
    'default_model_type': 'tflite',  # 'tflite' or 'onnx'
    'batch_size': 32,
    'target_devices': ['raspberry_pi', 'jetson_nano', 'mobile']
}

# ==================== Maintenance Planning ====================
# RUL Thresholds for maintenance zones
MAINTENANCE_THRESHOLDS = {
    'critical': 30,    # RUL < 30 cycles - Immediate maintenance required
    'warning': 80,     # 30 <= RUL < 80 - Schedule maintenance soon
    'healthy': 80      # RUL >= 80 - Routine monitoring
}

# Cost Analysis (in dollars)
COST_PARAMETERS = {
    'scheduled_maintenance': 10000,      # Cost of planned maintenance
    'unscheduled_maintenance': 50000,    # Cost of emergency repair (5x higher)
    'downtime_per_hour': 5000,           # Cost of aircraft being grounded
    'false_alarm_cost': 2000             # Cost of unnecessary maintenance
}

# Traditional maintenance schedule (for comparison)
TRADITIONAL_MAINTENANCE_INTERVAL = 150  # Fixed interval in cycles

# ==================== Output Configuration ====================
# Directories for saving models and results
PROJECT_ROOT = "/Users/neha/Aircraft-Engine-Degradation-RUL-Prediction"
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models', 'saved')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
PLOTS_DIR = os.path.join(PROJECT_ROOT, 'plots')
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')

# Create directories if they don't exist
for directory in [MODELS_DIR, RESULTS_DIR, PLOTS_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

# ==================== Evaluation Metrics ====================
# Target performance metrics
TARGET_METRICS = {
    'RMSE': 25,      # Target RMSE on test set (cycles)
    'MAE': 20,       # Target MAE on test set (cycles)
    'R2': 0.7        # Target R-squared score
}

# Asymmetric loss weights (penalize under-prediction more)
ASYMMETRIC_LOSS_WEIGHTS = {
    'late_prediction_penalty': 13,   # Weight for predicting failure too late
    'early_prediction_penalty': 10   # Weight for predicting failure too early
}

# ==================== Visualization Settings ====================
PLOT_CONFIG = {
    'figure_size': (12, 6),
    'dpi': 100,
    'style': 'seaborn-v0_8-darkgrid',
    'save_format': 'png'
}

# ==================== Logging Configuration ====================
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'datefmt': '%Y-%m-%d %H:%M:%S'
}


class ConfigurationManager:
    """
    Dynamic configuration manager with environment overrides
    Supports loading from YAML/JSON files and validation
    """
    
    def __init__(self, config_file: str = None):
        """
        Initialize configuration manager
        
        Args:
            config_file: Path to config file (YAML or JSON)
        """
        self.config = {}
        self.defaults = self._get_defaults()
        self.config_file = config_file
        
        # Load from file if provided
        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)
        else:
            self.config = self.defaults.copy()
        
        # Apply environment overrides
        self._apply_env_overrides()
    
    def _get_defaults(self) -> dict:
        """Get default configuration values"""
        return {
            'data_dir': DATA_DIR,
            'models_dir': MODELS_DIR,
            'results_dir': RESULTS_DIR,
            'sequence_length': LSTM_CONFIG.get('sequence_length', 30),
            'random_seed': RANDOM_SEED,
            'rul_cap': MAX_RUL,
            'lstm': LSTM_CONFIG.copy(),
            'logging': LOGGING_CONFIG.copy(),
            'target_metrics': TARGET_METRICS.copy()
        }
    
    def load_from_file(self, filepath: str):
        """
        Load configuration from file
        
        Args:
            filepath: Path to YAML or JSON config file
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Config file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            if filepath.endswith('.yaml') or filepath.endswith('.yml'):
                try:
                    import yaml
                    loaded = yaml.safe_load(f)
                except ImportError:
                    # Fallback if yaml not installed
                    loaded = {}
            elif filepath.endswith('.json'):
                import json
                loaded = json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {filepath}")
        
        # Merge with defaults
        self.config = self._deep_merge(self.defaults, loaded)
        self.config_file = filepath
    
    def _deep_merge(self, base: dict, override: dict) -> dict:
        """Deep merge two dictionaries"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides"""
        env_mappings = {
            'RUL_DATA_DIR': 'data_dir',
            'RUL_MODELS_DIR': 'models_dir',
            'RUL_RESULTS_DIR': 'results_dir',
            'RUL_SEQUENCE_LENGTH': ('sequence_length', int),
            'RUL_RANDOM_SEED': ('random_seed', int),
            'RUL_RUL_CAP': ('rul_cap', int),
            'RUL_LOG_LEVEL': ('logging.level', str)
        }
        
        for env_var, mapping in env_mappings.items():
            value = os.environ.get(env_var)
            if value is not None:
                if isinstance(mapping, tuple):
                    key, converter = mapping
                    value = converter(value)
                else:
                    key = mapping
                
                # Handle nested keys
                if '.' in key:
                    parts = key.split('.')
                    target = self.config
                    for part in parts[:-1]:
                        target = target.setdefault(part, {})
                    target[parts[-1]] = value
                else:
                    self.config[key] = value
    
    def get(self, key: str, default=None):
        """
        Get configuration value
        
        Args:
            key: Dot-separated key path (e.g., 'lstm.units')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        parts = key.split('.')
        value = self.config
        
        try:
            for part in parts:
                value = value[part]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value):
        """
        Set configuration value
        
        Args:
            key: Dot-separated key path
            value: Value to set
        """
        parts = key.split('.')
        target = self.config
        
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        
        target[parts[-1]] = value
    
    def validate(self) -> dict:
        """
        Validate configuration
        
        Returns:
            Validation results with any errors/warnings
        """
        errors = []
        warnings = []
        
        # Check required directories
        for dir_key in ['data_dir', 'models_dir', 'results_dir']:
            path = self.get(dir_key)
            if not path:
                errors.append(f"Missing required config: {dir_key}")
            elif not os.path.exists(path):
                warnings.append(f"Directory does not exist: {path}")
        
        # Check numeric constraints
        if self.get('sequence_length', 0) <= 0:
            errors.append("sequence_length must be positive")
        
        if self.get('rul_cap', 0) <= 0:
            errors.append("rul_cap must be positive")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    def save_to_file(self, filepath: str):
        """
        Save current configuration to file
        
        Args:
            filepath: Output file path
        """
        import json
        
        with open(filepath, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def get_summary(self) -> str:
        """Get configuration summary"""
        lines = [
            "=" * 60,
            "CONFIGURATION SUMMARY",
            "=" * 60,
            f"Config file: {self.config_file or 'None (defaults)'}",
            "",
            "Core Settings:"
        ]
        
        for key in ['data_dir', 'models_dir', 'results_dir', 'sequence_length', 'random_seed', 'rul_cap']:
            lines.append(f"  {key}: {self.get(key)}")
        
        lines.extend([
            "",
            "LSTM Config:"
        ])
        
        for key, value in self.get('lstm', {}).items():
            lines.append(f"  {key}: {value}")
        
        lines.append("=" * 60)
        
        return '\n'.join(lines)


print(f"Configuration loaded. Project root: {PROJECT_ROOT}")
print(f"Dataset directory: {DATA_DIR}")


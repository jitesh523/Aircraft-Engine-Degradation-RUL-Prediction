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

print(f"Configuration loaded. Project root: {PROJECT_ROOT}")
print(f"Dataset directory: {DATA_DIR}")

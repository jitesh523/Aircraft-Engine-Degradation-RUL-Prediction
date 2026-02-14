# Aircraft Engine Degradation & RUL Prediction

**Predictive Maintenance System for Turbofan Engines using NASA C-MAPSS Dataset**

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![CI/CD](https://github.com/jitesh523/Aircraft-Engine-Degradation-RUL-Prediction/workflows/CI%2FCD%20Pipeline/badge.svg)
![Security](https://img.shields.io/badge/Security-Bandit%20%7C%20CodeQL-brightgreen.svg)
![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)

## Overview

This project implements a comprehensive Remaining Useful Life (RUL) prediction system for aircraft turbofan engines using the NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) dataset. The system uses deep learning (LSTM networks) combined with advanced feature engineering to predict when engines will fail, enabling proactive maintenance scheduling that reduces costs and improves fleet availability.

### Key Features

#### Core Models
- **LSTM Neural Network**: Deep learning model for time-series RUL prediction
- **Gradient Boosting Models**: XGBoost, LightGBM, and CatBoost for enhanced accuracy
- **Stacking Ensemble**: Meta-learning approach combining multiple base models
- **Baseline Models**: Random Forest and Linear Regression for comparison
- **Anomaly Detection**: Early fault warning system using Isolation Forest

#### Advanced Features (Phase 1)
- **MLflow Integration**: Comprehensive experiment tracking and model versioning
- **A/B Testing Framework**: Statistical comparison and champion/challenger deployment
- **Model Registry**: Production-ready model management and versioning
- **Feature Engineering**: Rolling statistics, rate-of-change features, and domain-specific health indicators
- **Maintenance Planning**: AI-driven scheduling with cost/benefit analysis
- **Comprehensive Evaluation**: RMSE, MAE, R², asymmetric scoring, and visualization

#### Latest Enhancements (Phase 4)
- **Time-Series Cross-Validation**: Proper temporal CV with confidence intervals for robust model evaluation
- **Bootstrap Confidence Intervals**: Statistical confidence bounds on all performance metrics
- **Data Augmentation**: Time-series specific augmentation (jittering, scaling, window slicing, degradation interpolation)
- **Early Warning System**: Multi-level alerting (EMERGENCY → MONITOR) with degradation rate analysis
- **Fleet Health Scoring**: Real-time fleet-wide health metrics and prioritized maintenance queues
- **Model Comparison Utility**: Statistical significance testing and detailed error analysis by RUL ranges

#### Causal Analytics & Monitoring (Phase 5)
- **Instrumental Variables (IV) Estimator**: 2SLS regression for causal inference in observational data
- **Power Calculator**: Experiment design tools for A/B testing sample size and power analysis
- **Drift Monitoring Dashboard**: Real-time tracking of feature drift (PSI) and concept drift in the Streamlit dashboard

#### AI & Optimization (Phase 6)
- **LLM-Powered Maintenance Assistant**: Natural language fleet analysis using Google Gemini with rule-based fallback
- **RL-Based Maintenance Optimizer**: Q-Learning agent for optimal maintenance scheduling (23% cost reduction)

#### Survival Analysis, Multi-Dataset & Fleet Ops (Phase 7)
- **Survival Analysis Engine**: Kaplan-Meier and Cox Proportional Hazards models for time-to-failure probability distributions
- **Multi-Dataset Training Pipeline**: Cross-dataset training on FD001–FD004 with feature harmonization, MMD-based domain adaptation, and transfer learning
- **Real-Time Fleet Ops Center**: Live fleet health heatmap, health score gauge, geo-map, notification alerts, and sortable maintenance priority queue

#### Federated Learning, Root Cause & What-If (Phase 8)
- **Federated Learning Simulator**: Privacy-preserving distributed training across airline sites using FedAvg — trains without sharing raw sensor data
- **Anomaly Root Cause Analyzer**: Identifies which sensors caused anomalies, matches against known C-MAPSS failure mode patterns (HPC/Fan/LPT degradation), generates diagnostic reports
- **What-If Scenario Simulator**: Counterfactual simulations for delayed maintenance, accelerated sensor drift, and fleet-wide strategy comparison (proactive vs reactive vs fixed-interval)

#### Sensor Networks, Degradation Clustering & Scheduling (Phase 9)
- **Sensor Correlation Network**: Graph-based sensor interdependency analysis with community detection, anomaly propagation path tracing, and degradation correlation shifts
- **Degradation Pattern Clustering**: Unsupervised K-Means clustering on trajectory features to discover engine degradation archetypes with PCA visualization and silhouette optimization
- **Predictive Maintenance Scheduler**: Constraint-based fleet scheduling with hangar capacity limits, three strategies (priority, cost-min, balanced), Gantt charts, and utilization tracking

## Dataset

The NASA C-MAPSS dataset contains run-to-failure data from turbofan engine simulations with varying operating conditions and fault modes:

- **FD001**: Single operating condition, single fault mode (HPC degradation) - 100 train/100 test engines
- **FD002**: Six operating conditions, single fault mode - 260 train/259 test engines
- **FD003**: Single operating condition, two fault modes - 100 train/100 test engines
- **FD004**: Six operating conditions, two fault modes - 248 train/249 test engines

Each dataset provides:
- **26 columns**: Unit ID, time cycles, 3 operational settings, 21 sensor measurements
- **Training data**: Complete run-to-failure sequences
- **Test data**: Partial sequences with ground truth RUL values

**Data Source**: [NASA PCoE Datasets](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data)

## Installation

### Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd Aircraft-Engine-Degradation-RUL-Prediction

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- tensorflow (or pytorch)
- jupyter (optional, for notebooks)

## Project Structure

```
Aircraft-Engine-Degradation-RUL-Prediction/
├── config.py                 # Configuration and hyperparameters
├── utils.py                  # Utility functions
├── data_loader.py            # Dataset loading and parsing
├── preprocessor.py           # Data preprocessing and normalization
├── feature_engineer.py       # Feature engineering module
├── evaluator.py              # Model evaluation metrics
├── visualizer.py             # Visualization functions
├── maintenance_planner.py    # Maintenance scheduling logic
├── train.py                  # Training pipeline
├── predict.py                # Prediction script
├── models/
│   ├── __init__.py
│   ├── baseline_model.py     # Random Forest & Linear Regression
│   ├── lstm_model.py         # LSTM neural network
│   └── anomaly_detector.py   # Anomaly detection module
├── models/saved/             # Trained model files
├── results/                  # Evaluation results and metrics
├── plots/                    # Generated visualizations
└── logs/                     # Training logs
```

## Usage

### 1. Training Models

Train models on the FD001 dataset:

```bash
python train.py --dataset FD001
```

Options:
- `--dataset`: Choose dataset (FD001, FD002, FD003, FD004)
- `--skip-baseline`: Skip baseline model training
- `--skip-lstm`: Skip LSTM model training  
- `--skip-anomaly`: Skip anomaly detector training

Training Process:
1. Loads and preprocesses data
2. Engineers features (rolling stats, rate-of-change, health indicators)
3. Trains baseline models (Random Forest, Linear Regression)
4. Trains LSTM model with early stopping
5. Trains anomaly detector on healthy engines
6. Saves all models and scalers

Expected Training Time (FD001):
- Baseline models: ~1-2 minutes
- LSTM model: ~10-15 minutes (with early stopping)

### 2. Making Predictions

Make predictions on test data:

```bash
python predict.py --dataset FD001
```

Options:
- `--dataset`: Dataset to predict on
- `--no-viz`: Skip visualization generation

Prediction Process:
1. Loads trained models and preprocessor
2. Preprocesses and engineers features for test data
3. Generates time-series sequences
4. Makes RUL predictions
5. Evaluates performance (RMSE, MAE, R²)
6. Creates visualizations (scatter plots, error distributions)
7. Generates maintenance schedule
8. Performs cost/benefit analysis

### 3. Phase 1: Advanced Ensemble Methods

Train gradient boosting models and stacking ensemble with MLflow tracking:

```bash
# Train all Phase 1 models (XGBoost, LightGBM, CatBoost, Stacking Ensemble)
python train_phase1.py --dataset FD001

# Train without MLflow tracking
python train_phase1.py --dataset FD001 --no-mlflow

# View MLflow results
mlflow ui --port 5000
# Open http://localhost:5000 in browser
```

See [MLFLOW_GUIDE.md](MLFLOW_GUIDE.md) for detailed MLflow usage.

### 4. Configuration

Edit `config.py` to customize:

**Model Hyperparameters**:
```python
LSTM_CONFIG = {
    'sequence_length': 30,      # Time steps to look back
    'lstm_units': [100, 50],    # Units in each LSTM layer
    'dropout_rate': 0.2,
    'learning_rate': 0.001,
    'batch_size': 256,
    'epochs': 100,
    'patience': 15
}
```

**Maintenance Thresholds**:
```python
MAINTENANCE_THRESHOLDS = {
    'critical': 30,    # Immediate maintenance required
    'warning': 80,     # Schedule maintenance soon
    'healthy': 80      # Routine monitoring
}
```

## Model Architecture

### LSTM Network

```
Input: (sequence_length, num_features)
  ↓
LSTM Layer 1 (100 units) + Dropout (0.2)
  ↓
LSTM Layer 2 (50 units) + Dropout (0.2)
  ↓
Dense Output (1 unit, linear activation)
  ↓
Output: Predicted RUL (cycles)
```

### Feature Engineering

**Rolling Window Statistics** (windows: 5, 10, 15 cycles):
- Mean, std, min, max for all sensors

**Rate of Change**:
- First-order differences for degradation trends

**Health Indicators** (domain-specific):
- Temperature ratios (thermal efficiency)
- Pressure ratios (compression efficiency)
- Speed ratios (mechanical health)
- Coolant bleed (cooling demand)

## Performance

### Target Metrics (FD001)

- **RMSE**: ≤ 25 cycles
- **MAE**: ≤ 20 cycles
- **R²**: ≥ 0.7

### Expected Results

#### Baseline Performance
- Baseline Random Forest: **RMSE ~25-30 cycles**
- Well-tuned LSTM: **RMSE ~20-25 cycles** on FD001 test set
- Anomaly Detection: **Precision ~0.7, Recall ~0.6** for failing engines

#### Phase 1: Advanced Ensemble Methods
- XGBoost: **RMSE ~18-22 cycles**
- LightGBM: **RMSE ~18-22 cycles**
- CatBoost: **RMSE ~18-22 cycles**
- Stacking Ensemble: **RMSE ~15-18 cycles, R² ~0.80-0.85**

### Maintenance Impact

Compared to fixed 150-cycle maintenance schedule:
- **Cost Reduction**: 60-83%
- **Fleet Availability**: 75% → 90-100%
- **Unexpected Failures**: Reduced to near-zero

## Evaluation

The system evaluates models using:

### 1. Regression Metrics
- **RMSE** (Root Mean Squared Error): Primary metric for RUL prediction accuracy
- **MAE** (Mean Absolute Error): Average prediction error
- **R²**: Goodness of fit

### 2. Asymmetric Scoring
NASA's asymmetric scoring function that penalizes late predictions (under-predicting RUL) more heavily than early predictions, as missing failures is more dangerous than false alarms.

### 3. Maintenance Metrics
- Number of scheduled vs unscheduled maintenances
- Total maintenance cost
- Fleet availability percentage
- Cost savings vs traditional approaches

## Visualization

The system generates:

1. **Prediction Scatter Plots**: Predicted vs  Actual RUL
2. **Error Distributions**: Histogram and box plots of prediction errors
3. **Training History**: Loss and MAE curves over epochs
4. **Engine Trajectories**: RUL progression over engine lifetime
5. **Sensor Trends**: Degradation patterns in sensor readings

## Maintenance Planning

The maintenance planner classifies engines into three zones:

| Health Status | RUL Range | Action |
|---------------|-----------|--------|
| 🔴 Critical | < 30 cycles | Immediate maintenance - Ground aircraft |
| 🟡 Warning | 30-80 cycles | Schedule maintenance soon |
| 🟢 Healthy | ≥ 80 cycles | Continue routine monitoring |

### Cost/Benefit Analysis

Compares two strategies:
1. **Traditional**: Fixed 150-cycle maintenance intervals
2. **Predictive**: AI-driven scheduling based on RUL predictions

Metrics:
- Scheduled maintenance cost: $10,000
- Unscheduled maintenance cost: $50,000 (5x higher)
- False alarm cost: $2,000

## Implementation Details

### Data Pipeline

1. **Loading**: Parse space-separated text files, assign column names
2. **Preprocessing**: Create RUL labels, handle sensor noise, normalize features
3. **Feature Engineering**: Generate rolling statistics, rate-of-change, health indicators
4. **Sequence Generation**: Create time-series windows for LSTM input

### Training Strategy

- **Train/Validation Split**: 80/20 by engine units (maintains temporal integrity)
- **Normalization**: MinMax scaling fitted on training data only
- **Early Stopping**: Monitor validation loss with patience=15 epochs
- **Regularization**: Dropout (0.2) in LSTM layers

### Anomaly Detection

- **Method**: Isolation Forest
- **Training**: Fit on healthy engines only (RUL > 80)
- **Purpose**: Early fault detection before RUL becomes critical
- **Integration**: Complements RUL predictions with early warnings

## Results Files

After training and prediction:

```
results/
├── FD001_predictions.csv              # Unit ID, true RUL, predicted RUL
├── FD001_test_metrics.json            # RMSE, MAE, R², asymmetric score
├── FD001_maintenance_schedule.csv     # Health status, recommended actions
├── FD001_strategy_comparison.csv      # Traditional vs predictive comparison
├── baseline_metrics.json              # Random Forest & Linear Regression metrics
└── lstm_metrics.json                  # LSTM training metrics

plots/
├── FD001_prediction_scatter.png
├── FD001_error_distribution.png
└── lstm_training_history.png

models/saved/
├── lstm_model.h5                      # Trained LSTM model
├── baseline_rf.pkl                    # Random Forest model
├── baseline_lr.pkl                    # Linear Regression model
├── anomaly_detector.pkl               # Anomaly detector
├── scaler.pkl                         # Fitted scaler
└── feature_info.json                  # Feature column names and metadata
```

## References

1. **NASA C-MAPSS Dataset**: [NASA Open Data Portal](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data)
2. **PDF Guide**: "Aircraft Engine Degradation & RUL Prediction – Step-by-Step Guide (Using NASA C-MAPSS)"
3. **Research Paper**: A. Saxena et al., "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation", PHM08, 2008
4. **Medium Article**: Mihai Timoficiuc, "Predicting Jet Engine Failures with NASA's C-MAPSS Dataset (LSTM Guide)", 2025

## License

MIT License

## Acknowledgments

- NASA PCoE for the C-MAPSS dataset
- Step-by-step implementation guide from the provided PDF
- Machine learning community for baseline implementations

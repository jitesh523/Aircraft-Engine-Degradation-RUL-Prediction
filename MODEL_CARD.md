# Model Card: Aircraft Engine RUL Predictor

## Model Details

- **Name**: Hybrid Ensemble Engine RUL Predictor
- **Version**: 2.0.0
- **Date**: February 2026
- **Type**: Deep Learning / Ensemble / Time Series Regression
- **Framework**: TensorFlow/Keras, Scikit-learn, XGBoost, LightGBM
- **Architecture**: LSTM + Transformer + Gradient Boosting Ensemble with stacking

## Intended Use

### Primary Use Case
Predicting Remaining Useful Life (RUL) of aircraft turbofan engines based on sensor reading history. Enables predictive maintenance strategies that optimize fleet availability and reduce costs.

### Users
- Maintenance engineers
- Fleet managers
- Data scientists in aerospace predictive maintenance

### Out of Scope Use Cases
- Real-time flight safety critical systems (without further certification)
- Engines significantly different from the simulated C-MAPSS dataset specifications
- Scenarios with completely novel failure modes not present in training data

## Training Data

### Dataset
NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) Dataset.

### Variations
- **FD001**: Single operating condition, single failure mode (HPC degradation)
- **FD002**: Six operating conditions, single failure mode
- **FD003**: Single operating condition, two failure modes
- **FD004**: Six operating conditions, two failure modes

### Inputs (Features)
- 3 Operational settings
- 21 Sensor measurements (Temperature, Pressure, Fan Speed, etc.)
- Engineered features: Rolling statistics, Rate of change, Health indicators

### Targets (Labels)
- Remaining Useful Life (RUL) in flight cycles.
- RUL is clipped at 125 cycles (piece-wise linear degradation assumption).

## Model Architecture (v2.0)

| Component | Description |
|-----------|-------------|
| LSTM | 2-layer LSTM with dropout regularization |
| Transformer | Multi-head attention encoder for temporal patterns |
| XGBoost | Gradient-boosted trees for tabular features |
| LightGBM | Light gradient-boosted trees for fast inference |
| Stacking Ensemble | Meta-learner combining all base models |

## Performance Metrics

Performance evaluated on FD001 test set (partial trajectories):

| Metric | Description | Target Value |
|--------|-------------|--------------|
| RMSE | Root Mean Squared Error (cycles) | ≤ 25 |
| MAE | Mean Absolute Error (cycles) | ≤ 20 |
| R² | Coefficient of Determination | ≥ 0.7 |
| Score | NASA's Asymmetric Scoring Function | Lower is better |

The asymmetric score penalizes late predictions (estimating higher RUL than truth) more heavily than early predictions, as missing a failure is costlier/riskier than premature maintenance.

## Advanced Features (v2.0)

- **Uncertainty Quantification**: Monte Carlo Dropout confidence intervals
- **Survival Analysis**: Kaplan-Meier and Cox Proportional Hazards
- **Digital Twin Simulation**: Engine degradation simulation
- **Operational Envelope Analysis**: Operating limits monitoring
- **Engine Similarity Search**: DTW-based fleet matching
- **Fleet Risk Assessment**: Monte Carlo failure simulation
- **Cost Optimization**: Pareto multi-objective maintenance scheduling
- **Drift Detection**: PSI/KS-based model monitoring

## Limitations

1. **Simulated Data**: Trained on simulated data which may not capture all complexities of real-world engines.
2. **Fixed Failure Modes**: Can only predict degradation patterns seen during training.
3. **Data Drift**: Performance may degrade if engine operating profile changes significantly over time.
4. **Assumption of Smooth Degradation**: Assumes gradual degradation; sudden catastrophic failures are not modeled.

## Ethical Considerations

- **Safety**: Predictions are advisory. Critical maintenance decisions should always be verified by certified personnel.
- **Bias**: The model performs consistently across the simulated engine fleet, but applying it to real engines requires careful validation.

## Maintenance & Monitoring

The system includes:
- **Feature Drift**: PSI/KS tests with configurable sensitivity
- **Concept Drift**: Target distribution and covariate shift detection
- **Performance Monitoring**: RMSE degradation tracking

Retraining is recommended if PSI > 0.2 for key sensors or if RMSE increases by >10%.

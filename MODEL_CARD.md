# Model Card: Aircraft Engine RUL Predictor

## Model Details

- **Name**: Hybrid LSTM Engine RUL Predictor
- **Version**: 1.0.0
- **Date**: January 2026
- **Type**: Deep Learning / Time Series Regression
- **Framework**: TensorFlow/Keras & Scikit-learn
- **Architecture**: LSTM (Long Short-Term Memory) Network with dropout regularization

## Intended Use

### Primary Use Case
Predicting Remaining Useful Life (RUL) of aircraft turbofan engines based on sensor reading history. This enables predictive maintenance strategies that optimize fleet availability and reduce costs.

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

## Performance Metrics

Performance evaluated on FD001 test set (partial trajectories):

| Metric | Descriptions | Target Value |
|--------|--------------|--------------|
| RMSE | Root Mean Squared Error (cycles) | ≤ 25 |
| MAE | Mean Absolute Error (cycles) | ≤ 20 |
| R² | Coefficient of Determination | ≥ 0.7 |
| Score | NASA's Asymmetric Scoring Function | Lower is better |

The asymmetric score penalizes late predictions (estimating higher RUL than truth) more heavily than early predictions, as missing a failure is costlier/riskier than premature maintenance.

## Model Robustness

### Uncertainty Quantification
The model uses Monte Carlo Dropout at inference time to estimate prediction uncertainty (confidence intervals).
- **High uncertainty**: Indicates novel operating conditions or data drift
- **Low uncertainty**: High confidence in prediction

### Data Validation
Input data is validated for:
- Schema correctness
- Sensor range outliers
- Missing inputs

## Limitations

1. **Simulated Data**: Trained on simulated data which may not capture all complexities of real-world engines.
2. **Fixed Failure Modes**: Can only predict degradation patterns seen during training.
3. **Data Drift**: Performance may degrade if engine operating profile changes significantly over time.
4. **Assumption of Smooth Degradation**: Assumes somewhat gradual degradation; sudden catastrophic failures due to external factors are not modeled.

## Ethical Considerations

- **Safety**: Predictions are advisory. Critical maintenance decisions should always be verified by certified personnel.
- **Bias**: The model performs consistently across the specific simulated engine fleet, but applying it to real engines requires careful validation to avoid bias against specific engine vintages or operating environments.

## Maintenance & Monitoring

The system includes a Drift Monitor to detect:
- **Feature Drift**: Changes in sensor distributions using PSI/KS tests.
- **Performance Drift**: Degradation in accuracy if ground truth becomes available.

Retraining is recommended if PSI > 0.2 for key sensors or if RMSE increases by >10%.

# MLflow Guide for RUL Prediction System

## Overview

MLflow is integrated into the RUL prediction system to provide comprehensive experiment tracking, model versioning, and deployment capabilities. This guide explains how to use MLflow effectively with the project.

## Installation

MLflow is included in `requirements.txt`. If you need to install it separately:

```bash
pip install mlflow>=2.10.0
```

## Quick Start

### 1. Start MLflow UI

```bash
# From project root
mlflow ui --port 5000
```

Then open `http://localhost:5000` in your browser.

### 2. Train Models with MLflow Tracking

```bash
# Train Phase 1 models with MLflow enabled (default)
python train_phase1.py --dataset FD001

# Train without MLflow
python train_phase1.py --dataset FD001 --no-mlflow
```

## MLflow Components

### Experiment Tracking

All training runs are automatically logged to MLflow with:
- **Parameters**: Model hyperparameters (learning rate, depth, etc.)
- **Metrics**: RMSE, MAE, R² scores
- **Artifacts**: Trained models, plots, feature importance
- **Tags**: Model type, dataset name, timestamps

### Model Registry

Models are automatically registered with version control:
- `XGBoost_RUL`
- `LightGBM_RUL`
- `CatBoost_RUL`
- `Stacking_Ensemble_RUL`

## Viewing Experiments

### MLflow UI

1. **Runs Table**: Compare all training runs
   - Sort by any metric (RMSE, MAE, R²)
   - Filter by tags or parameters
   - Select runs to compare side-by-side

2. **Run Details**: Click any run to see:
   - All logged parameters
   - Metrics with history plots
   - Artifacts (models, plots)
   - System information

3. **Model Registry**: View all registered models
   - See all versions
   - Transition models to Production/Staging
   - Add descriptions and tags

### Programmatic Access

```python
from mlflow_tracker import MLflowTracker

# Initialize tracker
tracker = MLflowTracker(
    tracking_uri='file:./mlruns',
    experiment_name='RUL_Prediction'
)

# Get best run
best_run = tracker.get_best_run('rmse', maximize=False)
print(f"Best RMSE: {best_run.data.metrics['rmse']:")

# Load best model
best_model_uri = f"runs:/{best_run.info.run_id}/model"
model = tracker.load_model(best_model_uri)
```

## Comparing Runs

```python
# Compare multiple runs
run_ids = ['run_id_1', 'run_id_2', 'run_id_3']
comparison_df = tracker.compare_runs(
    run_ids=run_ids,
    metrics=['rmse', 'mae', 'r2']
)
print(comparison_df)
```

## Model Versioning

### Register a Model

Models are automatically registered during training. To manually register:

```python
# Register from specific run
model_uri = "runs:/<run_id>/model"
tracker.register_model(
    model_uri=model_uri,
    name="XGBoost_RUL_Custom",
    tags={'custom': 'true'},
    description="Custom XGBoost model with tuned hyperparameters"
)
```

### Promote Model to Production

```python
# Transition to Production stage
tracker.transition_model_stage(
    name="XGBoost_RUL",
    version=2,
    stage="Production",
    archive_existing=True  # Archive old production model
)
```

### Load Production Model

```python
# Load current production model
model = tracker.load_model("models:/XGBoost_RUL/Production")
predictions = model.predict(X_test)
```

## Model Stages

MLflow supports these stages:
- **None**: Default stage after registration
- **Staging**: For testing before production
- **Production**: Current production model
- **Archived**: Old versions

## Advanced Usage

### Custom Logging

```python
tracker.start_run(run_name="custom_experiment")

# Log parameters
tracker.log_params({
    'learning_rate': 0.01,
    'max_depth': 7
})

# Log metrics over epochs
for epoch in range(100):
    tracker.log_metric('loss', loss_value, step=epoch)

# Log artifacts
tracker.log_artifact('/path/to/plot.png')
tracker.log_dict({'config': 'value'}, 'config.json')

tracker.end_run()
```

### Batch Training Sessions

```python
import mlflow

# Train multiple models in parallel
with mlflow.start_run(run_name="batch_training"):
    for config in model_configs:
        with mlflow.start_run(run_name=f"model_{config['name']}", nested=True):
            model = train_model(config)
            mlflow.log_params(config)
            mlflow.log_metrics(evaluate(model))
```

## Remote Tracking Server

### Setup Remote Server

```bash
# Start MLflow server with backend store and artifact location
mlflow server \
    --backend-store-uri postgresql://user:password@localhost/mlflow \
    --default-artifact-root s3://my-bucket/mlflow-artifacts \
    --host 0.0.0.0 \
    --port 5000
```

### Connect Client

```python
tracker = MLflowTracker(
    tracking_uri='http://remote-server:5000',
    experiment_name='RUL_Prediction'
)
```

## Best Practices

1. **Consistent Naming**: Use clear, descriptive run names
2. **Tag Everything**: Add tags for filtering (dataset, model_type, etc.)
3. **Log Early**: Log parameters before training starts
4. **Version Control**: Always register important models
5. **Document Models**: Add descriptions to model versions
6. **Clean Up**: Archive old experiments and models

## Troubleshooting

### Issue: "Cannot find experiment"
**Solution**: Experiment is created automatically. Check tracking URI.

### Issue: "Model not found in registry"
**Solution**: Ensure model was registered during training or register manually.

### Issue: "Artifact store is not accessible"
**Solution**: Check file permissions for `./mlruns` directory.

## Integration with Deployment

### Docker Deployment

```dockerfile
# Include MLflow model serving
FROM python:3.8-slim

# Install MLflow
RUN pip install mlflow

# Copy MLflow models
COPY mlruns /app/mlruns

# Serve model
CMD ["mlflow", "models", "serve", "-m", "models:/XGBoost_RUL/Production", "-p", "8000"]
```

### API Integration

```python
# In api.py
from mlflow_tracker import MLflowTracker

tracker = MLflowTracker()
model = tracker.load_model("models:/Stacking_Ensemble_RUL/Production")

@app.post("/predict")
async def predict(data: EngineData):
    features = prepare_features(data)
    prediction = model.predict(features)
    return {"rul": float(prediction[0])}
```

## Further Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Model Registry Guide](https://mlflow.org/docs/latest/model-registry.html)
- [MLflow Tracking API](https://mlflow.org/docs/latest/tracking.html)

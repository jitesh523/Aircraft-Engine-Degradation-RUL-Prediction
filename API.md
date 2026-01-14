# FastAPI RUL Prediction API

## Quick Start

### Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Start API server
python api.py

# Or with uvicorn directly
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

API will be available at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

### Running with Docker

```bash
# Build image
docker build -t rul-prediction-api .

# Run container
docker run -p 8000:8000 -v $(pwd)/models/saved:/app/models/saved:ro rul-prediction-api

# Or use Docker Compose
docker-compose up -d
```

## API Endpoints

### GET /health
Check API health status

**Response**:
```json
{
  "status": "healthy",
  "models_loaded": true,
  "available_models": ["lstm", "random_forest", "linear_regression"],
  "version": "1.0.0"
}
```

### POST /predict
Predict RUL for one or more engines

**Request**:
```json
{
  "engines": [
    {
      "unit_id": 1,
      "sensor_history": [
        {
          "time_cycle": 1,
          "setting_1": 0.0025,
          "setting_2": 0.0003,
          "setting_3": 100.0,
          "sensor_2": 642.3,
          "sensor_3": 1589.7,
          ...
        },
        // ... at least 30 timesteps
      ]
    }
  ],
  "use_ensemble": true
}
```

**Response**:
```json
{
  "predictions": [
    {
      "unit_id": 1,
      "predicted_rul": 112.5,
      "uncertainty_lower": null,
      "uncertainty_upper": null,
      "health_status": "Healthy",
      "recommended_action": "Continue routine monitoring",
      "confidence": "High"
    }
  ],
  "timestamp": "2026-01-14T23:45:00",
  "model_version": "1.0.0"
}
```

### GET /models/info
Get information about loaded models

**Response**:
```json
{
  "lstm": {
    "architecture": "2-layer LSTM with dropout",
    "sequence_length": 30,
    "num_features": 123,
    "units": [100, 50]
  },
  "ensemble": {
    "strategy": "weighted_average",
    "weights": {
      "LSTM": 0.65,
      "Random Forest": 0.25,
      "Linear Regression": 0.10
    }
  },
  "maintenance_thresholds": {
    "critical": 30,
    "warning": 80,
    "healthy": 80
  }
}
```

## Example Usage

### Python Client

```python
import requests

# Prepare sensor data (30+ timesteps)
engine_data = {
    "engines": [{
        "unit_id": 1,
        "sensor_history": [
            {
                "time_cycle": i,
                "setting_1": 0.0025,
                "setting_2": 0.0003,
                "setting_3": 100.0,
                "sensor_2": 642.3,
                "sensor_3": 1589.7,
                # ... all required sensors
            }
            for i in range(1, 31)  # 30 timesteps
        ]
    }],
    "use_ensemble": True
}

# Make prediction
response = requests.post(
    "http://localhost:8000/predict",
    json=engine_data
)

result = response.json()
print(f"Predicted RUL: {result['predictions'][0]['predicted_rul']:.1f} cycles")
print(f"Health Status: {result['predictions'][0]['health_status']}")
```

### cURL

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

## Required Sensor Fields

Each timestep must include:
- `time_cycle`: Cycle number
- `setting_1`, `setting_2`, `setting_3`: Operational settings
- `sensor_2`, `sensor_3`, `sensor_4`, `sensor_7`, `sensor_8`, `sensor_9`
- `sensor_11`, `sensor_12`, `sensor_13`, `sensor_14`, `sensor_15`
- `sensor_17`, `sensor_20`, `sensor_21`

Dropped sensors (1, 5, 10, 16, 18, 19) are automatically set to 0.

## Production Deployment

### Environment Variables

```bash
# Optional configuration
export MODEL_DIR=/path/to/models
export RESULTS_DIR=/path/to/results
export LOG_LEVEL=info
```

### Docker Deployment

```bash
# Build and deploy
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Security Notes

- API runs on port 8000 by default
- Add authentication middleware for production
- Use HTTPS in production (reverse proxy with nginx/traefik)
- Implement rate limiting for public endpoints

## Performance

- **Prediction latency**: ~100ms per engine (LSTM)
- **Throughput**: ~10 requests/second (single instance)
- **Model loading time**: ~5 seconds on startup

## Troubleshooting

**Models not loading:**
- Ensure models are trained and saved in `models/saved/`
- Check file paths in `config.py`
- View logs for detailed error messages

**Insufficient time steps error:**
- Provide at least 30 timesteps of sensor data
- LSTM requires sequence_length timesteps for prediction

**High latency:**
- Consider batch prediction for multiple engines
- Use ensemble=false for faster LSTM-only predictions
- Scale horizontally with multiple API instances

# FastAPI RUL Prediction API

## Quick Start

### Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Start API server (option 1)
make run-api

# Start API server (option 2)
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

API will be available at:
- **Root** → redirects to Swagger UI
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

### Core Endpoints

#### GET /
Redirects to `/docs` (Swagger UI).

#### GET /health
Check API health status.

**Response**:
```json
{
  "status": "healthy",
  "models_loaded": true,
  "available_models": ["lstm", "random_forest", "linear_regression"],
  "version": "1.0.0"
}
```

#### POST /predict
Predict RUL for one or more engines.

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
          "sensor_3": 1589.7
        }
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

#### GET /models/info
Get information about loaded models.

### Phase 11 Endpoints

#### POST /analyze/similarity
Find engines with similar degradation trajectories using DTW.

**Request**:
```json
{
  "engine_id": 42,
  "top_k": 5
}
```

**Response**:
```json
{
  "query_engine": 42,
  "similar_engines": [
    {"engine_id": 17, "similarity": 0.92},
    {"engine_id": 33, "similarity": 0.88}
  ]
}
```

#### POST /optimize/cost
Run Pareto multi-objective cost optimization on fleet data.

**Request**:
```json
{
  "preference": "balanced"
}
```

**Response**:
```json
{
  "recommendation": {
    "total_cost": 125000.0,
    "risk_cost": 45000.0,
    "combined_cost": 170000.0,
    "preference": "balanced"
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
export CMAPSS_DATA_DIR=/path/to/data     # Dataset location
export GEMINI_API_KEY=your-key           # For AI assistant
export LOG_LEVEL=info                     # Logging level
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

## Quick curl Cheat Sheet

```bash
# Health check
curl http://localhost:8000/health

# Get model info
curl http://localhost:8000/models/info

# Predict (from file)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @sample_request.json

# Find similar engines
curl -X POST http://localhost:8000/analyze/similarity \
  -H "Content-Type: application/json" \
  -d '{"engine_id": 42, "top_k": 5}'

# Cost optimization
curl -X POST http://localhost:8000/optimize/cost \
  -H "Content-Type: application/json" \
  -d '{"preference": "balanced"}'
```

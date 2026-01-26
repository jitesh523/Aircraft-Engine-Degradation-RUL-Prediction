"""
FastAPI REST API for RUL Prediction
Provides HTTP endpoints for making predictions and getting model info
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from datetime import datetime
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from utils import setup_logging, load_model, load_scaler, load_results
from preprocessor import CMAPSSPreprocessor
from feature_engineer import FeatureEngineer
from models.lstm_model import LSTMModel
from models.baseline_model import BaselineModel
from ensemble_predictor import EnsemblePredictor
from ensemble_predictor import EnsemblePredictor
from maintenance_planner import MaintenancePlanner
from uncertainty_quantifier import UncertaintyQuantifier

logger = setup_logging(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Aircraft Engine RUL Prediction API",
    description="Predict Remaining Useful Life (RUL) of turbofan engines using deep learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global model storage
models = {}
preprocessor = None
feature_engineer = None
ensemble = None
ensemble = None
planner = None
uncertainty_quantifier = None


class SensorData(BaseModel):
    """Sensor readings for a single time step"""
    time_cycle: int = Field(..., description="Time cycle number", ge=1)
    setting_1: float = Field(..., description="Operational setting 1")
    setting_2: float = Field(..., description="Operational setting 2")
    setting_3: float = Field(..., description="Operational setting 3")
    sensor_2: float = Field(..., description="Total temperature at fan inlet (°R)")
    sensor_3: float = Field(..., description="Total temperature at LPC outlet (°R)")
    sensor_4: float = Field(..., description="Total temperature at HPC outlet (°R)")
    sensor_7: float = Field(..., description="Total pressure at HPC outlet (psia)")
    sensor_8: float = Field(..., description="Physical fan speed (rpm)")
    sensor_9: float = Field(..., description="Physical core speed (rpm)")
    sensor_11: float = Field(..., description="Static pressure at HPC outlet (psia)")
    sensor_12: float = Field(..., description="Ratio of fuel flow to Ps30 (pps/psia)")
    sensor_13: float = Field(..., description="Corrected fan speed (rpm)")
    sensor_14: float = Field(..., description="Corrected core speed (rpm)")
    sensor_15: float = Field(..., description="Bypass Ratio")
    sensor_17: float = Field(..., description="Bleed Enthalpy")
    sensor_20: float = Field(..., description="HPT coolant bleed (lbm/s)")
    sensor_21: float = Field(..., description="LPT coolant bleed (lbm/s)")


class EngineData(BaseModel):
    """Time series data for a single engine"""
    unit_id: int = Field(..., description="Engine unit ID", ge=1)
    sensor_history: List[SensorData] = Field(
        ..., 
        description="Time series of sensor readings (at least 30 time steps for LSTM)",
        min_items=30
    )


class PredictionRequest(BaseModel):
    """Request model for RUL prediction"""
    engines: List[EngineData] = Field(..., description="List of engines to predict")
    use_ensemble: bool = Field(True, description="Use ensemble prediction (recommended)")


class RULPrediction(BaseModel):
    """RUL prediction for a single engine"""
    unit_id: int
    predicted_rul: float = Field(..., description="Predicted RUL in cycles")
    uncertainty_lower: Optional[float] = Field(None, description="Lower bound of 95% CI")
    uncertainty_upper: Optional[float] = Field(None, description="Upper bound of 95% CI")
    health_status: str = Field(..., description="Health status: Critical/Warning/Healthy")
    recommended_action: str
    confidence: str = Field(..., description="Prediction confidence: Low/Medium/High")


class PredictionResponse(BaseModel):
    """Response model for RUL prediction"""
    predictions: List[RULPrediction]
    timestamp: str
    model_version: str


class HealthStatus(BaseModel):
    """API health status"""
    status: str
    models_loaded: bool
    available_models: List[str]
    version: str


@app.on_event("startup")
async def load_models():
    """Load trained models on startup"""
    global models, preprocessor, feature_engineer, ensemble, planner, uncertainty_quantifier
    
    logger.info("Loading models...")
    
    try:
        # Load feature info
        feature_info = load_results(os.path.join(config.MODELS_DIR, 'feature_info.json'))
        
        # Load preprocessor
        preprocessor = CMAPSSPreprocessor()
        preprocessor.load_scaler(os.path.join(config.MODELS_DIR, 'scaler.pkl'))
        
        # Load LSTM model
        lstm_model = LSTMModel()
        lstm_model.load(os.path.join(config.MODELS_DIR, 'lstm_model.h5'))
        
        # Load baseline models
        rf_model = BaselineModel('random_forest')
        rf_model.load(os.path.join(config.MODELS_DIR, 'baseline_rf.pkl'))
        
        lr_model = BaselineModel('linear_regression')
        lr_model.load(os.path.join(config.MODELS_DIR, 'baseline_lr.pkl'))
        
        # Initialize feature engineer
        feature_engineer = FeatureEngineer()
        
        # Initialize ensemble
        ensemble = EnsemblePredictor('weighted_average')
        
        # Load ensemble weights if available
        try:
            weights = load_results(os.path.join(config.RESULTS_DIR, 'FD001_ensemble_weights.json'))
            ensemble.weights = weights
        except:
            logger.warning("Ensemble weights not found, using equal weights")
            ensemble.weights = {'LSTM': 0.6, 'Random Forest': 0.3, 'Linear Regression': 0.1}
        
        # Initialize maintenance planner
        planner = MaintenancePlanner()

        # Initialize uncertainty quantifier
        uncertainty_quantifier = UncertaintyQuantifier(n_iterations=50) # Use 50 for faster inference
        
        models = {
            'lstm': lstm_model,
            'random_forest': rf_model,
            'linear_regression': lr_model,
            'feature_columns': feature_info['feature_columns'],
            'sequence_length': feature_info['sequence_length']
        }
        
        logger.info("✅ Models loaded successfully!")
        
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        logger.warning("API will run in degraded mode")


@app.get("/", response_class=JSONResponse)
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Aircraft Engine RUL Prediction API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health_check": "/health"
    }


@app.get("/health", response_model=HealthStatus)
async def health_check():
    """Check API health and model availability"""
    models_loaded = models is not None and len(models) > 0
    available_models = list(models.keys()) if models_loaded else []
    
    return HealthStatus(
        status="healthy" if models_loaded else "degraded",
        models_loaded=models_loaded,
        available_models=available_models,
        version="1.0.0"
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_rul(request: PredictionRequest):
    """
    Predict RUL for one or more engines
    
    Requires at least 30 time steps of sensor data per engine for LSTM prediction.
    """
    if not models:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not loaded. Please check server logs."
        )
    
    try:
        predictions = []
        
        for engine_data in request.engines:
            # Convert sensor history to DataFrame
            sensor_df = pd.DataFrame([s.dict() for s in engine_data.sensor_history])
            sensor_df['unit_id'] = engine_data.unit_id
            
            # Ensure required columns
            required_cols = ['unit_id', 'time_cycle', 'setting_1', 'setting_2', 'setting_3'] + \
                           [f'sensor_{i}' for i in [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]]
            
            # Add missing sensors (dropped ones) as zeros
            for col in config.COLUMN_NAMES:
                if col not in sensor_df.columns and col != 'RUL':
                    sensor_df[col] = 0
            
            # Preprocess
            normalized_df, _ = preprocessor.normalize_features(sensor_df, fit=False)
            
            # Engineer features
            engineered_df = feature_engineer.create_all_features(normalized_df)
            
            # Get last sequence_length timesteps
            sequence_length = models['sequence_length']
            feature_cols = models['feature_columns']
            
            if len(engineered_df) < sequence_length:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Engine {engine_data.unit_id} has insufficient data. "
                           f"Need at least {sequence_length} time steps, got {len(engineered_df)}"
                )
            
            # Create sequence
            X_seq = engineered_df.tail(sequence_length)[feature_cols].values
            X_seq = X_seq.reshape(1, sequence_length, len(feature_cols))
            
            # Predict with LSTM
            lstm_pred = models['lstm'].predict(X_seq)[0]
            
            # Predict with baselines (use last timestep)
            X_flat = engineered_df.tail(1)[feature_cols].values
            rf_pred = models['random_forest'].predict(X_flat)[0]
            lr_pred = models['linear_regression'].predict(X_flat)[0]
            
            # Ensemble prediction
            if request.use_ensemble:
                all_preds = {
                    'LSTM': np.array([lstm_pred]),
                    'Random Forest': np.array([rf_pred]),
                    'Linear Regression': np.array([lr_pred])
                }
                final_pred = ensemble.predict(all_preds)[0]
            else:
                final_pred = lstm_pred
            
            # Get health status
            health_status = planner.classify_health_status(final_pred)
            
            # Determine action
            if health_status == 'Critical':
                action = 'Immediate maintenance required - Ground aircraft'
            elif health_status == 'Warning':
                action = 'Schedule maintenance at next opportunity'
            else:
                action = 'Continue routine monitoring'
            
            # Determine confidence (simple heuristic based on model agreement)
            pred_std = np.std([lstm_pred, rf_pred, lr_pred])
            if pred_std < 10:
                confidence = 'High'
            elif pred_std < 20:
                confidence = 'Medium'
            else:
                confidence = 'Low'
            
            predictions.append(RULPrediction(
                unit_id=engine_data.unit_id,
                predicted_rul=float(final_pred),
                confidence = 'Low'
            
            # Calculate uncertainty estimates if using LSTM
            uncertainty_lower = None
            uncertainty_upper = None
            
            if not request.use_ensemble and models['lstm']:
                # Reshape for uncertainty quantifier
                # Note: This increases latency due to multiple forward passes
                try:
                    # We reuse the previously prepared X_seq
                    # For a real production system, you might want to make this optional via flag
                    _, lower, upper = uncertainty_quantifier.predict_with_uncertainty(
                        models['lstm'], 
                        X_seq, 
                        is_keras=True
                    )
                    uncertainty_lower = float(lower[0])
                    uncertainty_upper = float(upper[0])
                except Exception as e:
                    logger.warning(f"Uncertainty quantification failed: {e}")

            predictions.append(RULPrediction(
                unit_id=engine_data.unit_id,
                predicted_rul=float(final_pred),
                uncertainty_lower=uncertainty_lower,
                uncertainty_upper=uncertainty_upper,
                health_status=health_status,
                recommended_action=action,
                confidence=confidence
            ))
        
        return PredictionResponse(
            predictions=predictions,
            timestamp=datetime.now().isoformat(),
            model_version="1.0.0"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/models/info")
async def get_model_info():
    """Get information about loaded models"""
    if not models:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not loaded"
        )
    
    return {
        "lstm": {
            "architecture": "2-layer LSTM with dropout",
            "sequence_length": models['sequence_length'],
            "num_features": len(models['feature_columns']),
            "units": config.LSTM_CONFIG['lstm_units']
        },
        "ensemble": {
            "strategy": "weighted_average",
            "weights": ensemble.weights if ensemble else None
        },
        "maintenance_thresholds": config.MAINTENANCE_THRESHOLDS
    }


if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting RUL Prediction API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

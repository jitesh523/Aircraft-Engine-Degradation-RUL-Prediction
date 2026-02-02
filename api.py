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


class APIRequestLogger:
    """
    Comprehensive API request logging and analytics
    Tracks predictions, performance, and usage patterns
    """
    
    def __init__(self, log_dir: str = None):
        """
        Initialize API request logger
        
        Args:
            log_dir: Directory for log files
        """
        import json
        self.log_dir = log_dir or os.path.join(config.RESULTS_DIR, 'api_logs')
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.request_log = []
        self.error_log = []
        self.performance_metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_response_time_ms': 0,
            'total_engines_predicted': 0
        }
        
        # Rate limiting
        self.rate_limits = {
            'requests_per_minute': 60,
            'requests_per_hour': 1000
        }
        self.request_timestamps = []
        
        logger.info(f"Initialized APIRequestLogger (log_dir: {self.log_dir})")
    
    def log_request(self,
                    endpoint: str,
                    method: str,
                    request_data: Dict,
                    response_time_ms: float,
                    success: bool,
                    response_data: Dict = None,
                    error: str = None):
        """
        Log an API request
        
        Args:
            endpoint: API endpoint called
            method: HTTP method
            request_data: Request payload
            response_time_ms: Response time in milliseconds
            success: Whether request succeeded
            response_data: Response data (optional)
            error: Error message if failed (optional)
        """
        import time
        
        timestamp = datetime.now()
        
        log_entry = {
            'timestamp': timestamp.isoformat(),
            'endpoint': endpoint,
            'method': method,
            'response_time_ms': response_time_ms,
            'success': success,
            'engines_count': len(request_data.get('engines', [])) if request_data else 0
        }
        
        self.request_log.append(log_entry)
        self.request_timestamps.append(time.time())
        
        # Update metrics
        self.performance_metrics['total_requests'] += 1
        
        if success:
            self.performance_metrics['successful_requests'] += 1
            self.performance_metrics['total_engines_predicted'] += log_entry['engines_count']
        else:
            self.performance_metrics['failed_requests'] += 1
            self.error_log.append({
                'timestamp': timestamp.isoformat(),
                'endpoint': endpoint,
                'error': error
            })
        
        # Update average response time
        total = self.performance_metrics['total_requests']
        old_avg = self.performance_metrics['avg_response_time_ms']
        self.performance_metrics['avg_response_time_ms'] = (
            (old_avg * (total - 1) + response_time_ms) / total
        )
    
    def check_rate_limit(self, client_id: str = 'default') -> tuple:
        """
        Check if request should be rate limited
        
        Args:
            client_id: Client identifier
            
        Returns:
            (is_allowed, wait_seconds)
        """
        import time
        
        current_time = time.time()
        
        # Clean old timestamps
        minute_ago = current_time - 60
        hour_ago = current_time - 3600
        
        self.request_timestamps = [t for t in self.request_timestamps if t > hour_ago]
        
        # Check per-minute limit
        recent_minute = [t for t in self.request_timestamps if t > minute_ago]
        if len(recent_minute) >= self.rate_limits['requests_per_minute']:
            wait_time = 60 - (current_time - recent_minute[0])
            return False, wait_time
        
        # Check per-hour limit
        if len(self.request_timestamps) >= self.rate_limits['requests_per_hour']:
            wait_time = 3600 - (current_time - self.request_timestamps[0])
            return False, wait_time
        
        return True, 0
    
    def get_analytics(self, time_period: str = 'all') -> Dict:
        """
        Get usage analytics
        
        Args:
            time_period: 'hour', 'day', 'week', or 'all'
            
        Returns:
            Analytics dictionary
        """
        if not self.request_log:
            return {'status': 'no_data'}
        
        # Filter by time period
        now = datetime.now()
        
        if time_period == 'hour':
            cutoff = now.timestamp() - 3600
        elif time_period == 'day':
            cutoff = now.timestamp() - 86400
        elif time_period == 'week':
            cutoff = now.timestamp() - 604800
        else:
            cutoff = 0
        
        filtered_logs = [
            log for log in self.request_log
            if datetime.fromisoformat(log['timestamp']).timestamp() > cutoff
        ]
        
        if not filtered_logs:
            return {'status': 'no_data', 'time_period': time_period}
        
        # Calculate analytics
        total = len(filtered_logs)
        successful = sum(1 for log in filtered_logs if log['success'])
        total_engines = sum(log['engines_count'] for log in filtered_logs)
        avg_response = sum(log['response_time_ms'] for log in filtered_logs) / total
        
        # Response time distribution
        response_times = [log['response_time_ms'] for log in filtered_logs]
        
        return {
            'time_period': time_period,
            'total_requests': total,
            'successful_requests': successful,
            'failed_requests': total - successful,
            'success_rate': successful / total * 100,
            'total_engines_predicted': total_engines,
            'avg_engines_per_request': total_engines / total,
            'response_time': {
                'avg_ms': avg_response,
                'min_ms': min(response_times),
                'max_ms': max(response_times),
                'p50_ms': sorted(response_times)[len(response_times)//2],
                'p95_ms': sorted(response_times)[int(len(response_times)*0.95)] if len(response_times) > 20 else max(response_times)
            },
            'errors': len(self.error_log)
        }
    
    def get_error_summary(self) -> Dict:
        """Get summary of recent errors"""
        if not self.error_log:
            return {'total_errors': 0}
        
        # Group by error type
        error_counts = {}
        for error in self.error_log:
            error_type = error.get('error', 'Unknown')[:50]
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        return {
            'total_errors': len(self.error_log),
            'error_types': error_counts,
            'recent_errors': self.error_log[-5:]
        }
    
    def save_logs(self, filename: str = None):
        """
        Save logs to file
        
        Args:
            filename: Output filename
        """
        import json
        
        if filename is None:
            filename = f"api_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = os.path.join(self.log_dir, filename)
        
        data = {
            'saved_at': datetime.now().isoformat(),
            'performance_metrics': self.performance_metrics,
            'request_log': self.request_log[-1000:],  # Keep last 1000
            'error_log': self.error_log[-100:]  # Keep last 100
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Logs saved to {filepath}")
        return filepath
    
    def generate_report(self) -> str:
        """Generate formatted usage report"""
        analytics = self.get_analytics('day')
        errors = self.get_error_summary()
        
        lines = [
            "=" * 60,
            "API USAGE REPORT",
            "=" * 60,
            "",
            f"Total Requests: {self.performance_metrics['total_requests']}",
            f"Success Rate: {self.performance_metrics['successful_requests'] / max(1, self.performance_metrics['total_requests']) * 100:.1f}%",
            f"Avg Response Time: {self.performance_metrics['avg_response_time_ms']:.1f}ms",
            f"Total Engines Predicted: {self.performance_metrics['total_engines_predicted']}",
            "",
            "Last 24 Hours:",
            f"  Requests: {analytics.get('total_requests', 0)}",
            f"  Engines Predicted: {analytics.get('total_engines_predicted', 0)}",
            "",
            f"Errors: {errors['total_errors']}",
            "=" * 60
        ]
        
        return '\n'.join(lines)


class HealthCheckMonitor:
    """
    System health monitoring for production deployment
    Checks memory, disk, GPU, and model readiness
    """
    
    def __init__(self):
        """Initialize health check monitor"""
        self.last_check = None
        self.check_history = []
        self.thresholds = {
            'memory_percent': 90,  # Warning if > 90%
            'disk_percent': 90,    # Warning if > 90%
            'gpu_memory_percent': 95
        }
        logger.info("Initialized HealthCheckMonitor")
    
    def check_memory(self) -> dict:
        """Check system memory usage"""
        try:
            import psutil
            
            memory = psutil.virtual_memory()
            
            return {
                'status': 'healthy' if memory.percent < self.thresholds['memory_percent'] else 'warning',
                'total_gb': round(memory.total / (1024**3), 2),
                'used_gb': round(memory.used / (1024**3), 2),
                'available_gb': round(memory.available / (1024**3), 2),
                'percent': memory.percent
            }
        except ImportError:
            return {
                'status': 'unknown',
                'error': 'psutil not installed'
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def check_disk(self, path: str = '/') -> dict:
        """Check disk usage"""
        try:
            import psutil
            
            disk = psutil.disk_usage(path)
            
            return {
                'status': 'healthy' if disk.percent < self.thresholds['disk_percent'] else 'warning',
                'path': path,
                'total_gb': round(disk.total / (1024**3), 2),
                'used_gb': round(disk.used / (1024**3), 2),
                'free_gb': round(disk.free / (1024**3), 2),
                'percent': disk.percent
            }
        except ImportError:
            return {
                'status': 'unknown',
                'error': 'psutil not installed'
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def check_gpu(self) -> dict:
        """Check GPU availability and memory"""
        try:
            import tensorflow as tf
            
            gpus = tf.config.list_physical_devices('GPU')
            
            if not gpus:
                return {
                    'status': 'not_available',
                    'gpus': 0
                }
            
            return {
                'status': 'available',
                'gpus': len(gpus),
                'devices': [gpu.name for gpu in gpus]
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def check_model_readiness(self) -> dict:
        """Check if models are loaded and ready"""
        global models
        
        loaded_models = list(models.keys()) if models else []
        
        return {
            'status': 'ready' if loaded_models else 'not_ready',
            'loaded_models': loaded_models,
            'model_count': len(loaded_models)
        }
    
    def check_dependencies(self) -> dict:
        """Check critical dependencies"""
        dependencies = {}
        
        required_packages = [
            'numpy', 'pandas', 'sklearn', 'tensorflow',
            'fastapi', 'pydantic'
        ]
        
        for package in required_packages:
            try:
                __import__(package)
                dependencies[package] = {'status': 'installed'}
            except ImportError:
                dependencies[package] = {'status': 'missing'}
        
        all_installed = all(d['status'] == 'installed' for d in dependencies.values())
        
        return {
            'status': 'healthy' if all_installed else 'degraded',
            'packages': dependencies
        }
    
    def run_full_check(self) -> dict:
        """Run all health checks"""
        self.last_check = datetime.now()
        
        check_result = {
            'timestamp': self.last_check.isoformat(),
            'memory': self.check_memory(),
            'disk': self.check_disk(),
            'gpu': self.check_gpu(),
            'models': self.check_model_readiness(),
            'dependencies': self.check_dependencies()
        }
        
        # Determine overall status
        statuses = [
            check_result['memory'].get('status', 'unknown'),
            check_result['disk'].get('status', 'unknown'),
            check_result['models'].get('status', 'unknown'),
            check_result['dependencies'].get('status', 'unknown')
        ]
        
        if 'error' in statuses or 'not_ready' in statuses:
            overall = 'unhealthy'
        elif 'warning' in statuses or 'degraded' in statuses:
            overall = 'degraded'
        else:
            overall = 'healthy'
        
        check_result['overall_status'] = overall
        
        # Store in history
        self.check_history.append({
            'timestamp': self.last_check.isoformat(),
            'status': overall
        })
        
        # Keep last 100 checks
        self.check_history = self.check_history[-100:]
        
        logger.info(f"Health check completed: {overall}")
        
        return check_result
    
    def get_quick_status(self) -> dict:
        """Get quick health status"""
        return {
            'status': self.check_history[-1]['status'] if self.check_history else 'unknown',
            'last_check': self.last_check.isoformat() if self.last_check else None,
            'models_ready': self.check_model_readiness()['status'] == 'ready'
        }
    
    def get_health_summary(self) -> str:
        """Generate health summary report"""
        check = self.run_full_check()
        
        lines = [
            "=" * 60,
            "SYSTEM HEALTH SUMMARY",
            "=" * 60,
            f"Overall Status: {check['overall_status'].upper()}",
            f"Timestamp: {check['timestamp']}",
            "",
            f"Memory: {check['memory'].get('status', 'unknown')} "
            f"({check['memory'].get('percent', 'N/A')}% used)",
            f"Disk: {check['disk'].get('status', 'unknown')} "
            f"({check['disk'].get('percent', 'N/A')}% used)",
            f"GPU: {check['gpu'].get('status', 'unknown')}",
            f"Models: {check['models'].get('status', 'unknown')} "
            f"({check['models'].get('model_count', 0)} loaded)",
            f"Dependencies: {check['dependencies'].get('status', 'unknown')}",
            "=" * 60
        ]
        
        return '\n'.join(lines)


if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting RUL Prediction API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

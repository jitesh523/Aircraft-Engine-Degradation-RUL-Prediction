"""
Tests for MLflow Integration
"""

import pytest
import numpy as np
import mlflow
from mlflow_tracker import MLflowTracker
from pathlib import Path
import shutil


@pytest.fixture(scope="function")
def mlflow_tracker(tmp_path):
    """Create MLflow tracker with temporary directory"""
    tracking_uri = f"file:{tmp_path}/mlruns"
    tracker = MLflowTracker(
        tracking_uri=tracking_uri,
        experiment_name="test_experiment"
    )
    yield tracker
    # Cleanup
    if (tmp_path / "mlruns").exists():
        shutil.rmtree(tmp_path / "mlruns")


class TestMLflowTracker:
    """Test MLflow tracker functionality"""
    
    def test_initialization(self, mlflow_tracker):
        """Test tracker initialization"""
        assert mlflow_tracker.experiment_name == "test_experiment"
        assert mlflow_tracker.client is not None
    
    def test_start_end_run(self, mlflow_tracker):
        """Test starting and ending runs"""
        run = mlflow_tracker.start_run(run_name="test_run")
        assert mlflow_tracker.active_run is not None
        
        mlflow_tracker.end_run()
        assert mlflow_tracker.active_run is None
    
    def test_log_params(self, mlflow_tracker):
        """Test logging parameters"""
        mlflow_tracker.start_run()
        params = {
            'learning_rate': 0.01,
            'n_estimators': 100,
            'nested': {'key': 'value'}
        }
        mlflow_tracker.log_params(params)
        mlflow_tracker.end_run()
    
    def test_log_metrics(self, mlflow_tracker):
        """Test logging metrics"""
        mlflow_tracker.start_run()
        metrics = {
            'rmse': 20.5,
            'mae': 15.3,
            'r2': 0.85
        }
        mlflow_tracker.log_metrics(metrics)
        mlflow_tracker.end_run()
    
    def test_log_metric_with_step(self, mlflow_tracker):
        """Test logging metrics with steps"""
        mlflow_tracker.start_run()
        for step in range(5):
            mlflow_tracker.log_metric('loss', 100.0 / (step + 1), step=step)
        mlflow_tracker.end_run()
    
    def test_log_dict(self, mlflow_tracker):
        """Test logging dictionary"""
        mlflow_tracker.start_run()
        test_dict = {'key1': 'value1', 'key2': 42}
        mlflow_tracker.log_dict(test_dict, 'test_dict.json')
        mlflow_tracker.end_run()
    
    def test_set_tags(self, mlflow_tracker):
        """Test setting tags"""
        mlflow_tracker.start_run()
        tags = {
            'model_type': 'xgboost',
            'dataset': 'FD001'
        }
        mlflow_tracker.set_tags(tags)
        mlflow_tracker.end_run()
    
    def test_log_training_session(self, mlflow_tracker):
        """Test complete training session logging"""
        from sklearn.ensemble import RandomForestRegressor
        
        # Create dummy model
        model = RandomForestRegressor(n_estimators=10)
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        model.fit(X, y)
        
        params = {'n_estimators': 10, 'max_depth': 5}
        metrics = {'rmse': 20.0, 'mae': 15.0, 'r2': 0.8}
        
        run_id = mlflow_tracker.log_training_session(
            model=model,
            model_type='sklearn',
            params=params,
            metrics=metrics
        )
        
        assert run_id is not None
    
    def test_error_without_active_run(self, mlflow_tracker):
        """Test that operations fail without active run"""
        with pytest.raises(RuntimeError):
            mlflow_tracker.log_params({'key': 'value'})
        
        with pytest.raises(RuntimeError):
            mlflow_tracker.log_metrics({'metric': 1.0})

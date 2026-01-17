"""
MLflow Integration for RUL Prediction System
Handles experiment tracking, model logging, and versioning
"""

import mlflow
import mlflow.sklearn
import mlflow.tensorflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import numpy as np
import pandas as pd
from datetime import datetime


class MLflowTracker:
    """MLflow experiment tracking and model management"""
    
    def __init__(self, 
                 tracking_uri: str = "file:./mlruns",
                 experiment_name: str = "RUL_Prediction",
                 artifact_location: Optional[str] = None):
        """
        Initialize MLflow tracker
        
        Args:
            tracking_uri: MLflow tracking server URI
            experiment_name: Name of the experiment
            artifact_location: Optional custom artifact storage location
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        
        # Set tracking URI
        mlflow.set_tracking_uri(tracking_uri)
        
        # Create or get experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    experiment_name,
                    artifact_location=artifact_location
                )
            else:
                experiment_id = experiment.experiment_id
            mlflow.set_experiment(experiment_name)
            self.experiment_id = experiment_id
        except Exception as e:
            print(f"Warning: Could not set experiment: {e}")
            self.experiment_id = None
        
        self.client = MlflowClient()
        self.active_run = None
    
    def start_run(self, 
                  run_name: Optional[str] = None,
                  tags: Optional[Dict[str, str]] = None,
                  description: Optional[str] = None) -> Any:
        """
        Start a new MLflow run
        
        Args:
            run_name: Name for the run
            tags: Dictionary of tags
            description: Run description
            
        Returns:
            Active run object
        """
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        all_tags = tags or {}
        if description:
            all_tags['mlflow.note.content'] = description
        
        self.active_run = mlflow.start_run(run_name=run_name, tags=all_tags)
        return self.active_run
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters to active run"""
        if self.active_run is None:
            raise RuntimeError("No active run. Call start_run() first.")
        
        # Flatten nested dictionaries
        flat_params = self._flatten_dict(params)
        mlflow.log_params(flat_params)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to active run"""
        if self.active_run is None:
            raise RuntimeError("No active run. Call start_run() first.")
        
        mlflow.log_metrics(metrics, step=step)
    
    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """Log a single metric"""
        if self.active_run is None:
            raise RuntimeError("No active run. Call start_run() first.")
        
        mlflow.log_metric(key, value, step=step)
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log an artifact file"""
        if self.active_run is None:
            raise RuntimeError("No active run. Call start_run() first.")
        
        mlflow.log_artifact(local_path, artifact_path)
    
    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None):
        """Log directory of artifacts"""
        if self.active_run is None:
            raise RuntimeError("No active run. Call start_run() first.")
        
        mlflow.log_artifacts(local_dir, artifact_path)
    
    def log_dict(self, dictionary: Dict, filename: str):
        """Log a dictionary as JSON artifact"""
        if self.active_run is None:
            raise RuntimeError("No active run. Call start_run() first.")
        
        mlflow.log_dict(dictionary, filename)
    
    def log_figure(self, figure, filename: str):
        """Log matplotlib figure"""
        if self.active_run is None:
            raise RuntimeError("No active run. Call start_run() first.")
        
        mlflow.log_figure(figure, filename)
    
    def log_sklearn_model(self, 
                          model,
                          artifact_path: str = "model",
                          registered_model_name: Optional[str] = None,
                          signature=None,
                          input_example=None):
        """
        Log scikit-learn compatible model
        
        Args:
            model: The model object
            artifact_path: Path within run artifacts
            registered_model_name: If provided, register model with this name
            signature: Model signature
            input_example: Example input for the model
        """
        if self.active_run is None:
            raise RuntimeError("No active run. Call start_run() first.")
        
        mlflow.sklearn.log_model(
            model,
            artifact_path=artifact_path,
            registered_model_name=registered_model_name,
            signature=signature,
            input_example=input_example
        )
    
    def log_tensorflow_model(self,
                            model,
                            artifact_path: str = "model",
                            registered_model_name: Optional[str] = None,
                            signature=None,
                            input_example=None):
        """
        Log TensorFlow/Keras model
        
        Args:
            model: The model object
            artifact_path: Path within run artifacts
            registered_model_name: If provided, register model with this name
            signature: Model signature
            input_example: Example input for the model
        """
        if self.active_run is None:
            raise RuntimeError("No active run. Call start_run() first.")
        
        mlflow.tensorflow.log_model(
            model,
            artifact_path=artifact_path,
            registered_model_name=registered_model_name,
            signature=signature,
            input_example=input_example
        )
    
    def end_run(self, status: str = "FINISHED"):
        """
        End the active run
        
        Args:
            status: Run status (FINISHED, FAILED, KILLED)
        """
        if self.active_run is not None:
            mlflow.end_run(status=status)
            self.active_run = None
    
    def register_model(self, 
                       model_uri: str,
                       name: str,
                       tags: Optional[Dict[str, str]] = None,
                       description: Optional[str] = None) -> Any:
        """
        Register a model in the MLflow Model Registry
        
        Args:
            model_uri: URI of the model (e.g., "runs:/<run_id>/model")
            name: Name to register the model under
            tags: Optional tags for the model version
            description: Optional description
            
        Returns:
            ModelVersion object
        """
        model_version = mlflow.register_model(model_uri, name)
        
        if tags:
            for key, value in tags.items():
                self.client.set_model_version_tag(name, model_version.version, key, value)
        
        if description:
            self.client.update_model_version(
                name=name,
                version=model_version.version,
                description=description
            )
        
        return model_version
    
    def transition_model_stage(self,
                              name: str,
                              version: int,
                              stage: str,
                              archive_existing: bool = False):
        """
        Transition a model version to a new stage
        
        Args:
            name: Registered model name
            version: Model version number
            stage: Target stage (Staging, Production, Archived)
            archive_existing: Archive existing Production models
        """
        self.client.transition_model_version_stage(
            name=name,
            version=version,
            stage=stage,
            archive_existing_versions=archive_existing
        )
    
    def load_model(self, model_uri: str):
        """
        Load a model from MLflow
        
        Args:
            model_uri: Model URI (e.g., "models:/ModelName/Production")
            
        Returns:
            Loaded model
        """
        return mlflow.pyfunc.load_model(model_uri)
    
    def get_best_run(self, 
                     metric_name: str,
                     maximize: bool = True,
                     filter_string: Optional[str] = None) -> Any:
        """
        Get the best run based on a metric
        
        Args:
            metric_name: Metric to optimize
            maximize: If True, get max value; if False, get min
            filter_string: Optional filter for runs
            
        Returns:
            Best run object
        """
        runs = self.client.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string=filter_string,
            order_by=[f"metrics.{metric_name} {'DESC' if maximize else 'ASC'}"],
            max_results=1
        )
        
        return runs[0] if runs else None
    
    def compare_runs(self, run_ids: List[str], metrics: List[str]) -> pd.DataFrame:
        """
        Compare multiple runs
        
        Args:
            run_ids: List of run IDs to compare
            metrics: List of metric names to compare
            
        Returns:
            DataFrame with comparison results
        """
        comparison_data = []
        
        for run_id in run_ids:
            run = self.client.get_run(run_id)
            row = {
                'run_id': run_id,
                'run_name': run.data.tags.get('mlflow.runName', ''),
                'start_time': run.info.start_time
            }
            
            for metric in metrics:
                row[metric] = run.data.metrics.get(metric, None)
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def get_model_versions(self, model_name: str) -> List[Any]:
        """Get all versions of a registered model"""
        try:
            return self.client.search_model_versions(f"name='{model_name}'")
        except Exception as e:
            print(f"Error getting model versions: {e}")
            return []
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """Flatten nested dictionary for MLflow logging"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                # Convert to string if not a basic type
                if not isinstance(v, (str, int, float, bool)):
                    v = str(v)
                items.append((new_key, v))
        return dict(items)
    
    def set_tags(self, tags: Dict[str, str]):
        """Set tags for the active run"""
        if self.active_run is None:
            raise RuntimeError("No active run. Call start_run() first.")
        
        mlflow.set_tags(tags)
    
    def log_training_session(self,
                            model,
                            model_type: str,
                            params: Dict[str, Any],
                            metrics: Dict[str, float],
                            artifacts: Optional[Dict[str, str]] = None,
                            model_name: Optional[str] = None,
                            input_example=None,
                            signature=None) -> str:
        """
        Convenience method to log a complete training session
        
        Args:
            model: Trained model
            model_type: Type of model ('sklearn', 'tensorflow', etc.)
            params: Training parameters
            metrics: Evaluation metrics
            artifacts: Dict of {artifact_path: local_path}
            model_name: Name to register the model
            input_example: Example input for model
            signature: Model signature
            
        Returns:
            Run ID
        """
        run = self.start_run(
            run_name=f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            tags={'model_type': model_type}
        )
        
        try:
            # Log parameters
            self.log_params(params)
            
            # Log metrics
            self.log_metrics(metrics)
            
            # Log model
            if model_type == 'sklearn':
                self.log_sklearn_model(
                    model,
                    registered_model_name=model_name,
                    signature=signature,
                    input_example=input_example
                )
            elif model_type == 'tensorflow':
                self.log_tensorflow_model(
                    model,
                    registered_model_name=model_name,
                    signature=signature,
                    input_example=input_example
                )
            
            # Log artifacts
            if artifacts:
                for artifact_path, local_path in artifacts.items():
                    if os.path.isdir(local_path):
                        self.log_artifacts(local_path, artifact_path)
                    else:
                        self.log_artifact(local_path, artifact_path)
            
            self.end_run(status="FINISHED")
            return run.info.run_id
            
        except Exception as e:
            self.end_run(status="FAILED")
            raise e

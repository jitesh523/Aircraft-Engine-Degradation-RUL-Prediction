"""
Edge Inference Engine
Lightweight inference for TensorFlow Lite and ONNX models on edge devices
"""

import numpy as np
from typing import Union, Dict, Any
import os
import logging


class EdgeInferenceEngine:
    """Lightweight inference engine for edge deployment"""
    
    def __init__(self, model_path: str, model_type: str = 'tflite'):
        """
        Initialize edge inference engine
        
        Args:
            model_path: Path to model file (.tflite or .onnx)
            model_type: 'tflite' or 'onnx'
        """
        self.model_path = model_path
        self.model_type = model_type.lower()
        self.logger = logging.getLogger(__name__)
        
        if self.model_type == 'tflite':
            self._load_tflite()
        elif self.model_type == 'onnx':
            self._load_onnx()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        self.logger.info(f"Loaded {model_type} model from {model_path}")
    
    def _load_tflite(self):
        """Load TensorFlow Lite model"""
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError("TensorFlow required for TFLite models")
        
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Get input shape
        self.input_shape = self.input_details[0]['shape']
        self.output_shape = self.output_details[0]['shape']
    
    def _load_onnx(self):
        """Load ONNX model"""
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime required for ONNX models")
        
        self.session = ort.InferenceSession(self.model_path)
        
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_shape = self.session.get_outputs()[0].shape
    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        Run inference on input data
        
        Args:
            input_data: Input features (can be single sample or batch)
            
        Returns:
            Predictions
        """
        # Ensure proper shape
        if len(input_data.shape) == 1:
            input_data = input_data.reshape(1, -1)
        
        if self.model_type == 'tflite':
            return self._predict_tflite(input_data)
        else:
            return self._predict_onnx(input_data)
    
    def _predict_tflite(self, input_data: np.ndarray) -> np.ndarray:
        """TFLite inference"""
        predictions = []
        
        for i in range(len(input_data)):
            sample = input_data[i:i+1].astype(np.float32)
            self.interpreter.set_tensor(self.input_details[0]['index'], sample)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output_details[0]['index'])
            predictions.append(output[0])
        
        return np.array(predictions)
    
    def _predict_onnx(self, input_data: np.ndarray) -> np.ndarray:
        """ONNX inference"""
        input_data = input_data.astype(np.float32)
        result = self.session.run([self.output_name], {self.input_name: input_data})
        return result[0]
    
    def predict_single(self, input_features: np.ndarray) -> float:
        """
        Predict for a single sample
        
        Args:
            input_features: Feature vector
            
        Returns:
            Single prediction value
        """
        prediction = self.predict(input_features.reshape(1, -1))
        return float(prediction[0])
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        info = {
            'model_path': self.model_path,
            'model_type': self.model_type,
            'input_shape': [int(x) if x is not None else None for x in self.input_shape],
            'output_shape': [int(x) if x is not None else None for x in self.output_shape],
            'model_size_mb': os.path.getsize(self.model_path) / (1024 * 1024)
        }
        return info


class BatchPredictor:
    """Efficient batch prediction for edge devices"""
    
    def __init__(self, engine: EdgeInferenceEngine, batch_size: int = 32):
        """
        Initialize batch predictor
        
        Args:
            engine: EdgeInferenceEngine instance
            batch_size: Batch size for inference
        """
        self.engine = engine
        self.batch_size = batch_size
        self.logger = logging.getLogger(__name__)
    
    def predict_batch(self, data: np.ndarray) -> np.ndarray:
        """
        Predict in batches for efficiency
        
        Args:
            data: Input data array
            
        Returns:
            Predictions
        """
        n_samples = len(data)
        predictions = []
        
        for i in range(0, n_samples, self.batch_size):
            batch = data[i:i+self.batch_size]
            batch_preds = self.engine.predict(batch)
            predictions.extend(batch_preds)
            
            if (i + self.batch_size) % 100 == 0:
                self.logger.info(f"Processed {min(i + self.batch_size, n_samples)}/{n_samples} samples")
        
        return np.array(predictions)


# Example usage and CLI
if __name__ == "__main__":
    import argparse
    import time
    
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description='Edge inference engine')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model (.tflite or .onnx)')
    parser.add_argument('--type', type=str, required=True,
                       choices=['tflite', 'onnx'],
                       help='Model type')
    parser.add_argument('--data', type=str,
                       help='Path to test data (.npy)')
    parser.add_argument('--bench', action='store_true',
                       help='Run performance benchmark')
    
    args = parser.parse_args()
    
    # Initialize engine
    engine = EdgeInferenceEngine(args.model, args.type)
    
    # Print model info
    info = engine.get_model_info()
    print("\nModel Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Run inference if data provided
    if args.data:
        test_data = np.load(args.data)
        print(f"\nRunning inference on {len(test_data)} samples...")
        
        if args.bench:
            # Benchmark
            n_runs = 100
            start = time.time()
            for _ in range(n_runs):
                _ = engine.predict(test_data[:10])  # First 10 samples
            elapsed = time.time() - start
            
            avg_time = (elapsed / n_runs / 10) * 1000  # ms per sample
            print(f"\nBenchmark Results ({n_runs} runs):")
            print(f"  Average inference time: {avg_time:.2f} ms/sample")
            print(f"  Throughput: {1000/avg_time:.1f} samples/second")
        else:
            # Single prediction
            predictions = engine.predict(test_data)
            print(f"\nPredictions shape: {predictions.shape}")
            print(f"First 5 predictions: {predictions[:5]}")
            
            # Statistics
            print(f"\nPrediction Statistics:")
            print(f"  Mean: {np.mean(predictions):.2f}")
            print(f"  Std: {np.std(predictions):.2f}")
            print(f"  Min: {np.min(predictions):.2f}")
            print(f"  Max: {np.max(predictions):.2f}")

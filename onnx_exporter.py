"""
ONNX Model Export for Cross-platform Deployment
Convert TensorFlow and sklearn models to ONNX format
"""

import tensorflow as tf
import numpy as np
from pathlib import Path
import json
import os
from typing import Optional, Dict, Any
import logging
import pickle


class ONNXExporter:
    """Export models to ONNX format"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def export_tensorflow_model(self,
                               model_path: str,
                               output_path: str,
                               input_shape: tuple,
                               opset_version: int = 13) -> Dict[str, Any]:
        """
        Export TensorFlow/Keras model to ONNX
        
        Args:
            model_path: Path to saved Keras model
            output_path: Output path for ONNX model
            input_shape: Input shape (without batch dimension)
            opset_version: ONNX opset version
            
        Returns:
            Export metrics
        """
        try:
            import tf2onnx
            import onnx
        except ImportError:
            raise ImportError("tf2onnx and onnx required. Install with: pip install tf2onnx onnx")
        
        self.logger.info(f"Loading TensorFlow model from {model_path}")
        model = tf.keras.models.load_model(model_path)
        
        # Get input signature
        spec = (tf.TensorSpec(input_shape, tf.float32, name="input"),)
        
        # Convert to ONNX
        self.logger.info("Converting to ONNX...")
        onnx_model, _ = tf2onnx.convert.from_keras(
            model,
            input_signature=spec,
            opset=opset_version
        )
        
        # Save
        onnx.save(onnx_model, output_path)
        
        # Get file sizes
        original_size = os.path.getsize(model_path)
        onnx_size = os.path.getsize(output_path)
        
        metrics = {
            'original_size_mb': original_size / (1024 * 1024),
            'onnx_size_mb': onnx_size / (1024 * 1024),
            'opset_version': opset_version,
            'model_type': 'tensorflow'
        }
        
        self.logger.info(f"TensorFlow model exported to ONNX successfully:")
        self.logger.info(f"  Original size: {metrics['original_size_mb']:.2f} MB")
        self.logger.info(f"  ONNX size: {metrics['onnx_size_mb']:.2f} MB")
        
        return metrics
    
    def export_sklearn_model(self,
                            model_path: str,
                            output_path: str,
                            initial_types: list,
                            target_opset: Optional[int] = None) -> Dict[str, Any]:
        """
        Export sklearn model to ONNX
        
        Args:
            model_path: Path to pickled sklearn model
            output_path: Output path for ONNX model
            initial_types: List of (name, type) for inputs
            target_opset: Target ONNX opset version
            
        Returns:
            Export metrics
        """
        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
            import onnx
        except ImportError:
            raise ImportError("skl2onnx required. Install with: pip install skl2onnx")
        
        self.logger.info(f"Loading sklearn model from {model_path}")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # If model is wrapped, try to extract the underlying model
        if hasattr(model, 'model'):
            model = model.model
        
        # Convert to ONNX
        self.logger.info("Converting to ONNX...")
        onnx_model = convert_sklearn(
            model,
            initial_types=initial_types,
            target_opset=target_opset
        )
        
        # Save
        with open(output_path, 'wb') as f:
            f.write(onnx_model.SerializeToString())
        
        # Get file sizes
        original_size = os.path.getsize(model_path)
        onnx_size = os.path.getsize(output_path)
        
        metrics = {
            'original_size_mb': original_size / (1024 * 1024),
            'onnx_size_mb': onnx_size / (1024 * 1024),
            'model_type': 'sklearn'
        }
        
        self.logger.info(f"Sklearn model exported to ONNX successfully:")
        self.logger.info(f"  Original size: {metrics['original_size_mb']:.2f} MB")
        self.logger.info(f"  ONNX size: {metrics['onnx_size_mb']:.2f} MB")
        
        return metrics
    
    def validate_onnx_model(self,
                           onnx_path: str,
                           test_input: np.ndarray) -> Dict[str, Any]:
        """
        Validate ONNX model can run inference
        
        Args:
            onnx_path: Path to ONNX model
            test_input: Test input data
            
        Returns:
            Validation results
        """
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime required. Install with: pip install onnxruntime")
        
        self.logger.info(f"Validating ONNX model: {onnx_path}")
        
        # Create inference session
        sess = ort.InferenceSession(onnx_path)
        
        # Get input/output names
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name
        
        # Run inference
        result = sess.run([output_name], {input_name: test_input.astype(np.float32)})
        
        validation = {
            'status': 'success',
            'input_name': input_name,
            'input_shape': sess.get_inputs()[0].shape,
            'output_name': output_name,
            'output_shape': sess.get_outputs()[0].shape,
            'test_output_shape': result[0].shape
        }
        
        self.logger.info("ONNX model validation successful")
        self.logger.info(f"  Input: {input_name} {validation['input_shape']}")
        self.logger.info(f"  Output: {output_name} {validation['output_shape']}")
        
        return validation
    
    def compare_predictions(self,
                           original_model_path: str,
                           onnx_model_path: str,
                           test_data: np.ndarray,
                           model_type: str = 'tensorflow') -> Dict[str, Any]:
        """
        Compare predictions between original and ONNX models
        
        Args:
            original_model_path: Path to original model
            onnx_model_path: Path to ONNX model
            test_data: Test input data
            model_type: 'tensorflow' or 'sklearn'
            
        Returns:
            Comparison results
        """
        import onnxruntime as ort
        
        # Get original predictions
        if model_type == 'tensorflow':
            model = tf.keras.models.load_model(original_model_path)
            original_preds = model.predict(test_data, verbose=0)
        else:  # sklearn
            with open(original_model_path, 'rb') as f:
                model = pickle.load(f)
            if hasattr(model, 'model'):
                model = model.model
            original_preds = model.predict(test_data)
        
        # Get ONNX predictions
        sess = ort.InferenceSession(onnx_model_path)
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name
        onnx_preds = sess.run([output_name], {input_name: test_data.astype(np.float32)})[0]
        
        # Compare
        max_diff = np.max(np.abs(original_preds - onnx_preds))
        mean_diff = np.mean(np.abs(original_preds - onnx_preds))
        correlation = np.corrcoef(original_preds.flatten(), onnx_preds.flatten())[0, 1]
        
        comparison = {
            'max_absolute_difference': float(max_diff),
            'mean_absolute_difference': float(mean_diff),
            'correlation': float(correlation),
            'samples_tested': len(test_data)
        }
        
        self.logger.info("Prediction Comparison:")
        self.logger.info(f"  Max difference: {max_diff:.6f}")
        self.logger.info(f"  Mean difference: {mean_diff:.6f}")
        self.logger.info(f"  Correlation: {correlation:.6f}")
        
        return comparison


# CLI interface
if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description='Export models to ONNX')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model (.h5 or .pkl)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for ONNX model')
    parser.add_argument('--type', type=str, required=True,
                       choices=['tensorflow', 'sklearn'],
                       help='Model type')
    parser.add_argument('--input-shape', type=str,
                       help='Input shape for TensorFlow models (e.g., "None,30,50")')
    parser.add_argument('--num-features', type=int,
                       help='Number of features for sklearn models')
    parser.add_argument('--test-data', type=str,
                       help='Path to test data (.npy) for validation')
    
    args = parser.parse_args()
    
    exporter = ONNXExporter()
    
    if args.type == 'tensorflow':
        if not args.input_shape:
            raise ValueError("--input-shape required for TensorFlow models")
        
        # Parse input shape
        shape = tuple(int(x) if x != 'None' else None for x in args.input_shape.split(','))
        
        metrics = exporter.export_tensorflow_model(
            model_path=args.model,
            output_path=args.output,
            input_shape=shape
        )
    else:  # sklearn
        if not args.num_features:
            raise ValueError("--num-features required for sklearn models")
        
        from skl2onnx.common.data_types import FloatTensorType
        initial_types = [('float_input', FloatTensorType([None, args.num_features]))]
        
        metrics = exporter.export_sklearn_model(
            model_path=args.model,
            output_path=args.output,
            initial_types=initial_types
        )
    
    # Validate if test data provided
    if args.test_data:
        test_data = np.load(args.test_data)
        validation = exporter.validate_onnx_model(args.output, test_data)
        comparison = exporter.compare_predictions(args.model, args.output, test_data, args.type)
        
        print("\nValidation Results:")
        print(json.dumps(validation, indent=2))
        print("\nComparison Results:")
        print(json.dumps(comparison, indent=2))

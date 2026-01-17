"""
Model Quantization for Edge Deployment
Convert TensorFlow models to TensorFlow Lite with quantization
"""

import tensorflow as tf
import numpy as np
from pathlib import Path
import json
import os
from typing import Optional, Tuple, Dict, Any
import logging


class ModelQuantizer:
    """Quantize TensorFlow/Keras models for edge deployment"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def quantize_model(self,
                       model_path: str,
                       output_path: str,
                       quantization_type: str = 'dynamic',
                       representative_dataset: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Quantize a TensorFlow model to TensorFlow Lite
        
        Args:
            model_path: Path to saved Keras model (.h5 or SavedModel)
            output_path: Path to save quantized .tflite model
            quantization_type: Type of quantization ('dynamic', 'int8', 'float16')
            representative_dataset: Representative data for full integer quantization
            
        Returns:
            Dictionary with quantization metrics
        """
        self.logger.info(f"Loading model from {model_path}")
        
        # Load model
        if model_path.endswith('.h5'):
            model = tf.keras.models.load_model(model_path)
        else:
            model = tf.saved_model.load(model_path)
        
        # Create converter
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Configure quantization
        if quantization_type == 'dynamic':
            self.logger.info("Applying dynamic range quantization")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
        elif quantization_type == 'float16':
            self.logger.info("Applying float16 quantization")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            
        elif quantization_type == 'int8':
            self.logger.info("Applying full integer quantization (INT8)")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            if representative_dataset is not None:
                def representative_dataset_gen():
                    for sample in representative_dataset:
                        yield [sample.astype(np.float32)]
                
                converter.representative_dataset = representative_dataset_gen
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8
        else:
            raise ValueError(f"Unknown quantization type: {quantization_type}")
        
        # Convert
        self.logger.info("Converting model...")
        tflite_model = converter.convert()
        
        # Save
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        # Get file sizes
        original_size = os.path.getsize(model_path)
        quantized_size = os.path.getsize(output_path)
        size_reduction = (1 - quantized_size / original_size) * 100
        
        metrics = {
            'original_size_mb': original_size / (1024 * 1024),
            'quantized_size_mb': quantized_size / (1024 * 1024),
            'size_reduction_percent': size_reduction,
            'quantization_type': quantization_type
        }
        
        self.logger.info(f"Model quantized successfully:")
        self.logger.info(f"  Original size: {metrics['original_size_mb']:.2f} MB")
        self.logger.info(f"  Quantized size: {metrics['quantized_size_mb']:.2f} MB")
        self.logger.info(f"  Size reduction: {metrics['size_reduction_percent']:.1f}%")
        
        return metrics
    
    def benchmark_model(self,
                       original_model_path: str,
                       quantized_model_path: str,
                       test_data: np.ndarray,
                       test_labels: np.ndarray) -> Dict[str, Any]:
        """
        Benchmark original vs quantized model
        
        Args:
            original_model_path: Path to original model
            quantized_model_path: Path to quantized TFLite model
            test_data: Test input data
            test_labels: Test labels
            
        Returns:
            Benchmark results
        """
        import time
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        # Load original model
        self.logger.info("Benchmarking original model...")
        original_model = tf.keras.models.load_model(original_model_path)
        
        # Original model inference
        start = time.time()
        original_preds = original_model.predict(test_data, verbose=0)
        original_time = time.time() - start
        
        original_rmse = np.sqrt(mean_squared_error(test_labels, original_preds))
        original_mae = mean_absolute_error(test_labels, original_preds)
        original_r2 = r2_score(test_labels, original_preds)
        
        # Load quantized model
        self.logger.info("Benchmarking quantized model...")
        interpreter = tf.lite.Interpreter(model_path=quantized_model_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Quantized model inference
        quantized_preds = []
        start = time.time()
        
        for i in range(len(test_data)):
            sample = test_data[i:i+1].astype(np.float32)
            interpreter.set_tensor(input_details[0]['index'], sample)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            quantized_preds.append(output[0])
        
        quantized_time = time.time() - start
        quantized_preds = np.array(quantized_preds)
        
        quantized_rmse = np.sqrt(mean_squared_error(test_labels, quantized_preds))
        quantized_mae = mean_absolute_error(test_labels, quantized_preds)
        quantized_r2 = r2_score(test_labels, quantized_preds)
        
        # Calculate accuracy degradation
        rmse_diff = ((quantized_rmse - original_rmse) / original_rmse) * 100
        mae_diff = ((quantized_mae - original_mae) / original_mae) * 100
        r2_diff = ((original_r2 - quantized_r2) / abs(original_r2)) * 100
        
        results = {
            'original': {
                'rmse': float(original_rmse),
                'mae': float(original_mae),
                'r2': float(original_r2),
                'inference_time_ms': float(original_time * 1000 / len(test_data))
            },
            'quantized': {
                'rmse': float(quantized_rmse),
                'mae': float(quantized_mae),
                'r2': float(quantized_r2),
                'inference_time_ms': float(quantized_time * 1000 / len(test_data))
            },
            'degradation': {
                'rmse_diff_percent': float(rmse_diff),
                'mae_diff_percent': float(mae_diff),
                'r2_diff_percent': float(r2_diff)
            },
            'speedup': float(original_time / quantized_time)
        }
        
        self.logger.info("Benchmark Results:")
        self.logger.info(f"  Original RMSE: {original_rmse:.2f}, Quantized RMSE: {quantized_rmse:.2f} ({rmse_diff:+.1f}%)")
        self.logger.info(f"  Original MAE: {original_mae:.2f}, Quantized MAE: {quantized_mae:.2f} ({mae_diff:+.1f}%)")
        self.logger.info(f"  Original R²: {original_r2:.4f}, Quantized R²: {quantized_r2:.4f} ({r2_diff:+.1f}%)")
        self.logger.info(f"  Speedup: {results['speedup']:.2f}x")
        
        return results


def quantize_all_models(models_dir: str = 'models/saved',
                       output_dir: str = 'models/tflite',
                       test_data: Optional[np.ndarray] = None,
                       test_labels: Optional[np.ndarray] = None):
    """
    Quantize all TensorFlow models in a directory
    
    Args:
        models_dir: Directory containing models
        output_dir: Output directory for quantized models
        test_data: Optional test data for benchmarking
        test_labels: Optional test labels for benchmarking
    """
    os.makedirs(output_dir, exist_ok=True)
    quantizer = ModelQuantizer()
    
    # Find all .h5 files
    models_path = Path(models_dir)
    model_files = list(models_path.glob('*.h5'))
    
    results = {}
    
    for model_file in model_files:
        model_name = model_file.stem
        print(f"\n{'='*60}")
        print(f"Quantizing {model_name}")
        print('='*60)
        
        # Try different quantization types
        for quant_type in ['dynamic', 'float16']:
            output_path = os.path.join(output_dir, f"{model_name}_{quant_type}.tflite")
            
            try:
                metrics = quantizer.quantize_model(
                    model_path=str(model_file),
                    output_path=output_path,
                    quantization_type=quant_type
                )
                
                # Benchmark if test data provided
                if test_data is not None and test_labels is not None:
                    benchmark = quantizer.benchmark_model(
                        original_model_path=str(model_file),
                        quantized_model_path=output_path,
                        test_data=test_data,
                        test_labels=test_labels
                    )
                    metrics['benchmark'] = benchmark
                
                results[f"{model_name}_{quant_type}"] = metrics
                
            except Exception as e:
                print(f"Error quantizing {model_name} with {quant_type}: {e}")
    
    # Save results
    results_path = os.path.join(output_dir, 'quantization_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nQuantization complete. Results saved to {results_path}")
    return results


# CLI interface
if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description='Quantize TensorFlow models')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model (.h5)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for quantized model (.tflite)')
    parser.add_argument('--type', type=str, default='dynamic',
                       choices=['dynamic', 'float16', 'int8'],
                       help='Quantization type')
    parser.add_argument('--test-data', type=str,
                       help='Path to test data (.npy) for benchmarking')
    parser.add_argument('--test-labels', type=str,
                       help='Path to test labels (.npy) for benchmarking')
    
    args = parser.parse_args()
    
    quantizer = ModelQuantizer()
    
    # Quantize
    metrics = quantizer.quantize_model(
        model_path=args.model,
        output_path=args.output,
        quantization_type=args.type
    )
    
    # Benchmark if test data provided
    if args.test_data and args.test_labels:
        test_data = np.load(args.test_data)
        test_labels = np.load(args.test_labels)
        
        benchmark = quantizer.benchmark_model(
            original_model_path=args.model,
            quantized_model_path=args.output,
            test_data=test_data,
            test_labels=test_labels
        )
        
        print("\nBenchmark Results:")
        print(json.dumps(benchmark, indent=2))

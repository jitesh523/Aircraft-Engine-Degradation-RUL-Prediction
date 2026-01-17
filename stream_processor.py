"""
Stream Processor for Real-time RUL Predictions
Consumes sensor data streams and generates predictions
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
import logging
import asyncio
import pickle
import json
from pathlib import Path

from streaming_ingestion import StreamingIngestion, async_consume_stream
from preprocessor import Preprocessor
from feature_engineer import FeatureEngineer
import config


class StreamProcessor:
    """Process streaming sensor data and generate RUL predictions"""
    
    def __init__(self,
                 model_path: str,
                 scaler_path: str,
                 feature_info_path: str,
                 redis_host: str = 'localhost',
                 redis_port: int = 6379,
                 input_stream: str = 'engine:sensors',
                 output_stream: str = 'predictions:rul'):
        """
        Initialize stream processor
        
        Args:
            model_path: Path to trained model
            scaler_path: Path to fitted scaler
            feature_info_path: Path to feature info JSON
            redis_host: Redis server host
            redis_port: Redis server port
            input_stream: Name of input stream
            output_stream: Name of output stream
        """
        self.logger = logging.getLogger(__name__)
        
        # Load model
        self.logger.info(f"Loading model from {model_path}")
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        # Load scaler
        self.logger.info(f"Loading scaler from {scaler_path}")
        self.preprocessor = Preprocessor()
        self.preprocessor.load_scaler(scaler_path)
        
        # Load feature info
        with open(feature_info_path, 'r') as f:
            feature_info = json.load(f)
        self.feature_columns = feature_info['feature_columns']
        
        # Initialize Redis
        self.ingestion = StreamingIngestion(
            redis_host=redis_host,
            redis_port=redis_port,
            stream_name=input_stream
        )
        self.output_stream = output_stream
        
        # Feature engineer
        self.feature_engineer = FeatureEngineer()
        
        # Windowed data storage (for rolling features)
        self.engine_windows = {}  # unit_id -> list of recent readings
        self.window_size = max(config.ROLLING_WINDOW_SIZES) if config.ROLLING_WINDOW_SIZES else 15
        
        self.logger.info("Stream processor initialized")
    
    def _update_engine_window(self, unit_id: int, reading: Dict[str, Any]):
        """Update windowed data for an engine"""
        if unit_id not in self.engine_windows:
            self.engine_windows[unit_id] = []
        
        self.engine_windows[unit_id].append(reading)
        
        # Keep only recent window
        if len(self.engine_windows[unit_id]) > self.window_size:
            self.engine_windows[unit_id] = self.engine_windows[unit_id][-self.window_size:]
    
    def _create_dataframe_from_window(self, unit_id: int) -> pd.DataFrame:
        """Create DataFrame from engine window"""
        if unit_id not in self.engine_windows or not self.engine_windows[unit_id]:
            return None
        
        # Convert window to DataFrame
        records = []
        for reading in self.engine_windows[unit_id]:
            record = {
                'unit_id': reading['unit_id'],
                'time_cycles': reading['time_cycle'],
                'setting_1': reading['setting_1'],
                'setting_2': reading['setting_2'],
                'setting_3': reading['setting_3']
            }
            record.update(reading['sensors'])
            records.append(record)
        
        df = pd.DataFrame(records)
        return df
    
    async def process_message(self, message: Dict[str, Any]):
        """
        Process a single sensor reading and generate prediction
        
        Args:
            message: Sensor reading message
        """
        try:
            unit_id = message['unit_id']
            
            # Update window
            self._update_engine_window(unit_id, message)
            
            # Need enough data for rolling features
            if len(self.engine_windows[unit_id]) < self.window_size:
                self.logger.debug(f"Unit {unit_id}: Insufficient data for prediction ({len(self.engine_windows[unit_id])}/{self.window_size})")
                return
            
            # Create DataFrame from window
            df = self._create_dataframe_from_window(unit_id)
            
            # Engineer features
            df_engineered = self.feature_engineer.add_features(df)
            
            # Get latest reading
            latest_row = df_engineered.iloc[-1]
            
            # Extract features
            features = []
            for col in self.feature_columns:
                if col in latest_row:
                    features.append(latest_row[col])
                else:
                    features.append(0.0)
            
            features_array = np.array([features])
            
            # Make prediction
            if hasattr(self.model, 'predict'):
                rul_prediction = self.model.predict(features_array)[0]
            else:
                rul_prediction = self.model(features_array).numpy()[0][0]
            
            # Ensure non-negative
            rul_prediction = max(0, float(rul_prediction))
            
            # Determine health status
            if rul_prediction < config.MAINTENANCE_THRESHOLDS['critical']:
                health_status = 'CRITICAL'
            elif rul_prediction < config.MAINTENANCE_THRESHOLDS['warning']:
                health_status = 'WARNING'
            else:
                health_status = 'HEALTHY'
            
            # Publish prediction
            prediction_data = {
                'unit_id': str(unit_id),
                'time_cycle': str(message['time_cycle']),
                'rul_prediction': str(rul_prediction),
                'health_status': health_status,
                'timestamp': str(message['timestamp'])
            }
            
            self.ingestion.redis_client.xadd(
                self.output_stream,
                prediction_data,
                maxlen=50000
            )
            
            self.logger.info(f"Unit {unit_id}, Cycle {message['time_cycle']}: RUL={rul_prediction:.1f}, Status={health_status}")
            
        except Exception as e:
            self.logger.error(f"Error processing message: {e}", exc_info=True)
    
    async def start(self,
                   consumer_group: str = 'rul_processors',
                   consumer_name: str = 'processor-1'):
        """
        Start processing stream
        
        Args:
            consumer_group: Consumer group name
            consumer_name: Consumer name
        """
        self.logger.info(f"Starting stream processor: {consumer_name}")
        self.logger.info(f"Input stream: {self.ingestion.stream_name}")
        self.logger.info(f"Output stream: {self.output_stream}")
        
        await async_consume_stream(
            ingestion=self.ingestion,
            consumer_group=consumer_group,
            consumer_name=consumer_name,
            callback=self.process_message
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            'tracked_engines': len(self.engine_windows),
            'total_readings': sum(len(window) for window in self.engine_windows.values()),
            'input_stream': self.ingestion.stream_name,
            'output_stream': self.output_stream,
            'stream_info': self.ingestion.get_stream_info()
        }


# CLI interface
if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description='Stream processor for RUL predictions')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--scaler', type=str,
                       default='models/saved/scaler.pkl',
                       help='Path to scaler')
    parser.add_argument('--feature-info', type=str,
                       default='models/saved/feature_info.json',
                       help='Path to feature info JSON')
    parser.add_argument('--redis-host', type=str, default='localhost',
                       help='Redis host')
    parser.add_argument('--redis-port', type=int, default=6379,
                       help='Redis port')
    parser.add_argument('--consumer-group', type=str, default='rul_processors',
                       help='Consumer group name')
    parser.add_argument('--consumer-name', type=str, default='processor-1',
                       help='Consumer name')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = StreamProcessor(
        model_path=args.model,
        scaler_path=args.scaler,
        feature_info_path=args.feature_info,
        redis_host=args.redis_host,
        redis_port=args.redis_port
    )
    
    # Start processing
    try:
        asyncio.run(processor.start(
            consumer_group=args.consumer_group,
            consumer_name=args.consumer_name
        ))
    except KeyboardInterrupt:
        logging.info("Stream processor stopped by user")

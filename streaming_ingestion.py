"""
Real-time Streaming Data Ingestion using Redis Streams
Handles sensor data ingestion, buffering, and preprocessing
"""

import redis
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import time
from datetime import datetime
import logging
import asyncio
from dataclasses import dataclass, asdict


@dataclass
class SensorReading:
    """Single sensor reading from an engine"""
    unit_id: int
    timestamp: float
    time_cycle: int
    setting_1: float
    setting_2: float
    setting_3: float
    sensors: Dict[str, float]  # sensor_1 through sensor_21
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SensorReading':
        """Create from dictionary"""
        return cls(**data)


class StreamingIngestion:
    """Real-time data ingestion using Redis Streams"""
    
    def __init__(self, 
                 redis_host: str = 'localhost',
                 redis_port: int = 6379,
                 redis_db: int = 0,
                 stream_name: str = 'engine:sensors'):
        """
        Initialize streaming ingestion
        
        Args:
            redis_host: Redis server host
            redis_port: Redis server port
            redis_db: Redis database number
            stream_name: Name of the Redis stream
        """
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            decode_responses=True
        )
        self.stream_name = stream_name
        self.logger = logging.getLogger(__name__)
        
        # Test connection
        try:
            self.redis_client.ping()
            self.logger.info(f"Connected to Redis at {redis_host}:{redis_port}")
        except redis.ConnectionError as e:
            self.logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def publish_reading(self, reading: SensorReading) -> str:
        """
        Publish a single sensor reading to the stream
        
        Args:
            reading: SensorReading object
            
        Returns:
            Message ID
        """
        message_data = reading.to_dict()
        # Convert nested dict to JSON string for Redis
        message_data['sensors'] = json.dumps(message_data['sensors'])
        
        message_id = self.redis_client.xadd(
            self.stream_name,
            message_data,
            maxlen=100000  # Keep last 100k messages
        )
        
        return message_id
    
    def publish_batch(self, readings: List[SensorReading]) -> List[str]:
        """
        Publish multiple readings in a batch
        
        Args:
            readings: List of SensorReading objects
            
        Returns:
            List of message IDs
        """
        pipeline = self.redis_client.pipeline()
        
        for reading in readings:
            message_data = reading.to_dict()
            message_data['sensors'] = json.dumps(message_data['sensors'])
            pipeline.xadd(self.stream_name, message_data, maxlen=100000)
        
        message_ids = pipeline.execute()
        self.logger.info(f"Published {len(readings)} readings to stream")
        
        return message_ids
    
    def consume_stream(self, 
                      consumer_group: str = 'processors',
                      consumer_name: str = 'processor-1',
                      block_ms: int = 1000,
                      count: int = 10) -> List[Dict[str, Any]]:
        """
        Consume messages from the stream using consumer groups
        
        Args:
            consumer_group: Name of the consumer group
            consumer_name: Name of this consumer
            block_ms: Milliseconds to block waiting for messages
            count: Maximum number of messages to read
            
        Returns:
            List of messages
        """
        # Create consumer group if it doesn't exist
        try:
            self.redis_client.xgroup_create(
                self.stream_name,
                consumer_group,
                id='0',
                mkstream=True
            )
        except redis.ResponseError as e:
            if 'BUSYGROUP' not in str(e):
                raise
        
        # Read from stream
        messages = self.redis_client.xreadgroup(
            groupname=consumer_group,
            consumername=consumer_name,
            streams={self.stream_name: '>'},
            count=count,
            block=block_ms
        )
        
        if not messages:
            return []
        
        # Parse messages
        parsed_messages = []
        for stream_name, stream_messages in messages:
            for message_id, message_data in stream_messages:
                # Parse sensors JSON
                if 'sensors' in message_data:
                    message_data['sensors'] = json.loads(message_data['sensors'])
                
                # Convert numeric fields
                for key in ['unit_id', 'time_cycle']:
                    if key in message_data:
                        message_data[key] = int(message_data[key])
                
                for key in ['timestamp', 'setting_1', 'setting_2', 'setting_3']:
                    if key in message_data:
                        message_data[key] = float(message_data[key])
                
                parsed_messages.append({
                    'message_id': message_id,
                    'data': message_data
                })
        
        return parsed_messages
    
    def acknowledge_message(self, 
                           consumer_group: str,
                           message_id: str):
        """
        Acknowledge message processing
        
        Args:
            consumer_group: Name of the consumer group
            message_id: Message ID to acknowledge
        """
        self.redis_client.xack(self.stream_name, consumer_group, message_id)
    
    def get_stream_info(self) -> Dict[str, Any]:
        """Get information about the stream"""
        try:
            info = self.redis_client.xinfo_stream(self.stream_name)
            return info
        except redis.ResponseError:
            return {'length': 0, 'groups': 0}
    
    def get_pending_messages(self, consumer_group: str) -> int:
        """Get count of pending messages for a consumer group"""
        try:
            pending = self.redis_client.xpending(self.stream_name, consumer_group)
            return pending['pending']
        except redis.ResponseError:
            return 0


class DataFrameStreamer:
    """Stream data from DataFrame to Redis"""
    
    def __init__(self, ingestion: StreamingIngestion):
        """
        Initialize DataFrame streamer
        
        Args:
            ingestion: StreamingIngestion instance
        """
        self.ingestion = ingestion
        self.logger = logging.getLogger(__name__)
    
    def stream_from_dataframe(self,
                             df: pd.DataFrame,
                             batch_size: int = 100,
                             delay_ms: int = 10,
                             sensor_columns: Optional[List[str]] = None):
        """
        Stream data from a DataFrame simulating real-time ingestion
        
        Args:
            df: DataFrame with engine data
            batch_size: Number of rows to send in each batch
            delay_ms: Delay between batches in milliseconds
            sensor_columns: List of sensor column names
        """
        if sensor_columns is None:
            sensor_columns = [f'sensor_{i}' for i in range(1, 22)]
        
        total_rows = len(df)
        sent_count = 0
        
        self.logger.info(f"Starting to stream {total_rows} rows...")
        
        for i in range(0, total_rows, batch_size):
            batch_df = df.iloc[i:i+batch_size]
            readings = []
            
            for _, row in batch_df.iterrows():
                # Create sensor dict
                sensors = {col: float(row[col]) for col in sensor_columns if col in row}
                
                reading = SensorReading(
                    unit_id=int(row['unit_id']),
                    timestamp=time.time(),
                    time_cycle=int(row['time_cycles']),
                    setting_1=float(row['setting_1']),
                    setting_2=float(row['setting_2']),
                    setting_3=float(row['setting_3']),
                    sensors=sensors
                )
                readings.append(reading)
            
            # Publish batch
            self.ingestion.publish_batch(readings)
            sent_count += len(readings)
            
            if sent_count % 1000 == 0:
                self.logger.info(f"Streamed {sent_count}/{total_rows} rows")
            
            # Simulate real-time delay
            if delay_ms > 0:
                time.sleep(delay_ms / 1000.0)
        
        self.logger.info(f"Streaming complete. Sent {sent_count} readings.")


async def async_consume_stream(ingestion: StreamingIngestion,
                               consumer_group: str = 'processors',
                               consumer_name: str = 'async-processor',
                               callback=None):
    """
    Async consumer for processing stream messages
    
    Args:
        ingestion: StreamingIngestion instance
        consumer_group: Consumer group name
        consumer_name: Consumer name
        callback: Async callback function to process messages
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Starting async consumer: {consumer_name}")
    
    while True:
        try:
            messages = ingestion.consume_stream(
                consumer_group=consumer_group,
                consumer_name=consumer_name,
                block_ms=1000,
                count=10
            )
            
            if messages and callback:
                for msg in messages:
                    try:
                        await callback(msg['data'])
                        ingestion.acknowledge_message(consumer_group, msg['message_id'])
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
            
            # Small delay to prevent tight loop
            await asyncio.sleep(0.01)
            
        except KeyboardInterrupt:
            logger.info("Consumer stopped by user")
            break
        except Exception as e:
            logger.error(f"Error in consumer: {e}")
            await asyncio.sleep(1)


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize ingestion
    ingestion = StreamingIngestion()
    
    # Example: Create and publish a single reading
    example_reading = SensorReading(
        unit_id=1,
        timestamp=time.time(),
        time_cycle=100,
        setting_1=0.5,
        setting_2=0.3,
        setting_3=100.0,
        sensors={f'sensor_{i}': np.random.rand() for i in range(1, 22)}
    )
    
    message_id = ingestion.publish_reading(example_reading)
    print(f"Published message: {message_id}")
    
    # Get stream info
    info = ingestion.get_stream_info()
    print(f"Stream info: {info}")

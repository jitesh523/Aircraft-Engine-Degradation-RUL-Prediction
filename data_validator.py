"""
Data Validation Module
Validates input data quality, schema, and statistical properties
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from utils import setup_logging

logger = setup_logging(__name__)


class DataValidator:
    """
    Validates data quality and schema for aircraft engine sensor data
    """
    
    def __init__(self):
        """Initialize data validator with expected schema and ranges"""
        self.expected_columns = [
            'unit_id', 'time_cycles',
            'operational_setting_1', 'operational_setting_2', 'operational_setting_3'
        ] + [f'sensor_{i}' for i in range(1, 22)]
        
        # Expected sensor ranges (min, max) based on C-MAPSS dataset
        self.sensor_ranges = {
            'sensor_1': (518.0, 642.0),
            'sensor_2': (641.0, 644.0),
            'sensor_3': (1580.0, 1605.0),
            'sensor_4': (1398.0, 1615.0),
            'sensor_5': (14.0, 24.0),
            'sensor_6': (21.0, 24.0),
            'sensor_7': (553.0, 555.0),
            'sensor_8': (2388.0, 2389.0),
            'sensor_9': (9046.0, 9062.0),
            'sensor_10': (1.0, 2.0),
            'sensor_11': (47.0, 48.0),
            'sensor_12': (521.0, 523.0),
            'sensor_13': (2388.0, 2389.0),
            'sensor_14': (8127.0, 8143.0),
            'sensor_15': (8.0, 9.0),
            'sensor_16': (0.02, 0.04),
            'sensor_17': (392.0, 394.0),
            'sensor_18': (2388.0, 2389.0),
            'sensor_19': (100.0, 101.0),
            'sensor_20': (38.0, 40.0),
            'sensor_21': (23.0, 24.0),
        }
    
    def validate_schema(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate that data has expected schema
        
        Args:
            data: DataFrame to validate
            
        Returns:
            (is_valid, error_messages)
        """
        errors = []
        
        # Check for required columns
        missing_cols = set(self.expected_columns) - set(data.columns)
        if missing_cols:
            errors.append(f"Missing columns: {missing_cols}")
        
        # Check data types
        if 'unit_id' in data.columns and not pd.api.types.is_integer_dtype(data['unit_id']):
            errors.append("Column 'unit_id' should be integer type")
        
        if 'time_cycles' in data.columns and not pd.api.types.is_integer_dtype(data['time_cycles']):
            errors.append("Column 'time_cycles' should be integer type")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def validate_data_quality(self, data: pd.DataFrame) -> Tuple[bool, Dict[str, any]]:
        """
        Validate data quality: missing values, duplicates, outliers
        
        Args:
            data: DataFrame to validate
            
        Returns:
            (is_valid, quality_report)
        """
        report = {}
        issues = []
        
        # Check for missing values
        missing_counts = data.isnull().sum()
        if missing_counts.any():
            missing_info = missing_counts[missing_counts > 0].to_dict()
            report['missing_values'] = missing_info
            issues.append(f"Missing values found in {len(missing_info)} columns")
        
        # Check for duplicates
        duplicates = data.duplicated(subset=['unit_id', 'time_cycles'])
        if duplicates.any():
            n_duplicates = duplicates.sum()
            report['duplicates'] = n_duplicates
            issues.append(f"Found {n_duplicates} duplicate rows")
        
        # Check for negative time cycles
        if 'time_cycles' in data.columns:
            negative_cycles = (data['time_cycles'] < 0).sum()
            if negative_cycles > 0:
                report['negative_time_cycles'] = negative_cycles
                issues.append(f"Found {negative_cycles} negative time cycle values")
        
        # Check for non-sequential time cycles per engine
        if 'unit_id' in data.columns and 'time_cycles' in data.columns:
            for unit_id in data['unit_id'].unique():
                unit_data = data[data['unit_id'] == unit_id].sort_values('time_cycles')
                cycles = unit_data['time_cycles'].values
                
                # Check if cycles start at 1
                if cycles[0] != 1:
                    issues.append(f"Unit {unit_id}: time_cycles doesn't start at 1")
                
                # Check if cycles are sequential
                if not np.array_equal(cycles, np.arange(1, len(cycles) + 1)):
                    issues.append(f"Unit {unit_id}: time_cycles are not sequential")
        
        report['issues'] = issues
        is_valid = len(issues) == 0
        
        return is_valid, report
    
    def validate_sensor_ranges(self, data: pd.DataFrame, 
                               tolerance: float = 0.1) -> Tuple[bool, Dict[str, any]]:
        """
        Validate that sensor values are within expected ranges
        
        Args:
            data: DataFrame to validate
            tolerance: Tolerance factor for range (0.1 = 10% beyond range is acceptable)
            
        Returns:
            (is_valid, validation_report)
        """
        report = {}
        outliers = {}
        
        for sensor, (min_val, max_val) in self.sensor_ranges.items():
            if sensor not in data.columns:
                continue
            
            # Calculate acceptable range with tolerance
            range_span = max_val - min_val
            lower_bound = min_val - (range_span * tolerance)
            upper_bound = max_val + (range_span * tolerance)
            
            # Find outliers
            sensor_data = data[sensor]
            below_range = (sensor_data < lower_bound).sum()
            above_range = (sensor_data > upper_bound).sum()
            
            if below_range > 0 or above_range > 0:
                outliers[sensor] = {
                    'below_range': below_range,
                    'above_range': above_range,
                    'expected_range': (min_val, max_val),
                    'actual_range': (sensor_data.min(), sensor_data.max())
                }
        
        report['outliers'] = outliers
        report['n_sensors_with_outliers'] = len(outliers)
        
        # Consider valid if less than 5% of sensors have outliers
        is_valid = len(outliers) < (len(self.sensor_ranges) * 0.05)
        
        return is_valid, report
    
    def validate_statistical_properties(self, data: pd.DataFrame) -> Dict[str, any]:
        """
        Calculate and validate statistical properties of the data
        
        Args:
            data: DataFrame to analyze
            
        Returns:
            Dictionary with statistical properties
        """
        stats = {}
        
        # Number of engines
        if 'unit_id' in data.columns:
            stats['n_engines'] = data['unit_id'].nunique()
            
            # Lifecycle statistics
            lifecycles = data.groupby('unit_id')['time_cycles'].max()
            stats['lifecycle_stats'] = {
                'mean': lifecycles.mean(),
                'std': lifecycles.std(),
                'min': lifecycles.min(),
                'max': lifecycles.max(),
                'median': lifecycles.median()
            }
        
        # Sensor statistics
        sensor_cols = [col for col in data.columns if col.startswith('sensor_')]
        stats['sensor_stats'] = {}
        
        for sensor in sensor_cols:
            stats['sensor_stats'][sensor] = {
                'mean': data[sensor].mean(),
                'std': data[sensor].std(),
                'min': data[sensor].min(),
                'max': data[sensor].max(),
                'null_pct': (data[sensor].isnull().sum() / len(data)) * 100
            }
        
        return stats
    
    def validate_all(self, data: pd.DataFrame, 
                     verbose: bool = True) -> Tuple[bool, Dict[str, any]]:
        """
        Run all validation checks
        
        Args:
            data: DataFrame to validate
            verbose: Print validation results
            
        Returns:
            (is_valid, full_report)
        """
        logger.info("Running comprehensive data validation...")
        
        full_report = {}
        all_valid = True
        
        # Schema validation
        schema_valid, schema_errors = self.validate_schema(data)
        full_report['schema'] = {
            'valid': schema_valid,
            'errors': schema_errors
        }
        all_valid = all_valid and schema_valid
        
        if verbose and not schema_valid:
            logger.warning(f"Schema validation failed: {schema_errors}")
        
        # Data quality validation
        quality_valid, quality_report = self.validate_data_quality(data)
        full_report['data_quality'] = {
            'valid': quality_valid,
            'report': quality_report
        }
        all_valid = all_valid and quality_valid
        
        if verbose and not quality_valid:
            logger.warning(f"Data quality issues: {quality_report.get('issues', [])}")
        
        # Sensor range validation
        range_valid, range_report = self.validate_sensor_ranges(data)
        full_report['sensor_ranges'] = {
            'valid': range_valid,
            'report': range_report
        }
        
        if verbose and not range_valid:
            logger.warning(f"Sensor range validation: {range_report['n_sensors_with_outliers']} sensors with outliers")
        
        # Statistical properties
        stats = self.validate_statistical_properties(data)
        full_report['statistics'] = stats
        
        if verbose:
            logger.info(f"Data validation complete. Overall valid: {all_valid}")
            logger.info(f"Number of engines: {stats.get('n_engines', 'N/A')}")
        
        return all_valid, full_report


if __name__ == "__main__":
    # Test data validator
    print("="*60)
    print("Testing Data Validator")
    print("="*60)
    
    # Create sample data
    sample_data = pd.DataFrame({
        'unit_id': [1, 1, 1, 2, 2],
        'time_cycles': [1, 2, 3, 1, 2],
        **{f'sensor_{i}': np.random.randn(5) for i in range(1, 22)},
        'operational_setting_1': [1] * 5,
        'operational_setting_2': [0] * 5,
        'operational_setting_3': [100] * 5,
    })
    
    validator = DataValidator()
    is_valid, report = validator.validate_all(sample_data, verbose=True)
    
    print(f"\nValidation result: {'PASSED' if is_valid else 'FAILED'}")
    print("="*60)

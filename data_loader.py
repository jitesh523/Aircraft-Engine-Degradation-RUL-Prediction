"""
Data Loader for NASA C-MAPSS Turbofan Engine Degradation Dataset.

Loads space-delimited train, test, and RUL files for sub-datasets
FD001–FD004 and assigns standard column names from ``config.py``.

Classes:
    CMAPSSDataLoader   — Load train/test/RUL data with quality checks.
    DatasetVersioner   — Hash-based dataset versioning with rollback.

Functions:
    load_dataset       — Convenience one-liner to load any sub-dataset.

Usage::

    from data_loader import load_dataset
    train_df, test_df, rul_df = load_dataset('FD001')
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
import config
from utils import setup_logging

logger = setup_logging(__name__)


class CMAPSSDataLoader:
    """
    Data loader for NASA C-MAPSS dataset
    Handles loading and initial parsing of train/test/RUL files
    """
    
    def __init__(self, dataset_name: str = 'FD001'):
        """
        Initialize data loader for specified dataset
        
        Args:
            dataset_name: Name of dataset ('FD001', 'FD002', 'FD003', 'FD004')
        """
        if dataset_name not in config.DATASETS:
            raise ValueError(f"Invalid dataset name. Choose from {list(config.DATASETS.keys())}")
        
        self.dataset_name = dataset_name
        self.dataset_config = config.DATASETS[dataset_name]
        logger.info(f"Initialized data loader for {dataset_name}")
        logger.info(f"Description: {self.dataset_config['description']}")
    
    def load_train_data(self) -> pd.DataFrame:
        """
        Load training data
        
        Returns:
            DataFrame with columns defined in config.COLUMN_NAMES
        """
        filepath = self.dataset_config['train']
        logger.info(f"Loading training data from {filepath}")
        
        # Read space-separated file without header
        df = pd.read_csv(filepath, sep=r'\s+', header=None)
        
        # Assign column names
        df.columns = config.COLUMN_NAMES
        
        logger.info(f"Loaded {len(df)} training samples from {df['unit_id'].nunique()} engines")
        logger.info(f"Training data shape: {df.shape}")
        
        return df
    
    def load_test_data(self) -> pd.DataFrame:
        """
        Load test data
        
        Returns:
            DataFrame with columns defined in config.COLUMN_NAMES
        """
        filepath = self.dataset_config['test']
        logger.info(f"Loading test data from {filepath}")
        
        # Read space-separated file without header
        df = pd.read_csv(filepath, sep=r'\s+', header=None)
        
        # Assign column names
        df.columns = config.COLUMN_NAMES
        
        logger.info(f"Loaded {len(df)} test samples from {df['unit_id'].nunique()} engines")
        logger.info(f"Test data shape: {df.shape}")
        
        return df
    
    def load_rul_labels(self) -> pd.DataFrame:
        """
        Load ground truth RUL values for test data
        
        Returns:
            DataFrame with columns ['unit_id', 'RUL']
        """
        filepath = self.dataset_config['rul']
        logger.info(f"Loading RUL labels from {filepath}")
        
        # Read single column file
        rul_values = pd.read_csv(filepath, sep=r'\s+', header=None)
        rul_values.columns = ['RUL']
        
        # Add unit_id (1-indexed)
        rul_values['unit_id'] = range(1, len(rul_values) + 1)
        rul_values = rul_values[['unit_id', 'RUL']]
        
        logger.info(f"Loaded {len(rul_values)} RUL labels")
        logger.info(f"RUL range: {rul_values['RUL'].min()} to {rul_values['RUL'].max()} cycles")
        
        return rul_values
    
    def load_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load all data (train, test, RUL labels)
        
        Returns:
            Tuple of (train_df, test_df, rul_df)
        """
        train_df = self.load_train_data()
        test_df = self.load_test_data()
        rul_df = self.load_rul_labels()
        
        logger.info("All data loaded successfully")
        return train_df, test_df, rul_df
    
    def get_dataset_info(self) -> Dict:
        """
        Get information about the dataset
        
        Returns:
            Dictionary with dataset statistics
        """
        train_df, test_df, rul_df = self.load_all_data()
        
        info = {
            'dataset_name': self.dataset_name,
            'description': self.dataset_config['description'],
            'train_engines': train_df['unit_id'].nunique(),
            'test_engines': test_df['unit_id'].nunique(),
            'train_samples': len(train_df),
            'test_samples': len(test_df),
            'num_features': len(config.COLUMN_NAMES) - 2,  # Exclude unit_id and time_cycles
            'sensor_columns': [col for col in config.COLUMN_NAMES if col.startswith('sensor_')],
            'setting_columns': [col for col in config.COLUMN_NAMES if col.startswith('setting_')],
            'rul_min': rul_df['RUL'].min(),
            'rul_max': rul_df['RUL'].max(),
            'rul_mean': rul_df['RUL'].mean()
        }
        
        return info
    
    def check_data_quality(self, df: pd.DataFrame) -> Dict:
        """
        Check data quality (missing values, duplicates, etc.)
        
        Args:
            df: DataFrame to check
            
        Returns:
            Dictionary with data quality metrics
        """
        quality_report = {
            'total_samples': len(df),
            'missing_values': df.isnull().sum().sum(),
            'duplicate_rows': df.duplicated().sum(),
            'missing_by_column': df.isnull().sum().to_dict(),
            'zero_variance_columns': []
        }
        
        # Check for zero variance columns (sensors that don't change)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].std() == 0:
                quality_report['zero_variance_columns'].append(col)
        
        logger.info(f"Data quality check complete: {quality_report['missing_values']} missing values, "
                   f"{quality_report['duplicate_rows']} duplicates, "
                   f"{len(quality_report['zero_variance_columns'])} zero-variance columns")
        
        return quality_report


def load_dataset(dataset_name: str = 'FD001') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to load dataset
    
    Args:
        dataset_name: Name of dataset to load
        
    Returns:
        Tuple of (train_df, test_df, rul_df)
    """
    loader = CMAPSSDataLoader(dataset_name)
    return loader.load_all_data()


# Backward compatible alias
DataLoader = CMAPSSDataLoader


class DatasetVersioner:
    """
    Dataset versioning with hash-based tracking
    Supports version history and rollback
    """
    
    def __init__(self, versions_dir: str = None):
        """
        Initialize dataset versioner
        
        Args:
            versions_dir: Directory to store version metadata
        """
        import os
        import json
        
        self.versions_dir = versions_dir or os.path.join(config.DATA_DIR, 'versions')
        os.makedirs(self.versions_dir, exist_ok=True)
        
        self.versions = {}
        self.current_version = None
        self._load_versions()
        
        logger.info(f"Initialized DatasetVersioner (dir: {self.versions_dir})")
    
    def _load_versions(self):
        """Load existing version metadata"""
        import os
        import json
        
        registry_path = os.path.join(self.versions_dir, 'registry.json')
        if os.path.exists(registry_path):
            with open(registry_path, 'r') as f:
                data = json.load(f)
                self.versions = data.get('versions', {})
                self.current_version = data.get('current_version')
    
    def _save_versions(self):
        """Save version metadata"""
        import os
        import json
        
        registry_path = os.path.join(self.versions_dir, 'registry.json')
        with open(registry_path, 'w') as f:
            json.dump({
                'versions': self.versions,
                'current_version': self.current_version
            }, f, indent=2)
    
    def _compute_hash(self, df: pd.DataFrame) -> str:
        """Compute hash of DataFrame"""
        import hashlib
        
        data_bytes = pd.util.hash_pandas_object(df).values.tobytes()
        return hashlib.sha256(data_bytes).hexdigest()[:16]
    
    def create_version(self,
                       df: pd.DataFrame,
                       name: str,
                       description: str = '') -> str:
        """
        Create a new dataset version
        
        Args:
            df: DataFrame to version
            name: Version name
            description: Version description
            
        Returns:
            Version ID
        """
        import os
        from datetime import datetime
        
        data_hash = self._compute_hash(df)
        version_id = f"v_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        version_info = {
            'version_id': version_id,
            'name': name,
            'description': description,
            'hash': data_hash,
            'created_at': datetime.now().isoformat(),
            'rows': len(df),
            'columns': len(df.columns),
            'column_names': list(df.columns)
        }
        
        # Save version data
        version_path = os.path.join(self.versions_dir, f"{version_id}.parquet")
        df.to_parquet(version_path, index=False)
        
        version_info['path'] = version_path
        
        self.versions[version_id] = version_info
        self.current_version = version_id
        self._save_versions()
        
        logger.info(f"Created version {version_id}: {name} (hash: {data_hash})")
        
        return version_id
    
    def get_version(self, version_id: str) -> pd.DataFrame:
        """
        Load a specific version
        
        Args:
            version_id: Version ID to load
            
        Returns:
            DataFrame of that version
        """
        if version_id not in self.versions:
            raise ValueError(f"Version not found: {version_id}")
        
        version_info = self.versions[version_id]
        df = pd.read_parquet(version_info['path'])
        
        logger.info(f"Loaded version {version_id}")
        
        return df
    
    def rollback(self, version_id: str) -> pd.DataFrame:
        """
        Rollback to a specific version
        
        Args:
            version_id: Version to rollback to
            
        Returns:
            DataFrame of that version
        """
        df = self.get_version(version_id)
        self.current_version = version_id
        self._save_versions()
        
        logger.info(f"Rolled back to version {version_id}")
        
        return df
    
    def compare_versions(self,
                        version_a: str,
                        version_b: str) -> Dict:
        """
        Compare two versions
        
        Args:
            version_a: First version ID
            version_b: Second version ID
            
        Returns:
            Comparison results
        """
        a_info = self.versions.get(version_a, {})
        b_info = self.versions.get(version_b, {})
        
        return {
            'version_a': version_a,
            'version_b': version_b,
            'same_hash': a_info.get('hash') == b_info.get('hash'),
            'row_diff': b_info.get('rows', 0) - a_info.get('rows', 0),
            'column_diff': b_info.get('columns', 0) - a_info.get('columns', 0),
            'a_columns': set(a_info.get('column_names', [])),
            'b_columns': set(b_info.get('column_names', []))
        }
    
    def list_versions(self) -> pd.DataFrame:
        """List all versions"""
        if not self.versions:
            return pd.DataFrame()
        
        records = []
        for vid, info in self.versions.items():
            records.append({
                'version_id': vid,
                'name': info.get('name'),
                'rows': info.get('rows'),
                'columns': info.get('columns'),
                'created_at': info.get('created_at'),
                'current': vid == self.current_version
            })
        
        return pd.DataFrame(records)
    
    def get_version_summary(self) -> str:
        """Generate version summary"""
        lines = [
            "=" * 60,
            "DATASET VERSION SUMMARY",
            "=" * 60,
            f"Total Versions: {len(self.versions)}",
            f"Current Version: {self.current_version}",
            ""
        ]
        
        for vid, info in list(self.versions.items())[-5:]:
            current = " [CURRENT]" if vid == self.current_version else ""
            lines.append(f"  {vid}: {info.get('name')}{current}")
            lines.append(f"    Rows: {info.get('rows')}, Hash: {info.get('hash')}")
        
        lines.append("=" * 60)
        
        return '\n'.join(lines)


if __name__ == "__main__":
    # Test the data loader
    print("="*60)
    print("Testing NASA C-MAPSS Data Loader")
    print("="*60)
    
    # Load FD001 dataset
    loader = CMAPSSDataLoader('FD001')
    
    # Get dataset info
    info = loader.get_dataset_info()
    print("\nDataset Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Load all data
    train_df, test_df, rul_df = loader.load_all_data()
    
    print("\nTrain Data Sample:")
    print(train_df.head())
    
    print("\nTest Data Sample:")
    print(test_df.head())
    
    print("\nRUL Labels Sample:")
    print(rul_df.head())
    
    # Check data quality
    print("\nData Quality Report (Train):")
    quality = loader.check_data_quality(train_df)
    for key, value in quality.items():
        if key != 'missing_by_column':
            print(f"  {key}: {value}")
    
    print("\n" + "="*60)
    print("Data loader test complete!")
    print("="*60)

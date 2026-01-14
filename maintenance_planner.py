"""
Maintenance Planner for Predictive Maintenance
Uses RUL predictions to optimize maintenance scheduling
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import config
from utils import setup_logging

logger = setup_logging(__name__)


class MaintenancePlanner:
    """
    Maintenance planner using RUL predictions
    """
    
    def __init__(self):
        """Initialize maintenance planner"""
        self.thresholds = config.MAINTENANCE_THRESHOLDS
        self.costs = config.COST_PARAMETERS
        logger.info("Initialized Maintenance Planner")
        logger.info(f"Thresholds: Critical < {self.thresholds['critical']}, "
                   f"Warning {self.thresholds['critical']}-{self.thresholds['warning']}, "
                   f"Healthy ≥ {self.thresholds['healthy']}")
    
    def classify_health_status(self, rul: float) -> str:
        """
        Classify engine health status based on RUL
        
        Args:
            rul: Remaining Useful Life in cycles
            
        Returns:
            Health status: 'Critical', 'Warning', or 'Healthy'
        """
        if rul < self.thresholds['critical']:
            return 'Critical'
        elif rul < self.thresholds['warning']:
            return 'Warning'
        else:
            return 'Healthy'
    
    def create_maintenance_schedule(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create maintenance schedule based on RUL predictions
        
        Args:
            predictions_df: DataFrame with columns [unit_id, RUL_pred]
            
        Returns:
            DataFrame with maintenance recommendations
        """
        logger.info("Creating maintenance schedule...")
        
        df = predictions_df.copy()
        
        # Classify health status
        df['health_status'] = df['RUL_pred'].apply(self.classify_health_status)
        
        # Determine action
        def get_action(status):
            if status == 'Critical':
                return 'Immediate maintenance required - Ground aircraft'
            elif status == 'Warning':
                return 'Schedule maintenance at next opportunity'
            else:
                return 'Continue routine monitoring'
        
        df['recommended_action'] = df['health_status'].apply(get_action)
        
        # Priority (1 = highest)
        priority_map = {'Critical': 1, 'Warning': 2, 'Healthy': 3}
        df['priority'] = df['health_status'].map(priority_map)
        
        # Sort by priority
        df = df.sort_values('priority')
        
        # Summary statistics
        status_counts = df['health_status'].value_counts()
        logger.info(f"Maintenance Schedule Summary:")
        logger.info(f"  Critical: {status_counts.get('Critical', 0)} engines")
        logger.info(f"  Warning: {status_counts.get('Warning', 0)} engines")
        logger.info(f"  Healthy: {status_counts.get('Healthy', 0)} engines")
        
        return df
    
    def simulate_traditional_maintenance(self, 
                                        engine_lifetimes: np.ndarray,
                                        interval: int = None) -> Dict:
        """
        Simulate traditional fixed-interval maintenance
        
        Args:
            engine_lifetimes: Actual lifetimes of engines (in cycles)
            interval: Fixed maintenance interval
            
        Returns:
            Dictionary with simulation results
        """
        if interval is None:
            interval = config.TRADITIONAL_MAINTENANCE_INTERVAL
        
        logger.info(f"Simulating traditional maintenance (fixed {interval}-cycle interval)...")
        
        n_engines = len(engine_lifetimes)
        unexpected_failures = 0
        scheduled_maintenances = 0
        
        for lifetime in engine_lifetimes:
            # Number of scheduled maintenances before failure
            n_scheduled = int(lifetime / interval)
            scheduled_maintenances += n_scheduled
            
            # If engine fails before next scheduled maintenance
            if lifetime % interval > 0:
                unexpected_failures += 1
        
        # Calculate costs
        scheduled_cost = scheduled_maintenances * self.costs['scheduled_maintenance']
        unscheduled_cost = unexpected_failures * self.costs['unscheduled_maintenance']
        total_cost = scheduled_cost + unscheduled_cost
        
        # Fleet availability (engines not failed unexpectedly)
        availability = (n_engines - unexpected_failures) / n_engines * 100
        
        results = {
            'method': 'Traditional (Fixed Interval)',
            'interval': interval,
            'total_engines': n_engines,
            'scheduled_maintenances': scheduled_maintenances,
            'unexpected_failures': unexpected_failures,
            'scheduled_cost': scheduled_cost,
            'unscheduled_cost': unscheduled_cost,
            'total_cost': total_cost,
            'fleet_availability': availability
        }
        
        logger.info(f"  Total maintenances: {scheduled_maintenances}")
        logger.info(f"  Unexpected failures: {unexpected_failures}")
        logger.info(f"  Total cost: ${total_cost:,.0f}")
        logger.info(f"  Fleet availability: {availability:.1f}%")
        
        return results
    
    def simulate_predictive_maintenance(self,
                                       rul_predictions: np.ndarray,
                                       actual_rul: np.ndarray,
                                       threshold: int = None) -> Dict:
        """
        Simulate predictive maintenance based on RUL predictions
        
        Args:
            rul_predictions: Predicted RUL values
            actual_rul: Actual RUL values
            threshold: RUL threshold for triggering maintenance
            
        Returns:
            Dictionary with simulation results
        """
        if threshold is None:
            threshold = self.thresholds['critical']
        
        logger.info(f"Simulating predictive maintenance (RUL threshold = {threshold})...")
        
        n_engines = len(rul_predictions)
        scheduled_maintenances = 0
        unexpected_failures = 0
        false_alarms = 0
        
        for pred_rul, true_rul in zip(rul_predictions, actual_rul):
            # If prediction triggers maintenance
            if pred_rul < threshold:
                scheduled_maintenances += 1
                
                # Check if it was truly needed
                if true_rul ≥ threshold:
                    false_alarms += 1  # Over-conservative prediction
                # else: True positive - correctly identified failing engine
            else:
                # No maintenance scheduled
                if true_rul < threshold:
                    unexpected_failures += 1  # Missed a failing engine
        
        # Calculate costs
        scheduled_cost = scheduled_maintenances * self.costs['scheduled_maintenance']
        unscheduled_cost = unexpected_failures * self.costs['unscheduled_maintenance']
        false_alarm_cost = false_alarms * self.costs['false_alarm_cost']
        total_cost = scheduled_cost + unscheduled_cost + false_alarm_cost
        
        # Fleet availability
        availability = (n_engines - unexpected_failures) / n_engines * 100
        
        results = {
            'method': 'Predictive (AI-Driven)',
            'threshold': threshold,
            'total_engines': n_engines,
            'scheduled_maintenances': scheduled_maintenances,
            'unexpected_failures': unexpected_failures,
            'false_alarms': false_alarms,
            'scheduled_cost': scheduled_cost,
            'unscheduled_cost': unscheduled_cost,
            'false_alarm_cost': false_alarm_cost,
            'total_cost': total_cost,
            'fleet_availability': availability
        }
        
        logger.info(f"  Total maintenances: {scheduled_maintenances}")
        logger.info(f"  Unexpected failures: {unexpected_failures}")
        logger.info(f"  False alarms: {false_alarms}")
        logger.info(f"  Total cost: ${total_cost:,.0f}")
        logger.info(f"  Fleet availability: {availability:.1f}%")
        
        return results
    
    def compare_strategies(self, 
                          traditional_results: Dict, 
                          predictive_results: Dict) -> pd.DataFrame:
        """
        Compare traditional vs predictive maintenance strategies
        
        Args:
            traditional_results: Results from traditional maintenance simulation
            predictive_results: Results from predictive maintenance simulation
            
        Returns:
            DataFrame with comparison
        """
        logger.info("="*60)
        logger.info("Maintenance Strategy Comparison")
        logger.info("="*60)
        
        comparison = pd.DataFrame([traditional_results, predictive_results])
        
        # Calculate improvements
        cost_reduction = traditional_results['total_cost'] - predictive_results['total_cost']
        cost_reduction_pct = (cost_reduction / traditional_results['total_cost']) * 100
        
        availability_improvement = (predictive_results['fleet_availability'] - 
                                   traditional_results['fleet_availability'])
        
        logger.info(f"\nCost Reduction: ${cost_reduction:,.0f} ({cost_reduction_pct:.1f}%)")
        logger.info(f"Availability Improvement: {availability_improvement:.1f}%")
        logger.info(f"\nFrom ${traditional_results['total_cost']:,.0f} to ${predictive_results['total_cost']:,.0f}")
        logger.info(f"From {traditional_results['fleet_availability']:.1f}% to {predictive_results['fleet_availability']:.1f}%")
        
        return comparison


if __name__ == "__main__":
    # Test maintenance planner
    print("="*60)
    print("Testing Maintenance Planner")
    print("="*60)
    
    # Generate synthetic data
    np.random.seed(42)
    n_engines = 100
    
    # Simulated predictions and actuals
    rul_predictions = np.random.randint(0, 150, size=n_engines).astype(float)
    actual_rul = rul_predictions + np.random.randn(n_engines) * 10
    actual_rul = np.maximum(actual_rul, 0)
    
    # Create planner
    planner = MaintenancePlanner()
    
    # Test 1: Maintenance schedule
    print("\n1. Creating Maintenance Schedule:")
    predictions_df = pd.DataFrame({
        'unit_id': range(1, n_engines + 1),
        'RUL_pred': rul_predictions
    })
    schedule = planner.create_maintenance_schedule(predictions_df)
    print(f"\n{schedule.head(10)}")
    
    # Test 2: Traditional maintenance simulation
    print("\n2. Traditional Maintenance Simulation:")
    engine_lifetimes = np.random.randint(50, 200, size=n_engines)
    trad_results = planner.simulate_traditional_maintenance(engine_lifetimes)
    
    # Test 3: Predictive maintenance simulation
    print("\n3. Predictive Maintenance Simulation:")
    pred_results = planner.simulate_predictive_maintenance(rul_predictions, actual_rul)
    
    # Test 4: Comparison
    print("\n4. Strategy Comparison:")
    comparison = planner.compare_strategies(trad_results, pred_results)
    print(f"\n{comparison.to_string(index=False)}")
    
    print("\n" + "="*60)
    print("Maintenance planner test complete!")
    print("="*60)

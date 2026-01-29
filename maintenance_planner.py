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
                if true_rul >= threshold:
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


class EarlyWarningSystem:
    """
    Early Warning System for Proactive Maintenance Alerts
    
    Provides multi-level alerting based on:
    - Current RUL predictions
    - Rate of RUL degradation
    - Fleet-wide anomaly patterns
    - Confidence interval analysis
    """
    
    def __init__(self):
        """Initialize early warning system"""
        self.thresholds = config.MAINTENANCE_THRESHOLDS
        
        # Define alert levels with more granularity
        self.alert_levels = {
            'EMERGENCY': {'rul_max': 15, 'color': '🔴', 'priority': 1},
            'CRITICAL': {'rul_max': 30, 'color': '🟠', 'priority': 2},
            'WARNING': {'rul_max': 50, 'color': '🟡', 'priority': 3},
            'CAUTION': {'rul_max': 80, 'color': '🔵', 'priority': 4},
            'MONITOR': {'rul_max': float('inf'), 'color': '🟢', 'priority': 5}
        }
        
        logger.info("Initialized Early Warning System")
    
    def get_alert_level(self, rul: float) -> Tuple[str, Dict]:
        """
        Determine alert level based on RUL
        
        Args:
            rul: Remaining Useful Life in cycles
            
        Returns:
            Tuple of (alert_name, alert_details)
        """
        for level_name, level_info in self.alert_levels.items():
            if rul < level_info['rul_max']:
                return level_name, level_info
        return 'MONITOR', self.alert_levels['MONITOR']
    
    def analyze_degradation_rate(self,
                                  rul_history: np.ndarray,
                                  time_window: int = 10) -> Dict:
        """
        Analyze rate of RUL degradation over time
        
        Args:
            rul_history: Array of RUL predictions over time
            time_window: Number of recent observations to analyze
            
        Returns:
            Dictionary with degradation analysis
        """
        if len(rul_history) < 2:
            return {
                'rate': 0.0,
                'acceleration': 0.0,
                'trend': 'insufficient_data',
                'rapid_degradation': False
            }
        
        # Use recent window
        recent = rul_history[-time_window:] if len(rul_history) >= time_window else rul_history
        
        # Calculate degradation rate (cycles lost per time step)
        rate = np.mean(np.diff(recent)) if len(recent) > 1 else 0.0
        
        # Calculate acceleration (change in rate)
        if len(recent) > 2:
            rates = np.diff(recent)
            acceleration = np.mean(np.diff(rates))
        else:
            acceleration = 0.0
        
        # Determine trend
        if rate < -2.0:  # Losing more than 2 RUL cycles per time step
            trend = 'rapid_decline'
        elif rate < -1.0:
            trend = 'moderate_decline'
        elif rate < 0:
            trend = 'gradual_decline'
        else:
            trend = 'stable'
        
        # Flag rapid degradation
        rapid_degradation = (rate < -2.0) or (acceleration < -0.5)
        
        return {
            'rate': float(rate),
            'acceleration': float(acceleration),
            'trend': trend,
            'rapid_degradation': rapid_degradation
        }
    
    def generate_alerts(self,
                        predictions_df: pd.DataFrame,
                        rul_history_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Generate comprehensive alerts for all engines
        
        Args:
            predictions_df: DataFrame with [unit_id, RUL_pred, RUL_uncertainty (optional)]
            rul_history_df: Historical RUL predictions [unit_id, time_step, RUL_pred] (optional)
            
        Returns:
            DataFrame with alert information
        """
        logger.info("Generating early warning alerts...")
        
        alerts = []
        
        for _, row in predictions_df.iterrows():
            unit_id = row['unit_id']
            rul_pred = row['RUL_pred']
            
            # Get alert level
            alert_name, alert_info = self.get_alert_level(rul_pred)
            
            # Analyze degradation rate if history available
            degradation_info = {}
            if rul_history_df is not None:
                unit_history = rul_history_df[rul_history_df['unit_id'] == unit_id]['RUL_pred'].values
                degradation_info = self.analyze_degradation_rate(unit_history)
            
            # Check uncertainty if available
            high_uncertainty = False
            if 'RUL_uncertainty' in row:
                # Flag if uncertainty is > 20% of prediction
                high_uncertainty = row['RUL_uncertainty'] > (rul_pred * 0.2)
            
            # Determine action urgency
            if alert_name == 'EMERGENCY':
                action = 'IMMEDIATE: Ground aircraft for emergency maintenance'
                estimated_downtime = '8-12 hours'
            elif alert_name == 'CRITICAL':
                action = 'URGENT: Schedule maintenance within 48 hours'
                estimated_downtime = '4-6 hours'
            elif alert_name == 'WARNING':
                action = 'PRIORITY: Plan maintenance for this week'
                estimated_downtime = '2-4 hours'
            elif alert_name == 'CAUTION':
                action = 'NOTICE: Include in next maintenance window'
                estimated_downtime = 'Standard maintenance'
            else:
                action = 'ROUTINE: Continue monitoring'
                estimated_downtime = 'N/A'
            
            alert_record = {
                'unit_id': unit_id,
                'alert_level': alert_name,
                'alert_icon': alert_info['color'],
                'priority': alert_info['priority'],
                'rul_predicted': rul_pred,
                'recommended_action': action,
                'estimated_downtime': estimated_downtime,
                'high_uncertainty_flag': high_uncertainty
            }
            
            # Add degradation info if available
            if degradation_info:
                alert_record.update({
                    'degradation_rate': degradation_info['rate'],
                    'degradation_trend': degradation_info['trend'],
                    'rapid_degradation_flag': degradation_info['rapid_degradation']
                })
            
            alerts.append(alert_record)
        
        alerts_df = pd.DataFrame(alerts)
        alerts_df = alerts_df.sort_values('priority')
        
        # Summary
        alert_counts = alerts_df['alert_level'].value_counts()
        logger.info("Alert Summary:")
        for level in ['EMERGENCY', 'CRITICAL', 'WARNING', 'CAUTION', 'MONITOR']:
            count = alert_counts.get(level, 0)
            icon = self.alert_levels[level]['color']
            logger.info(f"  {icon} {level}: {count} engines")
        
        return alerts_df
    
    def fleet_health_score(self, predictions_df: pd.DataFrame) -> Dict:
        """
        Calculate overall fleet health score
        
        Args:
            predictions_df: DataFrame with [unit_id, RUL_pred]
            
        Returns:
            Dictionary with fleet health metrics
        """
        total_engines = len(predictions_df)
        avg_rul = predictions_df['RUL_pred'].mean()
        min_rul = predictions_df['RUL_pred'].min()
        
        # Count engines in each category
        emergency = (predictions_df['RUL_pred'] < 15).sum()
        critical = ((predictions_df['RUL_pred'] >= 15) & (predictions_df['RUL_pred'] < 30)).sum()
        warning = ((predictions_df['RUL_pred'] >= 30) & (predictions_df['RUL_pred'] < 50)).sum()
        
        # Calculate health score (0-100)
        # Weight: EMERGENCY = -20 per engine, CRITICAL = -10, WARNING = -5
        penalty = emergency * 20 + critical * 10 + warning * 5
        max_penalty = total_engines * 20  # All engines in emergency
        health_score = max(0, 100 - (penalty / max_penalty * 100))
        
        # Determine fleet status
        if emergency > 0:
            fleet_status = 'CRITICAL - Immediate attention required'
        elif critical > 0:
            fleet_status = 'AT RISK - Schedule maintenance urgently'
        elif warning > 0:
            fleet_status = 'WATCH - Plan maintenance activities'
        else:
            fleet_status = 'HEALTHY - Routine monitoring'
        
        return {
            'total_engines': total_engines,
            'fleet_health_score': round(health_score, 1),
            'fleet_status': fleet_status,
            'avg_rul_cycles': round(avg_rul, 1),
            'min_rul_cycles': round(min_rul, 1),
            'engines_emergency': emergency,
            'engines_critical': critical,
            'engines_warning': warning,
            'engines_healthy': total_engines - emergency - critical - warning
        }
    
    def priority_maintenance_queue(self,
                                    alerts_df: pd.DataFrame,
                                    max_concurrent: int = 5) -> List[Dict]:
        """
        Generate prioritized maintenance queue
        
        Args:
            alerts_df: DataFrame from generate_alerts()
            max_concurrent: Maximum concurrent maintenance slots
            
        Returns:
            List of maintenance batches in priority order
        """
        queue = []
        remaining = alerts_df[alerts_df['alert_level'] != 'MONITOR'].copy()
        batch_num = 1
        
        while len(remaining) > 0:
            # Select top priority engines for this batch
            batch = remaining.nsmallest(max_concurrent, 'priority')
            
            queue.append({
                'batch_number': batch_num,
                'engines': batch['unit_id'].tolist(),
                'alert_levels': batch['alert_level'].tolist(),
                'estimated_duration': f"{len(batch) * 4}-{len(batch) * 8} hours"
            })
            
            remaining = remaining[~remaining['unit_id'].isin(batch['unit_id'])]
            batch_num += 1
        
        logger.info(f"Created maintenance queue with {len(queue)} batches")
        return queue


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

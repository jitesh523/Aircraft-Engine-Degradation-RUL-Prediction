"""
What-If Scenario Simulator for Maintenance Decision Support.

Simulates counterfactual scenarios — delayed maintenance, accelerated
sensor drift, and fleet-wide strategy comparisons — to help operators
make data-driven maintenance decisions.

Classes:
    WhatIfSimulator — Scenario engine with Plotly visualizations.

Usage::

    from whatif_simulator import WhatIfSimulator
    sim = WhatIfSimulator()
    result = sim.simulate_delayed_maintenance(current_rul=60, delay_cycles=30)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import config
from utils import setup_logging

logger = setup_logging(__name__)


class WhatIfSimulator:
    """
    Interactive counterfactual simulator for maintenance decision support.
    
    Answers questions like:
    - "What if we delay maintenance by N cycles?"
    - "What if sensor_X degrades 2× faster?"
    - "What are the fleet-wide cost implications of different strategies?"
    """
    
    def __init__(self):
        """Initialize simulator."""
        self.degradation_models = {}
        self.cost_params = config.COST_PARAMETERS
        self.thresholds = config.MAINTENANCE_THRESHOLDS
        self.scenarios = []
        logger.info("WhatIfSimulator initialized")
    
    def learn_degradation_rates(self, train_df: pd.DataFrame):
        """
        Learn per-sensor degradation rates from training data.
        
        Args:
            train_df: Training data with RUL and sensors.
        """
        sensor_cols = [c for c in train_df.columns if c.startswith('sensor_')]
        
        for col in sensor_cols:
            rates = []
            for uid, grp in train_df.groupby('unit_id'):
                if len(grp) < 10:
                    continue
                # Linear fit of sensor vs cycle
                x = grp['time_cycles'].values
                y = grp[col].values
                if np.std(y) > 1e-8:
                    slope = np.polyfit(x, y, 1)[0]
                    rates.append(slope)
            
            if rates:
                self.degradation_models[col] = {
                    'mean_rate': float(np.mean(rates)),
                    'std_rate': float(np.std(rates)),
                    'min_rate': float(np.min(rates)),
                    'max_rate': float(np.max(rates))
                }
        
        logger.info(f"Learned degradation rates for {len(self.degradation_models)} sensors")
    
    # ------------------------------------------------------------------
    # Scenario simulators
    # ------------------------------------------------------------------
    def simulate_delayed_maintenance(self, 
                                      current_rul: float,
                                      delay_cycles: int,
                                      degradation_rate: float = -1.0) -> Dict:
        """
        Simulate what happens if maintenance is delayed by N cycles.
        
        Args:
            current_rul: Current estimated RUL.
            delay_cycles: Number of cycles to delay maintenance.
            degradation_rate: RUL decrease per cycle (negative).
            
        Returns:
            Scenario outcome dict.
        """
        timeline = np.arange(0, delay_cycles + 1)
        rul_trajectory = current_rul + degradation_rate * timeline
        # Add noise
        noise = np.random.normal(0, abs(degradation_rate) * 0.1, len(timeline))
        rul_trajectory += noise
        rul_trajectory = np.maximum(rul_trajectory, 0)
        
        # Determine outcomes
        failure_cycle = None
        failure_occurred = False
        for i, rul in enumerate(rul_trajectory):
            if rul <= 0:
                failure_cycle = i
                failure_occurred = True
                break
        
        final_rul = rul_trajectory[-1]
        
        # Cost analysis
        if failure_occurred:
            cost = self.cost_params.get('unscheduled_maintenance', 50000)
            risk = 'CRITICAL'
        elif final_rul < self.thresholds['critical']:
            cost = self.cost_params.get('unscheduled_maintenance', 50000) * 0.8
            risk = 'HIGH'
        elif final_rul < self.thresholds['warning']:
            cost = self.cost_params.get('scheduled_maintenance', 10000)
            risk = 'MEDIUM'
        else:
            cost = self.cost_params.get('scheduled_maintenance', 10000) * 0.5
            risk = 'LOW'
        
        scenario = {
            'type': 'delayed_maintenance',
            'current_rul': float(current_rul),
            'delay_cycles': delay_cycles,
            'final_rul': float(final_rul),
            'failure_occurred': failure_occurred,
            'failure_cycle': failure_cycle,
            'risk_level': risk,
            'estimated_cost': float(cost),
            'rul_trajectory': rul_trajectory.tolist(),
            'timeline': timeline.tolist()
        }
        
        self.scenarios.append(scenario)
        logger.info(f"Delayed maintenance: RUL {current_rul:.0f} → {final_rul:.0f} "
                    f"after {delay_cycles}cy delay (risk: {risk})")
        
        return scenario
    
    def simulate_sensor_drift(self,
                               engine_data: pd.DataFrame,
                               sensor: str,
                               drift_multiplier: float = 2.0,
                               n_cycles: int = 50) -> Dict:
        """
        Simulate accelerated or decelerated degradation on a specific sensor.
        
        Args:
            engine_data: Current engine sensor data.
            sensor: Sensor name to apply drift to.
            drift_multiplier: >1 = faster degradation, <1 = slower.
            n_cycles: Number of future cycles to simulate.
            
        Returns:
            Scenario outcome dict.
        """
        if sensor not in self.degradation_models:
            logger.warning(f"No degradation model for {sensor}")
            return {'error': f'No model for {sensor}'}
        
        model = self.degradation_models[sensor]
        base_rate = model['mean_rate']
        modified_rate = base_rate * drift_multiplier
        
        # Current value
        current_val = engine_data[sensor].iloc[-1] if sensor in engine_data.columns else 0
        
        timeline = np.arange(0, n_cycles + 1)
        
        # Normal trajectory
        normal = current_val + base_rate * timeline
        normal += np.random.normal(0, model['std_rate'] * 0.5, len(timeline))
        
        # Modified trajectory
        modified = current_val + modified_rate * timeline
        modified += np.random.normal(0, model['std_rate'] * 0.5, len(timeline))
        
        scenario = {
            'type': 'sensor_drift',
            'sensor': sensor,
            'drift_multiplier': drift_multiplier,
            'n_cycles': n_cycles,
            'base_rate': float(base_rate),
            'modified_rate': float(modified_rate),
            'normal_trajectory': normal.tolist(),
            'modified_trajectory': modified.tolist(),
            'timeline': timeline.tolist(),
            'final_normal': float(normal[-1]),
            'final_modified': float(modified[-1]),
            'delta_pct': float((modified[-1] - normal[-1]) / abs(normal[-1]) * 100)
                         if abs(normal[-1]) > 1e-8 else 0.0
        }
        
        self.scenarios.append(scenario)
        logger.info(f"Sensor drift ({sensor}): {drift_multiplier}× rate, "
                    f"Δ={scenario['delta_pct']:+.1f}% after {n_cycles}cy")
        
        return scenario
    
    def simulate_fleet_scenario(self,
                                 fleet_ruls: np.ndarray,
                                 strategy: str = 'proactive',
                                 horizon: int = 100) -> Dict:
        """
        Simulate fleet-wide outcomes under different maintenance strategies.
        
        Args:
            fleet_ruls: Array of current RUL estimates for all engines.
            strategy: 'proactive' (maintain at warning), 'reactive' (maintain at critical),
                      'fixed_interval' (every N cycles).
            horizon: Simulation horizon in cycles.
            
        Returns:
            Fleet scenario outcome.
        """
        n_engines = len(fleet_ruls)
        ruls = fleet_ruls.copy().astype(float)
        
        total_cost = 0
        failures = 0
        maintenances = 0
        availability_log = []
        
        for cycle in range(horizon):
            # Degrade
            ruls -= np.random.uniform(0.5, 1.5, n_engines)
            
            # Apply strategy
            if strategy == 'proactive':
                mask = ruls < self.thresholds['warning']
            elif strategy == 'reactive':
                mask = ruls < self.thresholds['critical']
            else:  # fixed_interval
                mask = np.zeros(n_engines, dtype=bool)
                if cycle > 0 and cycle % 50 == 0:
                    mask[:] = True
            
            # Check failures
            failed = ruls <= 0
            n_failed = failed.sum()
            if n_failed > 0:
                failures += n_failed
                total_cost += n_failed * self.cost_params.get('unscheduled_maintenance', 50000)
                ruls[failed] = np.random.uniform(100, 150, n_failed)
            
            # Perform maintenance
            maint_needed = mask & ~failed
            n_maint = maint_needed.sum()
            if n_maint > 0:
                maintenances += n_maint
                total_cost += n_maint * self.cost_params.get('scheduled_maintenance', 10000)
                ruls[maint_needed] = np.random.uniform(100, 150, n_maint)
            
            availability = (ruls > 0).sum() / n_engines
            availability_log.append(float(availability))
        
        scenario = {
            'type': 'fleet_scenario',
            'strategy': strategy,
            'horizon': horizon,
            'n_engines': n_engines,
            'total_cost': float(total_cost),
            'total_failures': int(failures),
            'total_maintenances': int(maintenances),
            'avg_availability': float(np.mean(availability_log)),
            'cost_per_engine': float(total_cost / n_engines),
            'availability_log': availability_log
        }
        
        self.scenarios.append(scenario)
        logger.info(f"Fleet scenario ({strategy}): cost=${total_cost:,.0f}, "
                    f"{failures} failures, {np.mean(availability_log):.1%} availability")
        
        return scenario
    
    def compare_scenarios(self, scenarios: List[Dict] = None) -> pd.DataFrame:
        """
        Compare multiple scenarios side by side.
        
        Args:
            scenarios: List of scenario dicts. If None, uses all recorded scenarios.
            
        Returns:
            Comparison DataFrame.
        """
        scenarios = scenarios or self.scenarios
        
        rows = []
        for i, s in enumerate(scenarios):
            row = {'scenario': f"#{i+1}: {s.get('type', 'unknown')}"}
            
            if s['type'] == 'delayed_maintenance':
                row['delay'] = s.get('delay_cycles', 0)
                row['risk'] = s.get('risk_level', '?')
                row['cost'] = s.get('estimated_cost', 0)
                row['failure'] = s.get('failure_occurred', False)
            elif s['type'] == 'fleet_scenario':
                row['strategy'] = s.get('strategy', '?')
                row['cost'] = s.get('total_cost', 0)
                row['failures'] = s.get('total_failures', 0)
                row['availability'] = s.get('avg_availability', 0)
            elif s['type'] == 'sensor_drift':
                row['sensor'] = s.get('sensor', '?')
                row['multiplier'] = s.get('drift_multiplier', 1)
                row['delta_pct'] = s.get('delta_pct', 0)
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def plot_delayed_maintenance(self, scenario: Dict,
                                  title: str = "Delayed Maintenance Projection") -> go.Figure:
        """Plot RUL trajectory for a delayed maintenance scenario."""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=scenario['timeline'], y=scenario['rul_trajectory'],
            mode='lines', name='Projected RUL',
            line=dict(color='#1f77b4', width=2.5)
        ))
        
        # Threshold lines
        fig.add_hline(y=self.thresholds['critical'], line_dash='dash',
                      line_color='red', annotation_text='Critical')
        fig.add_hline(y=self.thresholds['warning'], line_dash='dash',
                      line_color='orange', annotation_text='Warning')
        
        if scenario['failure_occurred']:
            fig.add_vline(x=scenario['failure_cycle'], line_dash='dot',
                          line_color='red',
                          annotation_text=f"FAILURE @ cycle {scenario['failure_cycle']}")
        
        fig.update_layout(
            title=title,
            xaxis_title='Delay Cycles',
            yaxis_title='Projected RUL',
            height=400
        )
        return fig
    
    def plot_fleet_comparison(self, scenarios: List[Dict],
                               title: str = "Strategy Comparison") -> go.Figure:
        """Compare fleet scenarios side by side."""
        fig = make_subplots(rows=1, cols=2,
                           subplot_titles=('Total Cost', 'Avg Availability'))
        
        strategies = [s.get('strategy', s.get('type', '?')) for s in scenarios]
        costs = [s.get('total_cost', s.get('estimated_cost', 0)) for s in scenarios]
        avails = [s.get('avg_availability', 0) * 100 for s in scenarios]
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        fig.add_trace(go.Bar(
            x=strategies, y=costs, name='Cost',
            marker_color=colors[:len(strategies)],
            text=[f"${c:,.0f}" for c in costs], textposition='outside'
        ), row=1, col=1)
        
        fig.add_trace(go.Bar(
            x=strategies, y=avails, name='Availability',
            marker_color=colors[:len(strategies)],
            text=[f"{a:.1f}%" for a in avails], textposition='outside'
        ), row=1, col=2)
        
        fig.update_layout(title=title, height=400, showlegend=False)
        fig.update_yaxes(title_text='Cost ($)', row=1, col=1)
        fig.update_yaxes(title_text='Availability (%)', row=1, col=2)
        return fig
    
    def plot_sensor_drift_comparison(self, scenario: Dict,
                                      title: str = "Sensor Drift Projection") -> go.Figure:
        """Plot normal vs modified sensor trajectory."""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=scenario['timeline'], y=scenario['normal_trajectory'],
            mode='lines', name='Normal',
            line=dict(color='#2ca02c', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=scenario['timeline'], y=scenario['modified_trajectory'],
            mode='lines', name=f"{scenario['drift_multiplier']}× drift",
            line=dict(color='#d62728', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title=f"{title} ({scenario['sensor']})",
            xaxis_title='Future Cycles',
            yaxis_title='Sensor Value',
            height=400
        )
        return fig


# ============================================================
# Standalone test
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Testing What-If Scenario Simulator")
    print("=" * 60)
    
    from data_loader import CMAPSSDataLoader
    from utils import add_remaining_useful_life
    
    loader = CMAPSSDataLoader('FD001')
    train_df, _, _ = loader.load_all_data()
    train_df = add_remaining_useful_life(train_df)
    
    sim = WhatIfSimulator()
    
    # Learn degradation rates
    print("\n--- Learning Degradation Rates ---")
    sim.learn_degradation_rates(train_df)
    print(f"Modeled sensors: {len(sim.degradation_models)}")
    
    # Delayed maintenance scenarios
    print("\n--- Delayed Maintenance ---")
    for delay in [10, 30, 50, 80]:
        result = sim.simulate_delayed_maintenance(
            current_rul=60, delay_cycles=delay
        )
        print(f"  Delay {delay}cy: RUL→{result['final_rul']:.0f}, "
              f"risk={result['risk_level']}, cost=${result['estimated_cost']:,.0f}")
    
    # Sensor drift
    print("\n--- Sensor Drift ---")
    engine_1 = train_df[train_df['unit_id'] == 1]
    drift = sim.simulate_sensor_drift(engine_1, 'sensor_7', drift_multiplier=2.0, n_cycles=50)
    print(f"  sensor_7 @ 2×: delta = {drift['delta_pct']:+.1f}%")
    
    # Fleet scenarios
    print("\n--- Fleet Scenarios ---")
    fleet_ruls = np.random.uniform(20, 150, 50)
    
    strategies = ['proactive', 'reactive', 'fixed_interval']
    fleet_results = []
    for strat in strategies:
        result = sim.simulate_fleet_scenario(fleet_ruls.copy(), strategy=strat)
        fleet_results.append(result)
        print(f"  {strat}: cost=${result['total_cost']:,.0f}, "
              f"failures={result['total_failures']}, "
              f"avail={result['avg_availability']:.1%}")
    
    # Compare
    print("\n--- Comparison ---")
    comp = sim.compare_scenarios()
    print(comp.to_string(index=False))
    
    print("\n✅ What-If Simulator test PASSED")

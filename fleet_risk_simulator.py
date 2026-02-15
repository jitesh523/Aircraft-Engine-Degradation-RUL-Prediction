"""
Fleet Risk Monte Carlo Simulator
Probabilistic fleet-wide risk assessment using Monte Carlo simulation
of failure scenarios, Value-at-Risk (VaR), and cost-at-risk analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import config
from utils import setup_logging

logger = setup_logging(__name__)


class FleetRiskSimulator:
    """
    Monte Carlo simulator for fleet-wide failure risk analysis.
    
    Simulates thousands of possible fleet scenarios to estimate:
    - Probability of N+ simultaneous failures
    - Value-at-Risk (VaR) and Conditional VaR (CVaR)
    - Cost-at-Risk distributions
    - Optimal spare parts and hangar capacity requirements
    """
    
    def __init__(self, n_simulations: int = 10000):
        """
        Args:
            n_simulations: Number of Monte Carlo simulation runs.
        """
        self.n_simulations = n_simulations
        self.cost_params = config.COST_PARAMETERS
        self.thresholds = config.MAINTENANCE_THRESHOLDS
        self.results = None
        self.fleet_data = None
        logger.info(f"FleetRiskSimulator initialized: {n_simulations:,} simulations")
    
    def setup_fleet(self, fleet_df: pd.DataFrame) -> Dict:
        """
        Configure fleet data for simulation.
        
        Args:
            fleet_df: DataFrame with engine_id, rul_pred, and optionally rul_std.
            
        Returns:
            Fleet statistics summary.
        """
        self.fleet_data = fleet_df.copy()
        n_engines = len(fleet_df)
        
        # Ensure required columns
        if 'rul_std' not in self.fleet_data.columns:
            self.fleet_data['rul_std'] = self.fleet_data.get(
                'rul_pred', self.fleet_data.get('RUL', 100)
            ) * 0.15  # 15% uncertainty
        
        if 'rul_pred' not in self.fleet_data.columns:
            self.fleet_data['rul_pred'] = self.fleet_data.get('RUL', 100)
        
        # Classify initial risk
        critical = (self.fleet_data['rul_pred'] < self.thresholds['critical']).sum()
        warning = ((self.fleet_data['rul_pred'] >= self.thresholds['critical']) & 
                   (self.fleet_data['rul_pred'] < self.thresholds['warning'])).sum()
        healthy = (self.fleet_data['rul_pred'] >= self.thresholds['warning']).sum()
        
        stats = {
            'n_engines': n_engines,
            'mean_rul': float(self.fleet_data['rul_pred'].mean()),
            'min_rul': float(self.fleet_data['rul_pred'].min()),
            'max_rul': float(self.fleet_data['rul_pred'].max()),
            'critical': int(critical),
            'warning': int(warning),
            'healthy': int(healthy)
        }
        
        logger.info(f"Fleet setup: {n_engines} engines — "
                    f"{critical} critical, {warning} warning, {healthy} healthy")
        
        return stats
    
    def run_simulation(self, horizon: int = 30) -> Dict:
        """
        Run Monte Carlo simulation of fleet failure scenarios.
        
        Args:
            horizon: Time horizon in cycles to simulate forward.
            
        Returns:
            Comprehensive risk analysis results.
        """
        if self.fleet_data is None:
            raise RuntimeError("Call setup_fleet() first.")
        
        n_engines = len(self.fleet_data)
        rul_pred = self.fleet_data['rul_pred'].values
        rul_std = self.fleet_data['rul_std'].values
        
        # Storage
        failure_counts = np.zeros(self.n_simulations)
        cost_scenarios = np.zeros(self.n_simulations)
        downtime_scenarios = np.zeros(self.n_simulations)
        failure_matrix = np.zeros((self.n_simulations, n_engines))
        
        cost_unsched = self.cost_params.get('unscheduled_maintenance', 50000)
        cost_sched = self.cost_params.get('scheduled_maintenance', 10000)
        
        for sim in range(self.n_simulations):
            # Sample actual RUL from prediction distribution
            sampled_rul = np.random.normal(rul_pred, rul_std)
            sampled_rul = np.maximum(sampled_rul, 0)
            
            # Which engines fail within horizon?
            failures = sampled_rul < horizon
            failure_matrix[sim] = failures.astype(float)
            n_failures = failures.sum()
            failure_counts[sim] = n_failures
            
            # Cost calculation
            scenario_cost = 0
            for i in range(n_engines):
                if failures[i]:
                    # Unscheduled if RUL < 10, otherwise scheduled
                    if sampled_rul[i] < 10:
                        scenario_cost += cost_unsched
                    else:
                        scenario_cost += cost_sched * 1.5  # rushed scheduled
                else:
                    scenario_cost += cost_sched * 0.1  # monitoring cost
            
            cost_scenarios[sim] = scenario_cost
            downtime_scenarios[sim] = n_failures * 3  # 3 cycles per repair
        
        # Compute risk metrics
        self.results = {
            'n_simulations': self.n_simulations,
            'horizon': horizon,
            'n_engines': n_engines,
            
            # Failure statistics
            'mean_failures': float(np.mean(failure_counts)),
            'max_failures': int(np.max(failure_counts)),
            'std_failures': float(np.std(failure_counts)),
            'p_zero_failures': float(np.mean(failure_counts == 0)),
            'p_one_plus': float(np.mean(failure_counts >= 1)),
            'p_three_plus': float(np.mean(failure_counts >= 3)),
            'p_five_plus': float(np.mean(failure_counts >= 5)),
            
            # Cost-at-Risk
            'mean_cost': float(np.mean(cost_scenarios)),
            'var_95': float(np.percentile(cost_scenarios, 95)),
            'var_99': float(np.percentile(cost_scenarios, 99)),
            'cvar_95': float(np.mean(cost_scenarios[cost_scenarios >= np.percentile(cost_scenarios, 95)])),
            'min_cost': float(np.min(cost_scenarios)),
            'max_cost': float(np.max(cost_scenarios)),
            
            # Downtime
            'mean_downtime': float(np.mean(downtime_scenarios)),
            'max_downtime': int(np.max(downtime_scenarios)),
            
            # Per-engine failure probability
            'engine_failure_prob': failure_matrix.mean(axis=0).tolist(),
            
            # Raw distributions (for plotting)
            '_failure_counts': failure_counts,
            '_cost_scenarios': cost_scenarios,
            '_downtime_scenarios': downtime_scenarios,
        }
        
        logger.info(f"MC simulation complete ({self.n_simulations:,} runs, {horizon}cy horizon): "
                    f"mean failures={self.results['mean_failures']:.1f}, "
                    f"VaR95=${self.results['var_95']:,.0f}")
        
        return self.results
    
    def sensitivity_analysis(self, horizons: List[int] = None) -> pd.DataFrame:
        """
        Run simulations across multiple horizons.
        
        Returns:
            DataFrame of risk metrics per horizon.
        """
        horizons = horizons or [10, 20, 30, 50, 80, 100]
        
        rows = []
        for h in horizons:
            res = self.run_simulation(horizon=h)
            rows.append({
                'horizon': h,
                'mean_failures': res['mean_failures'],
                'p_zero_failures': res['p_zero_failures'],
                'p_3plus': res['p_three_plus'],
                'mean_cost': res['mean_cost'],
                'var_95': res['var_95'],
                'cvar_95': res['cvar_95'],
                'mean_downtime': res['mean_downtime'],
            })
        
        return pd.DataFrame(rows)
    
    def spare_parts_analysis(self, max_spares: int = 10) -> pd.DataFrame:
        """
        Determine optimal number of spare parts to stock.
        
        Returns:
            DataFrame showing service level for each spare count.
        """
        if self.results is None:
            self.run_simulation()
        
        failures = self.results['_failure_counts']
        
        rows = []
        for n_spares in range(0, max_spares + 1):
            covered = np.mean(failures <= n_spares)
            uncovered_cost = np.mean(
                np.maximum(failures - n_spares, 0) * 
                self.cost_params.get('unscheduled_maintenance', 50000)
            )
            holding_cost = n_spares * 5000  # $5K per spare
            total_cost = uncovered_cost + holding_cost
            
            rows.append({
                'n_spares': n_spares,
                'service_level': float(covered),
                'uncovered_cost': float(uncovered_cost),
                'holding_cost': float(holding_cost),
                'total_cost': float(total_cost),
            })
        
        df = pd.DataFrame(rows)
        optimal = df.loc[df['total_cost'].idxmin()]
        logger.info(f"Optimal spares: {int(optimal['n_spares'])} "
                    f"(service level: {optimal['service_level']:.1%})")
        
        return df
    
    # ------------------------------------------------------------------
    # Plotly visualizations
    # ------------------------------------------------------------------
    def plot_risk_dashboard(self, title: str = "Fleet Risk Overview") -> go.Figure:
        """Comprehensive risk dashboard with 4 subplots."""
        if self.results is None:
            raise RuntimeError("Call run_simulation() first.")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Failure Count Distribution',
                'Cost-at-Risk Distribution',
                'Per-Engine Failure Probability',
                'Downtime Distribution'
            ),
            vertical_spacing=0.12, horizontal_spacing=0.1
        )
        
        # 1. Failure count histogram
        fig.add_trace(go.Histogram(
            x=self.results['_failure_counts'], nbinsx=30,
            marker_color='#e74c3c', opacity=0.7, name='Failures'
        ), row=1, col=1)
        
        # 2. Cost distribution
        fig.add_trace(go.Histogram(
            x=self.results['_cost_scenarios'], nbinsx=50,
            marker_color='#3498db', opacity=0.7, name='Cost'
        ), row=1, col=2)
        
        # VaR line
        fig.add_vline(x=self.results['var_95'], row=1, col=2,
                      line_dash='dash', line_color='red',
                      annotation_text=f"VaR95: ${self.results['var_95']:,.0f}")
        
        # 3. Per-engine failure probability bar chart
        probs = self.results['engine_failure_prob']
        eng_ids = [f'E{i+1}' for i in range(len(probs))]
        colors = ['#e74c3c' if p > 0.5 else '#f39c12' if p > 0.2 else '#27ae60' 
                  for p in probs]
        fig.add_trace(go.Bar(
            x=eng_ids, y=probs,
            marker_color=colors, name='P(Failure)'
        ), row=2, col=1)
        
        # 4. Downtime histogram
        fig.add_trace(go.Histogram(
            x=self.results['_downtime_scenarios'], nbinsx=30,
            marker_color='#9b59b6', opacity=0.7, name='Downtime'
        ), row=2, col=2)
        
        fig.update_layout(title=title, height=700, showlegend=False)
        fig.update_xaxes(title_text='# Failures', row=1, col=1)
        fig.update_xaxes(title_text='Cost ($)', row=1, col=2)
        fig.update_xaxes(title_text='Engine', row=2, col=1)
        fig.update_xaxes(title_text='Downtime (cycles)', row=2, col=2)
        fig.update_yaxes(title_text='Count', row=1, col=1)
        fig.update_yaxes(title_text='Count', row=1, col=2)
        fig.update_yaxes(title_text='P(failure)', row=2, col=1)
        fig.update_yaxes(title_text='Count', row=2, col=2)
        
        return fig
    
    def plot_sensitivity(self, sensitivity_df: pd.DataFrame,
                          title: str = "Risk Sensitivity Analysis") -> go.Figure:
        """Line charts showing how risk changes with horizon."""
        fig = make_subplots(rows=1, cols=2,
                           subplot_titles=('Failure Probability', 'Cost-at-Risk'))
        
        fig.add_trace(go.Scatter(
            x=sensitivity_df['horizon'], y=sensitivity_df['p_3plus'],
            mode='lines+markers', name='P(3+ failures)',
            line=dict(color='#e74c3c', width=2)
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=sensitivity_df['horizon'], y=sensitivity_df['p_zero_failures'],
            mode='lines+markers', name='P(zero failures)',
            line=dict(color='#27ae60', width=2)
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=sensitivity_df['horizon'], y=sensitivity_df['var_95'],
            mode='lines+markers', name='VaR 95%',
            line=dict(color='#3498db', width=2)
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=sensitivity_df['horizon'], y=sensitivity_df['cvar_95'],
            mode='lines+markers', name='CVaR 95%',
            line=dict(color='#e67e22', width=2, dash='dash')
        ), row=1, col=2)
        
        fig.update_xaxes(title_text='Horizon (cycles)', row=1, col=1)
        fig.update_xaxes(title_text='Horizon (cycles)', row=1, col=2)
        fig.update_yaxes(title_text='Probability', row=1, col=1)
        fig.update_yaxes(title_text='Cost ($)', row=1, col=2)
        fig.update_layout(title=title, height=400)
        
        return fig
    
    def plot_spare_parts(self, spares_df: pd.DataFrame,
                          title: str = "Spare Parts Optimization") -> go.Figure:
        """Show optimal spare parts stocking level."""
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(go.Bar(
            x=spares_df['n_spares'], y=spares_df['total_cost'],
            name='Total Cost', marker_color='#3498db', opacity=0.6
        ), secondary_y=False)
        
        fig.add_trace(go.Scatter(
            x=spares_df['n_spares'], y=spares_df['service_level'] * 100,
            name='Service Level', mode='lines+markers',
            line=dict(color='#27ae60', width=3)
        ), secondary_y=True)
        
        # Mark optimal
        optimal = spares_df.loc[spares_df['total_cost'].idxmin()]
        fig.add_vline(x=optimal['n_spares'], line_dash='dash',
                      line_color='red',
                      annotation_text=f"Optimal: {int(optimal['n_spares'])} spares")
        
        fig.update_xaxes(title_text='Number of Spares')
        fig.update_yaxes(title_text='Total Cost ($)', secondary_y=False)
        fig.update_yaxes(title_text='Service Level (%)', secondary_y=True)
        fig.update_layout(title=title, height=400)
        
        return fig


# ============================================================
# Standalone test
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Fleet Risk Monte Carlo Simulator")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Create fleet
    fleet = pd.DataFrame({
        'engine_id': [f'ENG-{i:03d}' for i in range(1, 41)],
        'rul_pred': np.concatenate([
            np.random.randint(5, 30, 8),    # 8 critical
            np.random.randint(30, 80, 12),   # 12 warning
            np.random.randint(80, 200, 20),  # 20 healthy
        ]).astype(float),
        'rul_std': np.random.uniform(5, 20, 40)
    })
    
    sim = FleetRiskSimulator(n_simulations=10000)
    
    print("\n--- Fleet Setup ---")
    stats = sim.setup_fleet(fleet)
    print(f"  Engines: {stats['n_engines']}")
    print(f"  Critical: {stats['critical']}, Warning: {stats['warning']}, Healthy: {stats['healthy']}")
    
    print("\n--- Monte Carlo Simulation (30-cycle horizon) ---")
    results = sim.run_simulation(horizon=30)
    print(f"  Mean failures: {results['mean_failures']:.1f}")
    print(f"  P(zero): {results['p_zero_failures']:.1%}")
    print(f"  P(3+ failures): {results['p_three_plus']:.1%}")
    print(f"  P(5+ failures): {results['p_five_plus']:.1%}")
    print(f"  Mean cost: ${results['mean_cost']:,.0f}")
    print(f"  VaR 95%: ${results['var_95']:,.0f}")
    print(f"  CVaR 95%: ${results['cvar_95']:,.0f}")
    
    print("\n--- Sensitivity Analysis ---")
    sens = sim.sensitivity_analysis([10, 30, 50, 100])
    print(sens.to_string(index=False))
    
    print("\n--- Spare Parts Optimization ---")
    spares = sim.spare_parts_analysis(max_spares=8)
    optimal = spares.loc[spares['total_cost'].idxmin()]
    print(f"  Optimal spares: {int(optimal['n_spares'])} "
          f"(service level: {optimal['service_level']:.1%}, "
          f"cost: ${optimal['total_cost']:,.0f})")
    
    print("\n✅ Fleet Risk Simulator test PASSED")

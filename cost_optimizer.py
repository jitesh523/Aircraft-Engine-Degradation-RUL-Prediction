"""
Maintenance Cost Optimizer
Multi-objective Pareto optimization balancing cost, risk, and fleet
availability under budget and capacity constraints.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import config
from utils import setup_logging

logger = setup_logging(__name__)


class CostOptimizer:
    """
    Multi-objective maintenance cost optimizer using Pareto frontier analysis.
    
    Objectives:
    1. Minimize total maintenance cost
    2. Minimize fleet-wide failure risk
    3. Maximize fleet availability
    
    Constraints:
    - Budget cap
    - Maintenance hangar capacity
    - Minimum safety RUL threshold
    """
    
    def __init__(self, budget_cap: float = 500000,
                 hangar_capacity: int = 5,
                 safety_rul: int = 20):
        """
        Args:
            budget_cap: Maximum maintenance budget ($).
            hangar_capacity: Max engines in maintenance simultaneously.
            safety_rul: Minimum RUL before mandatory maintenance.
        """
        self.budget_cap = budget_cap
        self.hangar_capacity = hangar_capacity
        self.safety_rul = safety_rul
        self.cost_params = config.COST_PARAMETERS
        self.pareto_front = None
        self.all_solutions = None
        logger.info(f"CostOptimizer initialized: budget=${budget_cap:,.0f}, "
                    f"capacity={hangar_capacity}, safety_rul={safety_rul}")
    
    def generate_solutions(self, fleet_df: pd.DataFrame,
                            n_solutions: int = 500) -> pd.DataFrame:
        """
        Generate candidate maintenance schedules via random sampling.
        
        Each solution is a binary vector: which engines to maintain now.
        
        Args:
            fleet_df: Fleet data with engine_id and rul_pred.
            n_solutions: Number of candidate solutions to generate.
            
        Returns:
            DataFrame of all candidate solutions with objectives.
        """
        rul_col = 'rul_pred' if 'rul_pred' in fleet_df.columns else 'RUL'
        n_engines = len(fleet_df)
        ruls = fleet_df[rul_col].values
        
        cost_sched = self.cost_params.get('scheduled_maintenance', 10000)
        cost_unsched = self.cost_params.get('unscheduled_maintenance', 50000)
        
        solutions = []
        
        for i in range(n_solutions):
            # Random maintenance decision for each engine
            if i == 0:
                # Solution 0: maintain nothing
                maintain = np.zeros(n_engines, dtype=bool)
            elif i == 1:
                # Solution 1: maintain all critical
                maintain = ruls < self.safety_rul * 2
            else:
                # Random probability of maintaining each engine
                # Bias towards maintaining low-RUL engines
                prob = np.clip(1.0 - ruls / 200.0, 0.05, 0.95)
                maintain = np.random.random(n_engines) < prob * np.random.uniform(0.3, 1.5)
            
            # Enforce capacity constraint
            if maintain.sum() > self.hangar_capacity:
                # Keep the most critical engines
                priorities = np.argsort(ruls)
                maintain[:] = False
                maintain[priorities[:self.hangar_capacity]] = True
            
            # Calculate objectives
            total_cost = maintain.sum() * cost_sched
            
            # Risk: expected failure cost for un-maintained engines
            unmaintained_ruls = ruls[~maintain]
            risk = 0
            for rul in unmaintained_ruls:
                # Probability of failure in next 30 cycles
                p_fail = max(0, 1 - rul / 60)
                risk += p_fail * cost_unsched
            
            # Mandatory maintenance for safety violations
            mandatory = ruls < self.safety_rul
            mandatory_not_maintained = mandatory & ~maintain
            risk += mandatory_not_maintained.sum() * cost_unsched * 2
            
            # Availability: fraction of fleet not in maintenance
            availability = 1.0 - maintain.sum() / n_engines
            
            # Budget feasibility
            within_budget = total_cost <= self.budget_cap
            
            solutions.append({
                'solution_id': i,
                'n_maintained': int(maintain.sum()),
                'total_cost': float(total_cost),
                'risk_cost': float(risk),
                'combined_cost': float(total_cost + risk),
                'availability': float(availability),
                'within_budget': within_budget,
                'maintained_engines': maintain.tolist(),
                'safety_violations': int(mandatory_not_maintained.sum()),
            })
        
        self.all_solutions = pd.DataFrame(solutions)
        logger.info(f"Generated {n_solutions} candidate solutions: "
                    f"cost range ${self.all_solutions['total_cost'].min():,.0f}"
                    f"–${self.all_solutions['total_cost'].max():,.0f}")
        
        return self.all_solutions
    
    def find_pareto_front(self) -> pd.DataFrame:
        """
        Extract Pareto-optimal solutions (non-dominated front).
        
        A solution is Pareto-optimal if no other solution is better in
        all objectives simultaneously.
        
        Returns:
            DataFrame of Pareto-optimal solutions.
        """
        if self.all_solutions is None:
            raise RuntimeError("Call generate_solutions() first.")
        
        solutions = self.all_solutions[self.all_solutions['within_budget']].copy()
        
        # Objectives to minimize: combined_cost, risk_cost
        # Objective to maximize: availability (minimize negative)
        costs = solutions['combined_cost'].values
        risks = solutions['risk_cost'].values
        avail = -solutions['availability'].values  # negate to minimize
        
        n = len(solutions)
        is_pareto = np.ones(n, dtype=bool)
        
        for i in range(n):
            if not is_pareto[i]:
                continue
            for j in range(n):
                if i == j:
                    continue
                # Check if j dominates i
                if (costs[j] <= costs[i] and risks[j] <= risks[i] and avail[j] <= avail[i] and
                    (costs[j] < costs[i] or risks[j] < risks[i] or avail[j] < avail[i])):
                    is_pareto[i] = False
                    break
        
        self.pareto_front = solutions[is_pareto].sort_values('combined_cost')
        logger.info(f"Pareto front: {len(self.pareto_front)} solutions "
                    f"from {len(solutions)} feasible")
        
        return self.pareto_front
    
    def recommend_solution(self, preference: str = 'balanced') -> Dict:
        """
        Recommend a single solution based on preference.
        
        Args:
            preference: 'cost' (minimize cost), 'safety' (minimize risk),
                       'balanced' (best trade-off), 'availability' (max availability).
            
        Returns:
            Best solution dict.
        """
        if self.pareto_front is None:
            self.find_pareto_front()
        
        front = self.pareto_front
        
        if preference == 'cost':
            best = front.loc[front['total_cost'].idxmin()]
        elif preference == 'safety':
            best = front.loc[front['risk_cost'].idxmin()]
        elif preference == 'availability':
            best = front.loc[front['availability'].idxmax()]
        else:  # balanced
            # Normalize objectives and find min weighted sum
            norm_cost = (front['combined_cost'] - front['combined_cost'].min()) / max(front['combined_cost'].max() - front['combined_cost'].min(), 1)
            norm_risk = (front['risk_cost'] - front['risk_cost'].min()) / max(front['risk_cost'].max() - front['risk_cost'].min(), 1)
            norm_avail = 1 - front['availability']  # lower availability = worse
            
            score = 0.4 * norm_cost + 0.4 * norm_risk + 0.2 * norm_avail
            best = front.loc[score.idxmin()]
        
        rec = best.to_dict()
        rec['preference'] = preference
        
        logger.info(f"Recommended ({preference}): "
                    f"cost=${rec['total_cost']:,.0f}, "
                    f"risk=${rec['risk_cost']:,.0f}, "
                    f"avail={rec['availability']:.1%}")
        
        return rec
    
    def budget_sensitivity(self, fleet_df: pd.DataFrame,
                            budgets: List[float] = None) -> pd.DataFrame:
        """
        Analyze how optimal solutions change with budget.
        
        Returns:
            DataFrame of optimal metrics per budget level.
        """
        budgets = budgets or [50000, 100000, 200000, 300000, 500000, 750000, 1000000]
        
        rows = []
        for budget in budgets:
            self.budget_cap = budget
            self.generate_solutions(fleet_df, n_solutions=200)
            self.find_pareto_front()
            
            if len(self.pareto_front) > 0:
                rec = self.recommend_solution('balanced')
                rows.append({
                    'budget': budget,
                    'optimal_cost': rec['total_cost'],
                    'risk_cost': rec['risk_cost'],
                    'combined_cost': rec['combined_cost'],
                    'availability': rec['availability'],
                    'n_maintained': rec['n_maintained'],
                    'pareto_size': len(self.pareto_front),
                    'safety_violations': rec['safety_violations'],
                })
        
        return pd.DataFrame(rows)
    
    # ------------------------------------------------------------------
    # Plotly visualizations
    # ------------------------------------------------------------------
    def plot_pareto_front(self, title: str = "Pareto Frontier — Cost vs Risk") -> go.Figure:
        """Scatter plot of Pareto frontier."""
        if self.all_solutions is None:
            raise RuntimeError("Call generate_solutions() first.")
        if self.pareto_front is None:
            self.find_pareto_front()
        
        fig = go.Figure()
        
        # All feasible solutions
        feasible = self.all_solutions[self.all_solutions['within_budget']]
        fig.add_trace(go.Scatter(
            x=feasible['total_cost'], y=feasible['risk_cost'],
            mode='markers',
            marker=dict(size=6, color=feasible['availability'],
                       colorscale='RdYlGn', showscale=True,
                       colorbar=dict(title='Availability')),
            name='All Solutions',
            text=[f"ID: {r['solution_id']}<br>Maintained: {r['n_maintained']}<br>"
                  f"Avail: {r['availability']:.0%}" for _, r in feasible.iterrows()],
            hoverinfo='text'
        ))
        
        # Pareto front
        front_sorted = self.pareto_front.sort_values('total_cost')
        fig.add_trace(go.Scatter(
            x=front_sorted['total_cost'], y=front_sorted['risk_cost'],
            mode='lines+markers',
            line=dict(color='#e74c3c', width=3),
            marker=dict(size=10, color='#e74c3c', symbol='diamond'),
            name='Pareto Front'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Maintenance Cost ($)',
            yaxis_title='Risk Cost ($)',
            height=500
        )
        return fig
    
    def plot_trade_off(self, title: str = "Cost-Risk-Availability Trade-off") -> go.Figure:
        """3D scatter of trade-off space."""
        if self.pareto_front is None:
            self.find_pareto_front()
        
        front = self.pareto_front
        
        fig = go.Figure(data=go.Scatter3d(
            x=front['total_cost'],
            y=front['risk_cost'],
            z=front['availability'],
            mode='markers',
            marker=dict(
                size=8,
                color=front['combined_cost'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Combined Cost')
            ),
            text=[f"Sol {r['solution_id']}: {r['n_maintained']} maintained"
                  for _, r in front.iterrows()],
        ))
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='Maintenance Cost',
                yaxis_title='Risk Cost',
                zaxis_title='Availability'
            ),
            height=550
        )
        return fig
    
    def plot_budget_sensitivity(self, budget_df: pd.DataFrame,
                                  title: str = "Budget Sensitivity Analysis") -> go.Figure:
        """Effect of budget on optimal solutions."""
        fig = make_subplots(rows=1, cols=2,
                           subplot_titles=('Cost Components', 'Fleet Metrics'))
        
        fig.add_trace(go.Scatter(
            x=budget_df['budget'], y=budget_df['optimal_cost'],
            mode='lines+markers', name='Maintenance Cost',
            line=dict(color='#3498db', width=2)
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=budget_df['budget'], y=budget_df['risk_cost'],
            mode='lines+markers', name='Risk Cost',
            line=dict(color='#e74c3c', width=2)
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=budget_df['budget'], y=budget_df['combined_cost'],
            mode='lines+markers', name='Combined Cost',
            line=dict(color='#9b59b6', width=2, dash='dash')
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=budget_df['budget'], y=budget_df['availability'] * 100,
            mode='lines+markers', name='Availability %',
            line=dict(color='#27ae60', width=2)
        ), row=1, col=2)
        fig.add_trace(go.Bar(
            x=budget_df['budget'], y=budget_df['n_maintained'],
            name='Engines Maintained',
            marker_color='#f39c12', opacity=0.5
        ), row=1, col=2)
        
        fig.update_xaxes(title_text='Budget ($)', row=1, col=1)
        fig.update_xaxes(title_text='Budget ($)', row=1, col=2)
        fig.update_yaxes(title_text='Cost ($)', row=1, col=1)
        fig.update_yaxes(title_text='Availability % / Count', row=1, col=2)
        fig.update_layout(title=title, height=400)
        
        return fig
    
    def plot_recommendation(self, recommendation: Dict,
                              fleet_df: pd.DataFrame,
                              title: str = "Recommended Schedule") -> go.Figure:
        """Visualize which engines are scheduled for maintenance."""
        rul_col = 'rul_pred' if 'rul_pred' in fleet_df.columns else 'RUL'
        ruls = fleet_df[rul_col].values
        maintained = recommendation['maintained_engines']
        
        engine_ids = fleet_df.get('engine_id', [f'E{i+1}' for i in range(len(fleet_df))])
        
        colors = ['#e74c3c' if m else '#3498db' for m in maintained]
        labels = ['Maintain' if m else 'Monitor' for m in maintained]
        
        fig = go.Figure(data=go.Bar(
            x=list(engine_ids), y=ruls,
            marker_color=colors,
            text=labels,
            textposition='outside'
        ))
        
        fig.add_hline(y=self.safety_rul, line_dash='dash', line_color='red',
                      annotation_text=f'Safety threshold ({self.safety_rul}cy)')
        
        fig.update_layout(
            title=f"{title} ({recommendation['preference']}): "
                  f"${recommendation['total_cost']:,.0f} cost, "
                  f"{recommendation['availability']:.0%} availability",
            xaxis_title='Engine',
            yaxis_title='RUL (cycles)',
            height=400
        )
        return fig


# ============================================================
# Standalone test
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Maintenance Cost Optimizer")
    print("=" * 60)
    
    np.random.seed(42)
    
    fleet = pd.DataFrame({
        'engine_id': [f'ENG-{i:03d}' for i in range(1, 31)],
        'rul_pred': np.concatenate([
            np.random.randint(5, 25, 6),
            np.random.randint(25, 70, 10),
            np.random.randint(70, 200, 14),
        ]).astype(float)
    })
    
    optimizer = CostOptimizer(budget_cap=300000, hangar_capacity=5, safety_rul=20)
    
    print("\n--- Generating Solutions ---")
    solutions = optimizer.generate_solutions(fleet, n_solutions=300)
    print(f"  Generated: {len(solutions)} solutions")
    print(f"  Feasible: {solutions['within_budget'].sum()}")
    
    print("\n--- Pareto Front ---")
    pareto = optimizer.find_pareto_front()
    print(f"  Pareto-optimal: {len(pareto)} solutions")
    
    print("\n--- Recommendations ---")
    for pref in ['cost', 'safety', 'balanced', 'availability']:
        rec = optimizer.recommend_solution(pref)
        print(f"  {pref:>12s}: cost=${rec['total_cost']:>8,.0f}, "
              f"risk=${rec['risk_cost']:>8,.0f}, "
              f"avail={rec['availability']:.0%}, "
              f"maintained={rec['n_maintained']}")
    
    print("\n--- Budget Sensitivity ---")
    budget_df = optimizer.budget_sensitivity(fleet, [50000, 100000, 250000, 500000])
    print(budget_df.to_string(index=False))
    
    print("\n✅ Maintenance Cost Optimizer test PASSED")

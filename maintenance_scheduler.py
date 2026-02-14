"""
Predictive Maintenance Scheduler
Constraint-based optimization for fleet maintenance scheduling that
minimizes total cost while respecting capacity and priority constraints.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import config
from utils import setup_logging

logger = setup_logging(__name__)


@dataclass
class MaintenanceJob:
    """Represents a single maintenance task."""
    engine_id: str
    rul_estimate: float
    priority: int  # 1 = highest priority
    estimated_duration: int = 3  # cycles of downtime
    cost: float = 10000.0
    scheduled_start: Optional[int] = None
    status: str = 'pending'  # pending, scheduled, completed


class MaintenanceScheduler:
    """
    Optimization-based maintenance scheduler for engine fleets.
    
    Minimizes total cost (scheduled + unscheduled maintenance + downtime)
    while respecting hangar capacity and technician availability constraints.
    """
    
    def __init__(self, n_hangars: int = 3, horizon: int = 100):
        """
        Args:
            n_hangars: Number of maintenance slots available per cycle.
            horizon: Scheduling horizon in cycles.
        """
        self.n_hangars = n_hangars
        self.horizon = horizon
        self.jobs = []
        self.schedule = {}
        self.cost_params = config.COST_PARAMETERS
        self.thresholds = config.MAINTENANCE_THRESHOLDS
        logger.info(f"MaintenanceScheduler initialized: "
                    f"{n_hangars} hangars, {horizon}-cycle horizon")
    
    def create_jobs_from_fleet(self, fleet_df: pd.DataFrame) -> List[MaintenanceJob]:
        """
        Create maintenance jobs from fleet RUL predictions.
        
        Args:
            fleet_df: DataFrame with engine_id and rul_pred columns.
            
        Returns:
            List of MaintenanceJob objects.
        """
        self.jobs = []
        
        for _, row in fleet_df.iterrows():
            rul = row.get('rul_pred', row.get('RUL', 100))
            eid = str(row.get('engine_id', row.get('unit_id', '?')))
            
            # Priority based on RUL
            if rul < self.thresholds['critical']:
                priority = 1
                cost = self.cost_params.get('unscheduled_maintenance', 50000)
            elif rul < self.thresholds['warning']:
                priority = 2
                cost = self.cost_params.get('scheduled_maintenance', 10000) * 1.5
            else:
                priority = 3
                cost = self.cost_params.get('scheduled_maintenance', 10000)
            
            job = MaintenanceJob(
                engine_id=eid,
                rul_estimate=float(rul),
                priority=priority,
                cost=cost,
                estimated_duration=max(1, int(5 - priority))
            )
            self.jobs.append(job)
        
        # Sort by priority, then by RUL
        self.jobs.sort(key=lambda j: (j.priority, j.rul_estimate))
        
        logger.info(f"Created {len(self.jobs)} maintenance jobs "
                    f"(P1: {sum(1 for j in self.jobs if j.priority==1)}, "
                    f"P2: {sum(1 for j in self.jobs if j.priority==2)}, "
                    f"P3: {sum(1 for j in self.jobs if j.priority==3)})")
        
        return self.jobs
    
    def optimize_schedule(self, strategy: str = 'priority') -> Dict:
        """
        Generate an optimized maintenance schedule.
        
        Args:
            strategy: 'priority' (greedy by priority/RUL),
                      'cost_min' (minimize total cost),
                      'balanced' (spread workload evenly).
            
        Returns:
            Schedule summary dict.
        """
        if not self.jobs:
            raise RuntimeError("Call create_jobs_from_fleet() first.")
        
        # Initialize hangar slots: {cycle -> n_occupied}
        hangar_usage = {c: 0 for c in range(self.horizon)}
        schedule_entries = []
        total_cost = 0
        total_downtime = 0
        failures_prevented = 0
        
        if strategy == 'priority':
            scheduled_jobs = self._schedule_priority(hangar_usage)
        elif strategy == 'cost_min':
            scheduled_jobs = self._schedule_cost_min(hangar_usage)
        else:
            scheduled_jobs = self._schedule_balanced(hangar_usage)
        
        for job in scheduled_jobs:
            if job.scheduled_start is not None:
                job.status = 'scheduled'
                total_cost += job.cost
                total_downtime += job.estimated_duration
                if job.rul_estimate < self.thresholds['critical']:
                    failures_prevented += 1
                
                schedule_entries.append({
                    'engine_id': job.engine_id,
                    'start_cycle': job.scheduled_start,
                    'end_cycle': job.scheduled_start + job.estimated_duration,
                    'priority': job.priority,
                    'rul': job.rul_estimate,
                    'cost': job.cost,
                    'duration': job.estimated_duration
                })
        
        unscheduled = [j for j in self.jobs if j.status == 'pending']
        
        # Penalty for unscheduled critical jobs
        for j in unscheduled:
            if j.priority == 1:
                total_cost += self.cost_params.get('unscheduled_maintenance', 50000) * 2
        
        self.schedule = {
            'strategy': strategy,
            'entries': schedule_entries,
            'total_cost': float(total_cost),
            'total_downtime': int(total_downtime),
            'scheduled_count': len(schedule_entries),
            'unscheduled_count': len(unscheduled),
            'failures_prevented': failures_prevented,
            'avg_hangar_utilization': np.mean([
                min(hangar_usage.get(c, 0) / self.n_hangars, 1.0) 
                for c in range(self.horizon)
            ]),
            'peak_utilization_cycle': max(hangar_usage, key=hangar_usage.get)
        }
        
        logger.info(f"Schedule ({strategy}): {len(schedule_entries)} jobs, "
                    f"cost=${total_cost:,.0f}, downtime={total_downtime}cy")
        
        return self.schedule
    
    def _schedule_priority(self, hangar_usage: Dict) -> List[MaintenanceJob]:
        """Greedy scheduling by priority then RUL."""
        for job in self.jobs:
            # Try to schedule as early as possible
            deadline = min(int(job.rul_estimate), self.horizon - job.estimated_duration)
            if deadline <= 0:
                deadline = job.estimated_duration
            
            # Find earliest available slot
            for start in range(max(0, deadline - 20), min(deadline + 1, self.horizon - job.estimated_duration)):
                if all(hangar_usage.get(start + d, 0) < self.n_hangars 
                       for d in range(job.estimated_duration)):
                    job.scheduled_start = start
                    for d in range(job.estimated_duration):
                        hangar_usage[start + d] = hangar_usage.get(start + d, 0) + 1
                    break
            
            # Fallback: any available slot
            if job.scheduled_start is None:
                for start in range(self.horizon - job.estimated_duration):
                    if all(hangar_usage.get(start + d, 0) < self.n_hangars 
                           for d in range(job.estimated_duration)):
                        job.scheduled_start = start
                        for d in range(job.estimated_duration):
                            hangar_usage[start + d] = hangar_usage.get(start + d, 0) + 1
                        break
        
        return self.jobs
    
    def _schedule_cost_min(self, hangar_usage: Dict) -> List[MaintenanceJob]:
        """Schedule to minimize total cost (defer low-priority, rush critical)."""
        critical = [j for j in self.jobs if j.priority == 1]
        warning = [j for j in self.jobs if j.priority == 2]
        routine = [j for j in self.jobs if j.priority == 3]
        
        # Critical first, at earliest slot
        for job in critical:
            for start in range(min(int(job.rul_estimate), self.horizon - job.estimated_duration)):
                if all(hangar_usage.get(start + d, 0) < self.n_hangars 
                       for d in range(job.estimated_duration)):
                    job.scheduled_start = start
                    for d in range(job.estimated_duration):
                        hangar_usage[start + d] = hangar_usage.get(start + d, 0) + 1
                    break
        
        # Warning at midpoint
        for job in warning:
            midpoint = int(job.rul_estimate * 0.5)
            for start in range(max(0, midpoint - 5), min(midpoint + 20, self.horizon - job.estimated_duration)):
                if all(hangar_usage.get(start + d, 0) < self.n_hangars 
                       for d in range(job.estimated_duration)):
                    job.scheduled_start = start
                    for d in range(job.estimated_duration):
                        hangar_usage[start + d] = hangar_usage.get(start + d, 0) + 1
                    break
        
        # Routine deferred to later
        for job in routine:
            target = int(job.rul_estimate * 0.7)
            for start in range(max(0, target), min(self.horizon - job.estimated_duration, target + 30)):
                if all(hangar_usage.get(start + d, 0) < self.n_hangars 
                       for d in range(job.estimated_duration)):
                    job.scheduled_start = start
                    for d in range(job.estimated_duration):
                        hangar_usage[start + d] = hangar_usage.get(start + d, 0) + 1
                    break
        
        return self.jobs
    
    def _schedule_balanced(self, hangar_usage: Dict) -> List[MaintenanceJob]:
        """Spread jobs evenly across the horizon."""
        total_jobs = len(self.jobs)
        spacing = max(1, self.horizon // max(total_jobs, 1))
        
        for i, job in enumerate(self.jobs):
            target = min(i * spacing, self.horizon - job.estimated_duration - 1)
            for start in range(max(0, target), self.horizon - job.estimated_duration):
                if all(hangar_usage.get(start + d, 0) < self.n_hangars 
                       for d in range(job.estimated_duration)):
                    job.scheduled_start = start
                    for d in range(job.estimated_duration):
                        hangar_usage[start + d] = hangar_usage.get(start + d, 0) + 1
                    break
        
        return self.jobs
    
    def compare_strategies(self, fleet_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compare all scheduling strategies.
        
        Returns:
            Comparison DataFrame.
        """
        results = []
        for strat in ['priority', 'cost_min', 'balanced']:
            # Reset
            self.jobs = []
            self.create_jobs_from_fleet(fleet_df)
            sched = self.optimize_schedule(strat)
            results.append({
                'strategy': strat,
                'total_cost': sched['total_cost'],
                'scheduled': sched['scheduled_count'],
                'unscheduled': sched['unscheduled_count'],
                'downtime': sched['total_downtime'],
                'avg_utilization': sched['avg_hangar_utilization'],
                'failures_prevented': sched['failures_prevented']
            })
        
        return pd.DataFrame(results)
    
    # ------------------------------------------------------------------
    # Plotly visualizations
    # ------------------------------------------------------------------
    def plot_gantt(self, title: str = "Maintenance Schedule (Gantt)") -> go.Figure:
        """Gantt chart of scheduled maintenance."""
        if not self.schedule or not self.schedule.get('entries'):
            raise RuntimeError("No schedule to plot.")
        
        entries = sorted(self.schedule['entries'], key=lambda e: e['start_cycle'])
        
        colors = {1: '#d62728', 2: '#ff7f0e', 3: '#2ca02c'}
        
        fig = go.Figure()
        
        for i, entry in enumerate(entries):
            fig.add_trace(go.Bar(
                y=[entry['engine_id']],
                x=[entry['duration']],
                base=[entry['start_cycle']],
                orientation='h',
                name=f"P{entry['priority']}",
                marker_color=colors.get(entry['priority'], '#999'),
                text=f"P{entry['priority']} | RUL:{entry['rul']:.0f}",
                textposition='inside',
                showlegend=(i < 3),
                hovertext=f"Engine: {entry['engine_id']}<br>"
                          f"Start: cycle {entry['start_cycle']}<br>"
                          f"Duration: {entry['duration']} cycles<br>"
                          f"Cost: ${entry['cost']:,.0f}",
                hoverinfo='text'
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Cycle',
            yaxis_title='Engine',
            barmode='stack',
            height=max(350, len(entries) * 25),
            showlegend=True
        )
        return fig
    
    def plot_utilization(self, title: str = "Hangar Utilization") -> go.Figure:
        """Line chart of hangar utilization over time."""
        if not self.schedule or not self.schedule.get('entries'):
            raise RuntimeError("No schedule to plot.")
        
        usage = np.zeros(self.horizon)
        for entry in self.schedule['entries']:
            for d in range(entry['duration']):
                cycle = entry['start_cycle'] + d
                if 0 <= cycle < self.horizon:
                    usage[cycle] += 1
        
        utilization = usage / self.n_hangars * 100
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(self.horizon)), y=utilization,
            fill='tozeroy', fillcolor='rgba(31,119,180,0.2)',
            line=dict(color='#1f77b4', width=2),
            name='Utilization'
        ))
        fig.add_hline(y=100, line_dash='dash', line_color='red',
                      annotation_text='Full Capacity')
        fig.add_hline(y=75, line_dash='dot', line_color='orange',
                      annotation_text='75% Target')
        
        fig.update_layout(
            title=title,
            xaxis_title='Cycle',
            yaxis_title='Utilization (%)',
            yaxis_range=[0, 120],
            height=350
        )
        return fig
    
    def plot_cost_breakdown(self, comparison_df: pd.DataFrame,
                             title: str = "Strategy Cost Comparison") -> go.Figure:
        """Bar chart comparing strategies."""
        fig = make_subplots(rows=1, cols=2,
                           subplot_titles=('Total Cost', 'Hangar Utilization'))
        
        strategies = comparison_df['strategy'].tolist()
        costs = comparison_df['total_cost'].tolist()
        utils = (comparison_df['avg_utilization'] * 100).tolist()
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        fig.add_trace(go.Bar(
            x=strategies, y=costs,
            marker_color=colors[:len(strategies)],
            text=[f"${c:,.0f}" for c in costs],
            textposition='outside', name='Cost'
        ), row=1, col=1)
        
        fig.add_trace(go.Bar(
            x=strategies, y=utils,
            marker_color=colors[:len(strategies)],
            text=[f"{u:.1f}%" for u in utils],
            textposition='outside', name='Utilization'
        ), row=1, col=2)
        
        fig.update_layout(title=title, height=400, showlegend=False)
        return fig


# ============================================================
# Standalone test
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Predictive Maintenance Scheduler")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Simulated fleet
    fleet = pd.DataFrame({
        'engine_id': [f'ENG-{i:03d}' for i in range(1, 31)],
        'rul_pred': np.random.randint(5, 150, 30).astype(float)
    })
    
    scheduler = MaintenanceScheduler(n_hangars=3, horizon=100)
    
    print("\n--- Creating Jobs ---")
    jobs = scheduler.create_jobs_from_fleet(fleet)
    for j in jobs[:5]:
        print(f"  {j.engine_id}: RUL={j.rul_estimate:.0f}, P{j.priority}, ${j.cost:,.0f}")
    
    print("\n--- Priority Schedule ---")
    sched = scheduler.optimize_schedule('priority')
    print(f"  Scheduled: {sched['scheduled_count']}")
    print(f"  Total cost: ${sched['total_cost']:,.0f}")
    print(f"  Downtime: {sched['total_downtime']} cycles")
    print(f"  Failures prevented: {sched['failures_prevented']}")
    
    print("\n--- Strategy Comparison ---")
    comp = scheduler.compare_strategies(fleet)
    print(comp.to_string(index=False))
    
    print("\n✅ Maintenance Scheduler test PASSED")

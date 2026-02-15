"""
Digital Twin Engine Simulator
Physics-inspired virtual engine model that mirrors a real engine's degradation,
generates synthetic sensor data, and projects remaining life with confidence bands.
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


# ============================================================
# Physical constants for C-MAPSS engine model
# ============================================================
ENGINE_PARAMS = {
    'T2_base': 518.67,    # Total temperature at fan inlet (°R)
    'T24_base': 642.15,   # Total temperature at LPC outlet (°R)
    'T30_base': 1589.70,  # Total temperature at HPC outlet (°R)
    'T50_base': 1400.60,  # Total temperature at LPT outlet (°R)
    'P2_base': 14.62,     # Pressure at fan inlet (psia)
    'P15_base': 21.61,    # Total pressure in bypass-duct (psia)
    'P30_base': 553.75,   # Total pressure at HPC outlet (psia)
    'Nf_base': 2388.02,   # Physical fan speed (rpm)
    'Nc_base': 9046.19,   # Physical core speed (rpm)
    'epr_base': 1.30,     # Engine pressure ratio (P50/P2)
    'Ps30_base': 47.47,   # Static pressure at HPC outlet (psia)
    'phi_base': 521.66,   # Ratio of fuel flow to Ps30
    'NRf_base': 2388.02,  # Corrected fan speed (rpm)
    'NRc_base': 8138.62,  # Corrected core speed (rpm)
    'BPR_base': 8.4195,   # Bypass ratio
    'farB_base': 0.03,    # Burner fuel-air ratio
    'htBleed_base': 392.0, # Bleed enthalpy
    'Nf_dmd_base': 2388.0, # Demanded fan speed (rpm)
    'W31_base': 38.86,    # HPT coolant bleed (lbm/s)
    'W32_base': 23.06,    # LPT coolant bleed (lbm/s)
}


@dataclass
class DegradationProfile:
    """Defines how a degradation mode affects sensors over time."""
    name: str
    affected_sensors: Dict[str, float]  # sensor_name -> rate per cycle
    onset_cycle: int = 50               # when degradation begins
    acceleration: float = 1.02          # exponential acceleration factor
    noise_level: float = 0.01           # sensor noise std (fraction of base)


# Pre-defined C-MAPSS degradation profiles
HPC_DEGRADATION = DegradationProfile(
    name='HPC Degradation',
    affected_sensors={
        'sensor_7': -0.005,   # HPC outlet temperature decreases
        'sensor_8': -0.003,   # Physical fan speed drops
        'sensor_9': +0.004,   # Physical core speed compensates
        'sensor_11': +0.008,  # Static pressure at HPC outlet rises
        'sensor_12': +0.006,  # Fuel flow ratio increases
        'sensor_15': +0.01,   # Bleed enthalpy rises
        'sensor_20': -0.004,  # BPR drops
        'sensor_21': +0.003,  # Burner fuel-air ratio rises
    },
    onset_cycle=30,
    acceleration=1.015,
    noise_level=0.005
)

FAN_DEGRADATION = DegradationProfile(
    name='Fan Degradation',
    affected_sensors={
        'sensor_2': -0.003,
        'sensor_3': +0.004,
        'sensor_4': -0.003,
        'sensor_7': +0.003,
        'sensor_14': -0.005,
        'sensor_20': +0.004,
    },
    onset_cycle=40,
    acceleration=1.012,
    noise_level=0.006
)


class DigitalTwin:
    """
    Physics-inspired digital twin of a turbofan engine.
    
    Simulates realistic sensor degradation trajectories, generates synthetic
    training data, and projects remaining life with uncertainty bands.
    """
    
    SENSOR_NAMES = [f'sensor_{i}' for i in range(1, 22)]
    SETTING_NAMES = ['setting_1', 'setting_2', 'setting_3']
    
    def __init__(self, engine_id: str = 'TWIN-001',
                 total_life: int = 200,
                 degradation: DegradationProfile = None):
        """
        Args:
            engine_id: Identifier for this twin.
            total_life: Total engine life in cycles before failure.
            degradation: Degradation profile to apply.
        """
        self.engine_id = engine_id
        self.total_life = total_life
        self.degradation = degradation or HPC_DEGRADATION
        self.base_values = self._init_base_values()
        self.history = None
        self.health_index = None
        logger.info(f"DigitalTwin '{engine_id}' initialized: "
                    f"{total_life} cycle life, {self.degradation.name}")
    
    def _init_base_values(self) -> Dict[str, float]:
        """Initialize baseline sensor values from engine parameters."""
        base = {}
        param_keys = list(ENGINE_PARAMS.keys())
        for i, sensor in enumerate(self.SENSOR_NAMES):
            if i < len(param_keys):
                base[sensor] = ENGINE_PARAMS[param_keys[i]]
            else:
                base[sensor] = np.random.uniform(10, 500)
        return base
    
    def simulate(self, n_cycles: int = None,
                 operating_conditions: int = 1) -> pd.DataFrame:
        """
        Run a full engine life simulation.
        
        Args:
            n_cycles: Number of cycles to simulate (default: total_life).
            operating_conditions: Number of operating condition regimes.
            
        Returns:
            DataFrame of simulated sensor readings over time.
        """
        n_cycles = n_cycles or self.total_life
        
        records = []
        health_values = []
        
        for cycle in range(1, n_cycles + 1):
            record = {
                'unit_id': self.engine_id,
                'time_cycles': cycle,
            }
            
            # Operating settings (slight variation per regime)
            for j, setting in enumerate(self.SETTING_NAMES):
                regime = np.random.randint(0, operating_conditions)
                record[setting] = np.random.normal(
                    [0.0, 0.0, 100.0][j] + regime * 0.1, 0.002
                )
            
            # Sensor values with degradation
            health = 1.0
            for sensor in self.SENSOR_NAMES:
                base_val = self.base_values[sensor]
                
                # Degradation effect
                deg_rate = self.degradation.affected_sensors.get(sensor, 0)
                if cycle > self.degradation.onset_cycle and deg_rate != 0:
                    elapsed = cycle - self.degradation.onset_cycle
                    # Exponential degradation
                    effect = deg_rate * elapsed * (
                        self.degradation.acceleration ** (elapsed / 50)
                    )
                    degraded_val = base_val * (1 + effect)
                    
                    # Track health via affected sensors
                    health -= abs(effect) * 0.5
                else:
                    degraded_val = base_val
                
                # Add realistic noise
                noise = np.random.normal(0, abs(base_val) * self.degradation.noise_level)
                record[sensor] = degraded_val + noise
            
            health = max(0, min(1, health))
            health_values.append(health)
            records.append(record)
        
        self.history = pd.DataFrame(records)
        self.health_index = np.array(health_values)
        
        # Add RUL
        self.history['RUL'] = n_cycles - self.history['time_cycles']
        
        logger.info(f"Simulated {n_cycles} cycles: "
                    f"final health = {health_values[-1]:.2f}")
        
        return self.history
    
    def project_remaining_life(self, current_cycle: int,
                                n_simulations: int = 500) -> Dict:
        """
        Monte Carlo projection of remaining useful life.
        
        Args:
            current_cycle: Current engine cycle.
            n_simulations: Number of Monte Carlo simulations.
            
        Returns:
            Dict with RUL projections and confidence bands.
        """
        remaining_max = self.total_life - current_cycle
        
        rul_samples = []
        for _ in range(n_simulations):
            # Random life with noise
            noise = np.random.normal(0, remaining_max * 0.15)
            rul = remaining_max + noise
            rul = max(0, rul)
            rul_samples.append(rul)
        
        rul_arr = np.array(rul_samples)
        
        projection = {
            'current_cycle': current_cycle,
            'mean_rul': float(np.mean(rul_arr)),
            'median_rul': float(np.median(rul_arr)),
            'std_rul': float(np.std(rul_arr)),
            'ci_lower': float(np.percentile(rul_arr, 5)),
            'ci_upper': float(np.percentile(rul_arr, 95)),
            'p_failure_30': float(np.mean(rul_arr < 30)),
            'p_failure_60': float(np.mean(rul_arr < 60)),
            'n_simulations': n_simulations,
            'rul_distribution': rul_arr.tolist()
        }
        
        logger.info(f"RUL projection @ cycle {current_cycle}: "
                    f"{projection['mean_rul']:.0f} ± {projection['std_rul']:.0f} "
                    f"(90% CI: {projection['ci_lower']:.0f}–{projection['ci_upper']:.0f})")
        
        return projection
    
    def generate_fleet(self, n_engines: int = 20,
                        life_range: Tuple[int, int] = (150, 300)) -> pd.DataFrame:
        """
        Generate a synthetic fleet of digital twin engines.
        
        Args:
            n_engines: Number of engines to simulate.
            life_range: (min_life, max_life) in cycles.
            
        Returns:
            Combined DataFrame of all engine simulations.
        """
        all_data = []
        profiles = [HPC_DEGRADATION, FAN_DEGRADATION]
        
        for i in range(n_engines):
            life = np.random.randint(life_range[0], life_range[1])
            profile = np.random.choice(profiles)
            
            twin = DigitalTwin(
                engine_id=str(i + 1),
                total_life=life,
                degradation=profile
            )
            engine_df = twin.simulate()
            all_data.append(engine_df)
        
        fleet_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Generated synthetic fleet: {n_engines} engines, "
                    f"{len(fleet_df)} total records")
        
        return fleet_df
    
    def compare_with_real(self, real_engine_df: pd.DataFrame) -> Dict:
        """
        Compare digital twin simulation with real engine data.
        
        Args:
            real_engine_df: Real engine sensor data.
            
        Returns:
            Comparison metrics dict.
        """
        if self.history is None:
            self.simulate(n_cycles=len(real_engine_df))
        
        sensor_cols = [c for c in self.SENSOR_NAMES if c in real_engine_df.columns]
        min_len = min(len(self.history), len(real_engine_df))
        
        comparison = {}
        for sensor in sensor_cols:
            real = real_engine_df[sensor].values[:min_len]
            sim = self.history[sensor].values[:min_len]
            
            # Normalize for comparison
            r_norm = (real - real.mean()) / max(real.std(), 1e-8)
            s_norm = (sim - sim.mean()) / max(sim.std(), 1e-8)
            
            corr = np.corrcoef(r_norm, s_norm)[0, 1]
            rmse = np.sqrt(np.mean((r_norm - s_norm) ** 2))
            
            comparison[sensor] = {
                'correlation': float(corr) if not np.isnan(corr) else 0.0,
                'rmse': float(rmse),
                'real_mean': float(real.mean()),
                'sim_mean': float(sim.mean()),
            }
        
        avg_corr = np.mean([v['correlation'] for v in comparison.values()])
        logger.info(f"Twin vs Real comparison: avg correlation = {avg_corr:.3f}")
        
        return {
            'per_sensor': comparison,
            'avg_correlation': float(avg_corr),
            'n_sensors_compared': len(sensor_cols),
            'n_cycles_compared': min_len
        }
    
    # ------------------------------------------------------------------
    # Plotly visualizations
    # ------------------------------------------------------------------
    def plot_sensor_trajectories(self, sensors: List[str] = None,
                                  title: str = "Digital Twin — Sensor Trajectories") -> go.Figure:
        """Plot sensor degradation trajectories over engine life."""
        if self.history is None:
            raise RuntimeError("Call simulate() first.")
        
        sensors = sensors or ['sensor_7', 'sensor_9', 'sensor_11', 'sensor_12', 'sensor_15']
        n = len(sensors)
        
        fig = make_subplots(rows=n, cols=1, shared_xaxes=True,
                           subplot_titles=[s for s in sensors],
                           vertical_spacing=0.03)
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2']
        
        for i, sensor in enumerate(sensors):
            if sensor in self.history.columns:
                fig.add_trace(go.Scatter(
                    x=self.history['time_cycles'],
                    y=self.history[sensor],
                    mode='lines',
                    name=sensor,
                    line=dict(color=colors[i % len(colors)], width=1.5),
                    showlegend=True
                ), row=i + 1, col=1)
        
        # Mark degradation onset
        fig.add_vline(x=self.degradation.onset_cycle, line_dash='dot',
                      line_color='red', annotation_text='Degradation onset')
        
        fig.update_layout(title=title, height=150 * n + 100)
        fig.update_xaxes(title_text='Cycle', row=n, col=1)
        return fig
    
    def plot_health_index(self, title: str = "Digital Twin — Health Index") -> go.Figure:
        """Plot engine health index over time."""
        if self.health_index is None:
            raise RuntimeError("Call simulate() first.")
        
        cycles = list(range(1, len(self.health_index) + 1))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=cycles, y=self.health_index,
            fill='tozeroy',
            fillcolor='rgba(39,174,96,0.2)',
            line=dict(color='#27ae60', width=2),
            name='Health Index'
        ))
        
        fig.add_hline(y=0.3, line_dash='dash', line_color='red',
                      annotation_text='Critical')
        fig.add_hline(y=0.6, line_dash='dash', line_color='orange',
                      annotation_text='Warning')
        
        fig.update_layout(
            title=title,
            xaxis_title='Cycle',
            yaxis_title='Health Index',
            yaxis_range=[0, 1.05],
            height=350
        )
        return fig
    
    def plot_rul_projection(self, projection: Dict,
                             title: str = "RUL Projection Distribution") -> go.Figure:
        """Histogram of Monte Carlo RUL projections."""
        rul = projection['rul_distribution']
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=rul, nbinsx=50,
            marker_color='#3498db',
            opacity=0.7,
            name='RUL Samples'
        ))
        
        fig.add_vline(x=projection['mean_rul'], line_dash='solid',
                      line_color='red', annotation_text=f"Mean: {projection['mean_rul']:.0f}")
        fig.add_vline(x=projection['ci_lower'], line_dash='dash',
                      line_color='orange', annotation_text=f"5%: {projection['ci_lower']:.0f}")
        fig.add_vline(x=projection['ci_upper'], line_dash='dash',
                      line_color='orange', annotation_text=f"95%: {projection['ci_upper']:.0f}")
        
        fig.update_layout(
            title=title,
            xaxis_title='Projected RUL (cycles)',
            yaxis_title='Count',
            height=350
        )
        return fig
    
    def plot_twin_vs_real(self, comparison: Dict, sensor: str,
                           real_df: pd.DataFrame,
                           title: str = None) -> go.Figure:
        """Overlay digital twin and real engine sensor traces."""
        n_cycles = comparison['n_cycles_compared']
        title = title or f"Digital Twin vs Real — {sensor}"
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(n_cycles)),
            y=real_df[sensor].values[:n_cycles],
            mode='lines', name='Real Engine',
            line=dict(color='#2c3e50', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=list(range(n_cycles)),
            y=self.history[sensor].values[:n_cycles],
            mode='lines', name='Digital Twin',
            line=dict(color='#e74c3c', width=2, dash='dash')
        ))
        
        corr = comparison['per_sensor'].get(sensor, {}).get('correlation', 0)
        fig.update_layout(
            title=f"{title} (r = {corr:.3f})",
            xaxis_title='Cycle',
            yaxis_title='Sensor Value',
            height=350
        )
        return fig


# ============================================================
# Standalone test
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Digital Twin Engine Simulator")
    print("=" * 60)
    
    # Single engine
    print("\n--- Single Engine Simulation ---")
    twin = DigitalTwin(engine_id='TWIN-001', total_life=200,
                        degradation=HPC_DEGRADATION)
    history = twin.simulate()
    print(f"  Simulated: {len(history)} cycles, {len(twin.SENSOR_NAMES)} sensors")
    print(f"  Final health: {twin.health_index[-1]:.3f}")
    
    # RUL projection
    print("\n--- RUL Projection ---")
    proj = twin.project_remaining_life(current_cycle=120, n_simulations=1000)
    print(f"  Mean RUL: {proj['mean_rul']:.0f} ± {proj['std_rul']:.0f}")
    print(f"  90% CI: [{proj['ci_lower']:.0f}, {proj['ci_upper']:.0f}]")
    print(f"  P(failure < 30cy): {proj['p_failure_30']:.1%}")
    
    # Fleet generation
    print("\n--- Synthetic Fleet ---")
    fleet_df = twin.generate_fleet(n_engines=10)
    print(f"  Generated: {fleet_df['unit_id'].nunique()} engines, "
          f"{len(fleet_df)} records")
    
    # Compare with real data
    print("\n--- Twin vs Real Comparison ---")
    from data_loader import CMAPSSDataLoader
    loader = CMAPSSDataLoader('FD001')
    train_df, _, _ = loader.load_all_data()
    
    real_eng = train_df[train_df['unit_id'] == 1]
    twin2 = DigitalTwin(total_life=len(real_eng))
    twin2.simulate(n_cycles=len(real_eng))
    comp = twin2.compare_with_real(real_eng)
    print(f"  Avg correlation: {comp['avg_correlation']:.3f}")
    print(f"  Sensors compared: {comp['n_sensors_compared']}")
    
    print("\n✅ Digital Twin test PASSED")

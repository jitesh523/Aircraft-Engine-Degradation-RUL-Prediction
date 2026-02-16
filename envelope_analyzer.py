"""
Operational Envelope Analyzer
Detects safe operating boundaries from historical data, scores engines
for envelope violations, and identifies out-of-spec sensor regimes.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats as sp_stats
import config
from utils import setup_logging

logger = setup_logging(__name__)


@dataclass
class OperatingEnvelope:
    """Defines safe operating boundaries for a set of sensors."""
    sensor_name: str
    lower_bound: float
    upper_bound: float
    mean: float
    std: float
    percentile_5: float
    percentile_95: float


class EnvelopeAnalyzer:
    """
    Learns safe operating envelopes from healthy engine data and scores
    new readings for envelope violations.
    
    Features:
    - Statistical boundary learning (percentile + IQR)
    - Multi-sensor violation scoring
    - Temporal violation tracking (degradation onset detection)
    - Radar chart and boundary visualization
    """
    
    SENSOR_COLS = [f'sensor_{i}' for i in range(1, 22)]
    
    def __init__(self, method: str = 'percentile', margin: float = 0.05):
        """
        Args:
            method: 'percentile' (5th/95th) or 'iqr' (1.5*IQR).
            margin: Extra margin beyond boundaries (fraction).
        """
        self.method = method
        self.margin = margin
        self.envelopes: Dict[str, OperatingEnvelope] = {}
        self.healthy_stats = None
        logger.info(f"EnvelopeAnalyzer initialized: method={method}, margin={margin}")
    
    def learn_envelope(self, train_df: pd.DataFrame,
                        rul_threshold: int = 120) -> Dict[str, OperatingEnvelope]:
        """
        Learn safe operating envelope from healthy engine cycles.
        
        Args:
            train_df: Training data with sensors and RUL.
            rul_threshold: Only use cycles with RUL > threshold (healthy).
            
        Returns:
            Dict of sensor envelopes.
        """
        rul_col = 'RUL' if 'RUL' in train_df.columns else 'rul_pred'
        healthy = train_df[train_df[rul_col] > rul_threshold]
        
        if len(healthy) == 0:
            healthy = train_df[train_df[rul_col] > train_df[rul_col].median()]
        
        sensors = [c for c in self.SENSOR_COLS if c in train_df.columns]
        stats_rows = []
        
        for sensor in sensors:
            vals = healthy[sensor].dropna().values
            if len(vals) < 10:
                continue
            
            mean = float(np.mean(vals))
            std = float(np.std(vals))
            p5 = float(np.percentile(vals, 5))
            p95 = float(np.percentile(vals, 95))
            
            if self.method == 'iqr':
                q1, q3 = np.percentile(vals, [25, 75])
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
            else:
                rng = p95 - p5
                lower = p5 - self.margin * rng
                upper = p95 + self.margin * rng
            
            env = OperatingEnvelope(
                sensor_name=sensor,
                lower_bound=lower,
                upper_bound=upper,
                mean=mean,
                std=std,
                percentile_5=p5,
                percentile_95=p95
            )
            self.envelopes[sensor] = env
            stats_rows.append({
                'sensor': sensor, 'mean': mean, 'std': std,
                'lower': lower, 'upper': upper, 'p5': p5, 'p95': p95,
                'range': upper - lower
            })
        
        self.healthy_stats = pd.DataFrame(stats_rows)
        logger.info(f"Learned envelope for {len(self.envelopes)} sensors "
                    f"from {len(healthy)} healthy cycles (RUL>{rul_threshold})")
        
        return self.envelopes
    
    def score_violations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score each row for envelope violations.
        
        Returns:
            DataFrame with per-sensor violation flags and overall score.
        """
        if not self.envelopes:
            raise RuntimeError("Call learn_envelope() first.")
        
        result = df.copy()
        violation_cols = []
        
        for sensor, env in self.envelopes.items():
            if sensor not in df.columns:
                continue
            
            vals = df[sensor].values
            below = vals < env.lower_bound
            above = vals > env.upper_bound
            
            # Normalized violation magnitude
            violation = np.zeros(len(vals))
            rng = env.upper_bound - env.lower_bound
            if rng > 0:
                violation[below] = (env.lower_bound - vals[below]) / rng
                violation[above] = (vals[above] - env.upper_bound) / rng
            
            col_name = f'{sensor}_violation'
            result[col_name] = violation
            violation_cols.append(col_name)
        
        # Overall violation score (mean of all sensor violations)
        if violation_cols:
            result['violation_score'] = result[violation_cols].mean(axis=1)
            result['n_violations'] = (result[violation_cols] > 0).sum(axis=1)
        else:
            result['violation_score'] = 0.0
            result['n_violations'] = 0
        
        n_total = len(result)
        n_violated = (result['violation_score'] > 0).sum()
        logger.info(f"Scored {n_total} cycles: {n_violated} ({n_violated/n_total:.1%}) with violations")
        
        return result
    
    def detect_degradation_onset(self, engine_df: pd.DataFrame,
                                   window: int = 10,
                                   threshold: float = 0.05) -> Dict:
        """
        Detect when an engine begins operating outside its envelope
        (degradation onset point).
        
        Args:
            engine_df: Single engine data sorted by time.
            window: Rolling window for smoothing.
            threshold: Violation score threshold for onset.
            
        Returns:
            Dict with onset cycle and stats.
        """
        scored = self.score_violations(engine_df)
        rolling_score = scored['violation_score'].rolling(window, min_periods=1).mean()
        
        above_thresh = np.where(rolling_score > threshold)[0]
        onset = int(above_thresh[0]) + 1 if len(above_thresh) > 0 else None
        
        return {
            'onset_cycle': onset,
            'total_cycles': len(engine_df),
            'remaining_after_onset': len(engine_df) - onset if onset else None,
            'max_violation': float(scored['violation_score'].max()),
            'mean_violation': float(scored['violation_score'].mean()),
            'violation_trajectory': rolling_score.tolist()
        }
    
    def fleet_violation_summary(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Summarize violations across all engines in fleet."""
        engines = train_df['unit_id'].unique()
        rows = []
        
        for eid in engines:
            eng_data = train_df[train_df['unit_id'] == eid].sort_values('time_cycles')
            scored = self.score_violations(eng_data)
            onset = self.detect_degradation_onset(eng_data)
            
            rows.append({
                'engine_id': eid,
                'total_cycles': len(eng_data),
                'onset_cycle': onset['onset_cycle'],
                'max_violation': onset['max_violation'],
                'mean_violation': onset['mean_violation'],
                'pct_violated': float((scored['violation_score'] > 0).mean()),
                'n_sensors_violated': int(scored['n_violations'].max()),
            })
        
        summary = pd.DataFrame(rows).sort_values('max_violation', ascending=False)
        logger.info(f"Fleet violation summary: {len(engines)} engines analyzed")
        
        return summary
    
    # ------------------------------------------------------------------
    # Plotly visualizations
    # ------------------------------------------------------------------
    def plot_envelope_radar(self, engine_row: pd.Series,
                             title: str = "Operational Envelope Radar") -> go.Figure:
        """Radar chart showing engine sensor values vs envelope boundaries."""
        sensors = list(self.envelopes.keys())[:12]  # max 12 for readability
        
        # Normalize values to 0-1 scale relative to envelope
        norm_vals = []
        norm_lower = []
        norm_upper = []
        
        for s in sensors:
            env = self.envelopes[s]
            rng = env.upper_bound - env.lower_bound
            if rng > 0:
                norm_vals.append((engine_row.get(s, env.mean) - env.lower_bound) / rng)
                norm_lower.append(0)  
                norm_upper.append(1)  
            else:
                norm_vals.append(0.5)
                norm_lower.append(0)
                norm_upper.append(1)
        
        fig = go.Figure()
        
        # Safe zone
        fig.add_trace(go.Scatterpolar(
            r=norm_upper + [norm_upper[0]],
            theta=sensors + [sensors[0]],
            fill='toself',
            fillcolor='rgba(39,174,96,0.1)',
            line=dict(color='#27ae60', dash='dash'),
            name='Safe Envelope'
        ))
        
        # Engine values
        fig.add_trace(go.Scatterpolar(
            r=norm_vals + [norm_vals[0]],
            theta=sensors + [sensors[0]],
            fill='toself',
            fillcolor='rgba(231,76,60,0.15)',
            line=dict(color='#e74c3c', width=2),
            name='Current Values'
        ))
        
        fig.update_layout(
            title=title,
            polar=dict(radialaxis=dict(visible=True, range=[0, 1.5])),
            height=500
        )
        return fig
    
    def plot_violation_timeline(self, engine_df: pd.DataFrame,
                                 onset_info: Dict,
                                 title: str = "Violation Timeline") -> go.Figure:
        """Plot violation score over engine life with onset marker."""
        scored = self.score_violations(engine_df)
        cycles = scored['time_cycles'].values if 'time_cycles' in scored.columns else range(len(scored))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(cycles), y=scored['violation_score'],
            mode='lines', name='Raw Score',
            line=dict(color='rgba(231,76,60,0.3)', width=1)
        ))
        fig.add_trace(go.Scatter(
            x=list(cycles), y=onset_info['violation_trajectory'],
            mode='lines', name='Smoothed Score',
            line=dict(color='#e74c3c', width=2.5)
        ))
        
        if onset_info['onset_cycle']:
            fig.add_vline(x=onset_info['onset_cycle'], line_dash='dash',
                          line_color='#f39c12',
                          annotation_text=f"Onset: cycle {onset_info['onset_cycle']}")
        
        fig.add_hline(y=0.05, line_dash='dot', line_color='gray',
                      annotation_text='Threshold')
        
        fig.update_layout(title=title, xaxis_title='Cycle',
                          yaxis_title='Violation Score', height=350)
        return fig
    
    def plot_sensor_boundaries(self, sensor: str,
                                 engine_df: pd.DataFrame,
                                 title: str = None) -> go.Figure:
        """Plot single sensor values with envelope boundaries."""
        env = self.envelopes.get(sensor)
        if not env:
            raise ValueError(f"No envelope for {sensor}")
        
        title = title or f"Envelope: {sensor}"
        cycles = engine_df['time_cycles'].values if 'time_cycles' in engine_df.columns else range(len(engine_df))
        vals = engine_df[sensor].values
        
        fig = go.Figure()
        
        # Boundaries
        fig.add_hrect(y0=env.lower_bound, y1=env.upper_bound,
                      fillcolor='rgba(39,174,96,0.1)', line_width=0)
        fig.add_hline(y=env.upper_bound, line_dash='dash', line_color='#27ae60')
        fig.add_hline(y=env.lower_bound, line_dash='dash', line_color='#27ae60')
        fig.add_hline(y=env.mean, line_dash='dot', line_color='gray')
        
        # Sensor values colored by violation
        colors = ['#e74c3c' if v < env.lower_bound or v > env.upper_bound else '#3498db' for v in vals]
        fig.add_trace(go.Scatter(
            x=list(cycles), y=vals,
            mode='markers+lines',
            marker=dict(color=colors, size=3),
            line=dict(color='#3498db', width=1),
            name=sensor
        ))
        
        fig.update_layout(title=title, xaxis_title='Cycle',
                          yaxis_title='Value', height=350)
        return fig


# ============================================================
# Standalone test
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Operational Envelope Analyzer")
    print("=" * 60)
    
    from data_loader import CMAPSSDataLoader
    from utils import add_remaining_useful_life
    
    loader = CMAPSSDataLoader('FD001')
    train_df, _, _ = loader.load_all_data()
    train_df = add_remaining_useful_life(train_df)
    
    analyzer = EnvelopeAnalyzer(method='percentile', margin=0.05)
    
    print("\n--- Learning Envelope ---")
    envelopes = analyzer.learn_envelope(train_df, rul_threshold=120)
    print(f"  Sensors: {len(envelopes)}")
    for s, e in list(envelopes.items())[:5]:
        print(f"  {s}: [{e.lower_bound:.2f}, {e.upper_bound:.2f}] (μ={e.mean:.2f})")
    
    print("\n--- Scoring Violations ---")
    eng1 = train_df[train_df['unit_id'] == 1].sort_values('time_cycles')
    scored = analyzer.score_violations(eng1)
    print(f"  Cycles: {len(scored)}")
    print(f"  Max violation: {scored['violation_score'].max():.4f}")
    print(f"  Violated cycles: {(scored['violation_score'] > 0).sum()}")
    
    print("\n--- Degradation Onset Detection ---")
    onset = analyzer.detect_degradation_onset(eng1)
    print(f"  Onset cycle: {onset['onset_cycle']}")
    print(f"  Remaining after onset: {onset['remaining_after_onset']}")
    
    print("\n--- Fleet Summary ---")
    summary = analyzer.fleet_violation_summary(train_df.head(5000))
    print(summary.head(10).to_string(index=False))
    
    print("\n✅ Operational Envelope Analyzer test PASSED")

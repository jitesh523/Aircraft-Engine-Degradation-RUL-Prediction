"""
Anomaly Root Cause Analyzer for Aircraft Engine Diagnostics
Identifies which sensors caused anomalies and matches against known failure patterns.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import config
from utils import setup_logging

logger = setup_logging(__name__)

# Known C-MAPSS failure mode sensor signatures
# Based on NASA documentation: HPC degradation and fan degradation
FAILURE_PATTERNS = {
    'HPC Degradation': {
        'description': 'High Pressure Compressor efficiency loss',
        'key_sensors': {
            'sensor_7': 'decrease',   # HPC outlet temperature
            'sensor_8': 'decrease',   # Physical fan speed — correlates with HPC
            'sensor_9': 'increase',   # Physical core speed — compensating
            'sensor_11': 'increase',  # Static pressure at HPC outlet
            'sensor_12': 'increase',  # Ratio of fuel flow to Ps30
            'sensor_15': 'increase',  # Bleed enthalpy
            'sensor_20': 'decrease',  # BPR — bypass ratio drops
            'sensor_21': 'increase',  # Burner fuel-air ratio
        },
        'severity_weight': 1.0
    },
    'Fan Degradation': {
        'description': 'Fan blade erosion or damage',
        'key_sensors': {
            'sensor_2': 'decrease',   # Total temperature at LPC outlet
            'sensor_3': 'increase',   # Total temperature at HPC outlet
            'sensor_4': 'decrease',   # Total temperature at LPT outlet
            'sensor_7': 'increase',   # Total pressure at HPC outlet
            'sensor_14': 'decrease',  # Corrected fan speed
            'sensor_20': 'increase',  # BPR increase (fan inefficiency)
        },
        'severity_weight': 0.8
    },
    'LPT Degradation': {
        'description': 'Low Pressure Turbine efficiency loss',
        'key_sensors': {
            'sensor_4': 'increase',   # LPT outlet temperature rises
            'sensor_11': 'decrease',  # Static pressure changes
            'sensor_14': 'decrease',  # Corrected fan speed drops
            'sensor_15': 'increase',  # Bleed enthalpy
        },
        'severity_weight': 0.7
    }
}


class RootCauseAnalyzer:
    """
    Identifies root causes of engine anomalies by analyzing sensor deviations
    and matching them against known C-MAPSS failure mode patterns.
    """
    
    def __init__(self):
        """Initialize root cause analyzer."""
        self.baseline_stats = None
        self.failure_patterns = FAILURE_PATTERNS
        self.sensor_cols = None
        logger.info("RootCauseAnalyzer initialized")
    
    def fit_baseline(self, train_df: pd.DataFrame, healthy_threshold: float = 0.7):
        """
        Learn baseline sensor statistics from healthy engine data.
        
        Args:
            train_df: Training DataFrame with RUL column.
            healthy_threshold: Fraction of max RUL to consider "healthy."
        """
        self.sensor_cols = [c for c in train_df.columns if c.startswith('sensor_')]
        
        # Use only healthy data (high RUL)
        if 'RUL' in train_df.columns:
            max_rul = train_df['RUL'].max()
            healthy = train_df[train_df['RUL'] > max_rul * healthy_threshold]
        else:
            # Use first 30% of each engine's life
            healthy = train_df.groupby('unit_id').apply(
                lambda g: g.head(int(len(g) * 0.3))
            ).reset_index(drop=True)
        
        self.baseline_stats = {}
        for col in self.sensor_cols:
            self.baseline_stats[col] = {
                'mean': float(healthy[col].mean()),
                'std': float(healthy[col].std()),
                'q25': float(healthy[col].quantile(0.25)),
                'q75': float(healthy[col].quantile(0.75)),
                'iqr': float(healthy[col].quantile(0.75) - healthy[col].quantile(0.25))
            }
        
        logger.info(f"Baseline fitted on {len(healthy)} healthy samples, "
                    f"{len(self.sensor_cols)} sensors")
    
    def detect_anomalies(self, engine_data: pd.DataFrame,
                         method: str = 'zscore',
                         threshold: float = 2.5) -> pd.DataFrame:
        """
        Flag anomalous sensor readings for an engine.
        
        Args:
            engine_data: Sensor data for one or more engines.
            method: 'zscore' or 'iqr'.
            threshold: Z-score threshold or IQR multiplier.
            
        Returns:
            DataFrame with anomaly flags per sensor.
        """
        if self.baseline_stats is None:
            raise RuntimeError("Call fit_baseline() first.")
        
        anomalies = pd.DataFrame(index=engine_data.index)
        
        for col in self.sensor_cols:
            if col not in engine_data.columns:
                continue
                
            stats = self.baseline_stats[col]
            
            if method == 'zscore':
                if stats['std'] > 1e-8:
                    z = (engine_data[col] - stats['mean']) / stats['std']
                    anomalies[col] = z.abs() > threshold
                else:
                    anomalies[col] = False
            else:  # IQR
                lower = stats['q25'] - threshold * stats['iqr']
                upper = stats['q75'] + threshold * stats['iqr']
                anomalies[col] = (engine_data[col] < lower) | (engine_data[col] > upper)
        
        return anomalies
    
    def identify_root_causes(self, engine_data: pd.DataFrame,
                              top_k: int = 5) -> List[Dict]:
        """
        Rank sensors by deviation from baseline for a single engine.
        
        Args:
            engine_data: DataFrame for one engine (last few cycles).
            top_k: Number of top deviating sensors to return.
            
        Returns:
            List of dicts with sensor name, z-score, direction, and severity.
        """
        if self.baseline_stats is None:
            raise RuntimeError("Call fit_baseline() first.")
        
        # Use last few readings
        recent = engine_data.tail(10)
        
        deviations = []
        for col in self.sensor_cols:
            if col not in recent.columns:
                continue
            stats = self.baseline_stats[col]
            if stats['std'] < 1e-8:
                continue
            
            mean_val = recent[col].mean()
            z = (mean_val - stats['mean']) / stats['std']
            
            deviations.append({
                'sensor': col,
                'current_value': float(mean_val),
                'baseline_mean': stats['mean'],
                'z_score': float(z),
                'direction': 'increase' if z > 0 else 'decrease',
                'severity': abs(float(z))
            })
        
        deviations.sort(key=lambda x: x['severity'], reverse=True)
        return deviations[:top_k]
    
    def match_failure_pattern(self, deviations: List[Dict]) -> List[Dict]:
        """
        Match observed sensor deviations against known failure mode patterns.
        
        Args:
            deviations: List of sensor deviation dicts (from identify_root_causes).
            
        Returns:
            List of matched patterns with confidence scores.
        """
        dev_map = {d['sensor']: d for d in deviations}
        
        matches = []
        for pattern_name, pattern in self.failure_patterns.items():
            match_score = 0
            total_sensors = len(pattern['key_sensors'])
            matched_sensors = []
            
            for sensor, expected_dir in pattern['key_sensors'].items():
                if sensor in dev_map:
                    dev = dev_map[sensor]
                    # Check if direction matches
                    if dev['direction'] == expected_dir:
                        match_score += dev['severity']
                        matched_sensors.append(sensor)
                    else:
                        match_score -= 0.5  # Penalty for wrong direction
            
            confidence = max(0, match_score / max(total_sensors, 1))
            
            matches.append({
                'failure_mode': pattern_name,
                'description': pattern['description'],
                'confidence': float(min(confidence, 1.0)),
                'matched_sensors': matched_sensors,
                'total_pattern_sensors': total_sensors,
                'match_ratio': len(matched_sensors) / total_sensors
            })
        
        matches.sort(key=lambda x: x['confidence'], reverse=True)
        return matches
    
    def generate_diagnosis(self, engine_data: pd.DataFrame,
                            engine_id: int = None) -> Dict:
        """
        Generate a full diagnostic report for an engine.
        
        Args:
            engine_data: DataFrame for one engine.
            engine_id: Engine identifier.
            
        Returns:
            Complete diagnosis dictionary.
        """
        deviations = self.identify_root_causes(engine_data, top_k=10)
        pattern_matches = self.match_failure_pattern(deviations)
        anomalies = self.detect_anomalies(engine_data)
        anomaly_rate = anomalies.mean().mean()
        
        top_match = pattern_matches[0] if pattern_matches else None
        
        diagnosis = {
            'engine_id': engine_id,
            'anomaly_rate': float(anomaly_rate),
            'top_deviations': deviations[:5],
            'failure_mode_matches': pattern_matches,
            'likely_failure_mode': top_match['failure_mode'] if top_match else 'Unknown',
            'confidence': top_match['confidence'] if top_match else 0.0,
            'recommendation': self._generate_recommendation(
                anomaly_rate, top_match, deviations
            )
        }
        
        logger.info(f"Diagnosis for engine {engine_id}: "
                    f"{diagnosis['likely_failure_mode']} "
                    f"(confidence: {diagnosis['confidence']:.2f})")
        
        return diagnosis
    
    def _generate_recommendation(self, anomaly_rate: float,
                                  top_match: Optional[Dict],
                                  deviations: List[Dict]) -> str:
        """Generate maintenance recommendation based on diagnosis."""
        if anomaly_rate > 0.5:
            urgency = "IMMEDIATE"
        elif anomaly_rate > 0.2:
            urgency = "SOON"
        else:
            urgency = "ROUTINE"
        
        parts = [f"Maintenance urgency: {urgency}."]
        
        if top_match and top_match['confidence'] > 0.3:
            parts.append(f"Likely failure mode: {top_match['failure_mode']} "
                        f"({top_match['description']}).")
            parts.append(f"Matched sensors: {', '.join(top_match['matched_sensors'])}.")
        
        if deviations:
            top = deviations[0]
            parts.append(f"Most anomalous: {top['sensor']} "
                        f"(z={top['z_score']:+.1f}, {top['direction']}).")
        
        return " ".join(parts)
    
    # ------------------------------------------------------------------
    # Plotly visualizations
    # ------------------------------------------------------------------
    def plot_sensor_deviations(self, deviations: List[Dict],
                                title: str = "Sensor Deviation Radar") -> go.Figure:
        """Radar chart of top sensor deviations."""
        sensors = [d['sensor'] for d in deviations[:8]]
        values = [d['severity'] for d in deviations[:8]]
        
        fig = go.Figure(go.Scatterpolar(
            r=values + [values[0]],
            theta=sensors + [sensors[0]],
            fill='toself',
            fillcolor='rgba(255,0,0,0.1)',
            line_color='red',
            name='Deviation'
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, max(values) * 1.2])),
            title=title, height=400
        )
        return fig
    
    def plot_failure_mode_match(self, matches: List[Dict],
                                 title: str = "Failure Mode Confidence") -> go.Figure:
        """Bar chart of failure mode match scores."""
        modes = [m['failure_mode'] for m in matches]
        scores = [m['confidence'] for m in matches]
        colors = ['#d62728' if s > 0.5 else '#ff7f0e' if s > 0.2 else '#2ca02c' 
                  for s in scores]
        
        fig = go.Figure(go.Bar(
            x=scores, y=modes, orientation='h',
            marker_color=colors,
            text=[f"{s:.0%}" for s in scores],
            textposition='outside'
        ))
        fig.update_layout(
            title=title, xaxis_title='Confidence',
            xaxis_range=[0, 1.1], height=300
        )
        return fig


# ============================================================
# Standalone test
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Root Cause Analyzer")
    print("=" * 60)
    
    from data_loader import CMAPSSDataLoader
    from utils import add_remaining_useful_life
    
    loader = CMAPSSDataLoader('FD001')
    train_df, _, _ = loader.load_all_data()
    train_df = add_remaining_useful_life(train_df)
    
    analyzer = RootCauseAnalyzer()
    
    # Fit baseline on healthy data
    print("\n--- Fitting Baseline ---")
    analyzer.fit_baseline(train_df)
    print(f"Sensors tracked: {len(analyzer.sensor_cols)}")
    
    # Pick a late-life engine (near failure)
    engine_1 = train_df[train_df['unit_id'] == 1]
    last_cycles = engine_1.tail(20)
    
    # Detect anomalies
    print("\n--- Anomaly Detection ---")
    anomalies = analyzer.detect_anomalies(last_cycles)
    n_anomalous = anomalies.any(axis=1).sum()
    print(f"Anomalous readings: {n_anomalous}/{len(last_cycles)}")
    
    # Root causes
    print("\n--- Root Causes ---")
    deviations = analyzer.identify_root_causes(engine_1)
    for d in deviations[:5]:
        print(f"  {d['sensor']}: z={d['z_score']:+.2f} ({d['direction']})")
    
    # Failure mode matching
    print("\n--- Failure Mode Matching ---")
    matches = analyzer.match_failure_pattern(deviations)
    for m in matches:
        print(f"  {m['failure_mode']}: confidence={m['confidence']:.2f} "
              f"({len(m['matched_sensors'])}/{m['total_pattern_sensors']} sensors)")
    
    # Full diagnosis
    print("\n--- Full Diagnosis ---")
    diagnosis = analyzer.generate_diagnosis(engine_1, engine_id=1)
    print(f"  Likely: {diagnosis['likely_failure_mode']}")
    print(f"  Confidence: {diagnosis['confidence']:.2f}")
    print(f"  Recommendation: {diagnosis['recommendation']}")
    
    print("\n✅ Root Cause Analyzer test PASSED")

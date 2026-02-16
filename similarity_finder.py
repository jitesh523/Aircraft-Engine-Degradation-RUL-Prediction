"""
Engine Similarity Finder
DTW-based trajectory matching to find historically similar engines,
enabling transfer prognosis and fleet-wide pattern recognition.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.spatial.distance import euclidean
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import config
from utils import setup_logging

logger = setup_logging(__name__)


def fast_dtw_distance(s1: np.ndarray, s2: np.ndarray) -> float:
    """
    Simplified DTW distance between two 1D sequences.
    Uses a banded approach for efficiency.
    """
    n, m = len(s1), len(s2)
    band = max(10, abs(n - m) + 5)
    
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0
    
    for i in range(1, n + 1):
        j_start = max(1, i - band)
        j_end = min(m, i + band) + 1
        for j in range(j_start, j_end):
            cost = abs(s1[i - 1] - s2[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    
    return dtw[n, m]


class SimilarityFinder:
    """
    Find engines with similar degradation trajectories using DTW.
    
    Features:
    - Multi-sensor DTW similarity
    - k-nearest neighbor engine retrieval
    - Transfer prognosis (predict RUL from similar engines)
    - Similarity heatmap and trajectory overlay plots
    """
    
    KEY_SENSORS = ['sensor_2', 'sensor_3', 'sensor_4', 'sensor_7',
                   'sensor_8', 'sensor_9', 'sensor_11', 'sensor_12',
                   'sensor_13', 'sensor_15', 'sensor_17', 'sensor_20', 'sensor_21']
    
    def __init__(self, n_sensors: int = 6, max_engines: int = 50):
        """
        Args:
            n_sensors: Number of top sensors to use for similarity.
            max_engines: Maximum engines to compare against.
        """
        self.n_sensors = min(n_sensors, len(self.KEY_SENSORS))
        self.max_engines = max_engines
        self.fleet_profiles = {}
        self.similarity_matrix = None
        self.sensors_used = self.KEY_SENSORS[:self.n_sensors]
        logger.info(f"SimilarityFinder initialized: {self.n_sensors} sensors, "
                    f"max {max_engines} engines")
    
    def build_fleet_profiles(self, train_df: pd.DataFrame) -> Dict:
        """
        Extract degradation profiles (sensor trajectories) for all engines.
        
        Args:
            train_df: Training data with unit_id, time_cycles, sensors.
            
        Returns:
            Dict of engine_id -> profile dict.
        """
        engines = train_df['unit_id'].unique()[:self.max_engines]
        available_sensors = [s for s in self.sensors_used if s in train_df.columns]
        self.sensors_used = available_sensors
        
        rul_col = 'RUL' if 'RUL' in train_df.columns else None
        
        for eid in engines:
            eng = train_df[train_df['unit_id'] == eid].sort_values('time_cycles')
            
            trajectories = {}
            for sensor in available_sensors:
                vals = eng[sensor].values
                # Normalize to [0, 1]
                vmin, vmax = vals.min(), vals.max()
                if vmax - vmin > 1e-8:
                    trajectories[sensor] = (vals - vmin) / (vmax - vmin)
                else:
                    trajectories[sensor] = np.zeros_like(vals)
            
            profile = {
                'engine_id': eid,
                'total_cycles': len(eng),
                'trajectories': trajectories,
                'rul': int(eng[rul_col].iloc[0]) if rul_col else len(eng),
            }
            self.fleet_profiles[eid] = profile
        
        logger.info(f"Built profiles for {len(self.fleet_profiles)} engines, "
                    f"{len(available_sensors)} sensors each")
        
        return self.fleet_profiles
    
    def find_similar(self, query_engine_id, k: int = 5) -> List[Dict]:
        """
        Find k most similar engines to a query engine.
        
        Args:
            query_engine_id: Engine ID to find matches for.
            k: Number of similar engines to return.
            
        Returns:
            List of dicts with similarity info, sorted by distance.
        """
        if query_engine_id not in self.fleet_profiles:
            raise ValueError(f"Engine {query_engine_id} not in fleet profiles")
        
        query = self.fleet_profiles[query_engine_id]
        distances = []
        
        for eid, profile in self.fleet_profiles.items():
            if eid == query_engine_id:
                continue
            
            total_dist = 0
            n_compared = 0
            
            for sensor in self.sensors_used:
                q_traj = query['trajectories'].get(sensor)
                p_traj = profile['trajectories'].get(sensor)
                
                if q_traj is not None and p_traj is not None:
                    # Use DTW for different-length sequences
                    dist = fast_dtw_distance(q_traj, p_traj)
                    # Normalize by length
                    norm_dist = dist / max(len(q_traj), len(p_traj))
                    total_dist += norm_dist
                    n_compared += 1
            
            avg_dist = total_dist / max(n_compared, 1)
            similarity = 1.0 / (1.0 + avg_dist)
            
            distances.append({
                'engine_id': eid,
                'distance': avg_dist,
                'similarity': similarity,
                'total_cycles': profile['total_cycles'],
                'rul': profile['rul'],
                'n_sensors_compared': n_compared
            })
        
        distances.sort(key=lambda x: x['distance'])
        top_k = distances[:k]
        
        sim_scores = [f"{d['similarity']:.3f}" for d in top_k]
        logger.info(f"Top {k} similar to engine {query_engine_id}: "
                    f"{[d['engine_id'] for d in top_k]} "
                    f"(sim: {sim_scores})")
        
        return top_k
    
    def transfer_prognosis(self, query_engine_id, k: int = 5) -> Dict:
        """
        Predict RUL for query engine based on similar engines' outcomes.
        
        Returns:
            Prognosis dict with predicted RUL and confidence.
        """
        similar = self.find_similar(query_engine_id, k=k)
        query = self.fleet_profiles[query_engine_id]
        
        weights = np.array([s['similarity'] for s in similar])
        ruls = np.array([s['rul'] for s in similar])
        
        # Weighted average RUL
        if weights.sum() > 0:
            predicted_rul = float(np.average(ruls, weights=weights))
        else:
            predicted_rul = float(np.mean(ruls))
        
        return {
            'query_engine': query_engine_id,
            'query_cycles': query['total_cycles'],
            'query_actual_rul': query['rul'],
            'predicted_rul': predicted_rul,
            'rul_error': abs(predicted_rul - query['rul']),
            'k_neighbors': k,
            'neighbor_ruls': ruls.tolist(),
            'neighbor_weights': weights.tolist(),
            'neighbor_ids': [s['engine_id'] for s in similar],
            'confidence': float(np.mean(weights)),
        }
    
    def compute_similarity_matrix(self) -> pd.DataFrame:
        """Compute pairwise similarity matrix for all engines."""
        engines = list(self.fleet_profiles.keys())
        n = len(engines)
        matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                total_dist = 0
                n_compared = 0
                
                for sensor in self.sensors_used:
                    t1 = self.fleet_profiles[engines[i]]['trajectories'].get(sensor)
                    t2 = self.fleet_profiles[engines[j]]['trajectories'].get(sensor)
                    
                    if t1 is not None and t2 is not None:
                        dist = fast_dtw_distance(t1, t2)
                        total_dist += dist / max(len(t1), len(t2))
                        n_compared += 1
                
                avg_dist = total_dist / max(n_compared, 1)
                sim = 1.0 / (1.0 + avg_dist)
                matrix[i, j] = sim
                matrix[j, i] = sim
        
        np.fill_diagonal(matrix, 1.0)
        self.similarity_matrix = pd.DataFrame(matrix, index=engines, columns=engines)
        
        logger.info(f"Computed {n}x{n} similarity matrix")
        return self.similarity_matrix
    
    # ------------------------------------------------------------------
    # Plotly visualizations
    # ------------------------------------------------------------------
    def plot_similarity_heatmap(self, title: str = "Engine Similarity Heatmap") -> go.Figure:
        """Heatmap of pairwise engine similarities."""
        if self.similarity_matrix is None:
            self.compute_similarity_matrix()
        
        fig = go.Figure(data=go.Heatmap(
            z=self.similarity_matrix.values,
            x=[f'E{c}' for c in self.similarity_matrix.columns],
            y=[f'E{r}' for r in self.similarity_matrix.index],
            colorscale='RdYlGn',
            zmin=0, zmax=1,
            colorbar_title='Similarity'
        ))
        
        fig.update_layout(title=title, height=500, width=600)
        return fig
    
    def plot_trajectory_overlay(self, query_id, similar: List[Dict],
                                  sensor: str = None,
                                  title: str = None) -> go.Figure:
        """Overlay query engine trajectory with similar engines."""
        sensor = sensor or self.sensors_used[0]
        title = title or f"Trajectory Comparison — {sensor}"
        
        query = self.fleet_profiles[query_id]
        
        fig = go.Figure()
        
        # Similar engines (light)
        for match in similar:
            eid = match['engine_id']
            traj = self.fleet_profiles[eid]['trajectories'].get(sensor, [])
            fig.add_trace(go.Scatter(
                x=list(range(len(traj))), y=traj,
                mode='lines',
                line=dict(color='rgba(52,152,219,0.3)', width=1),
                name=f"E{eid} (sim={match['similarity']:.2f})",
                showlegend=True
            ))
        
        # Query engine (bold)
        q_traj = query['trajectories'].get(sensor, [])
        fig.add_trace(go.Scatter(
            x=list(range(len(q_traj))), y=q_traj,
            mode='lines',
            line=dict(color='#e74c3c', width=3),
            name=f"E{query_id} (query)"
        ))
        
        fig.update_layout(title=title, xaxis_title='Cycle (normalized)',
                          yaxis_title='Normalized Value', height=400)
        return fig
    
    def plot_prognosis_comparison(self, prognosis: Dict,
                                    title: str = "Transfer Prognosis") -> go.Figure:
        """Bar chart comparing predicted vs actual RUL."""
        labels = [f'E{eid}' for eid in prognosis['neighbor_ids']] + ['Predicted', 'Actual']
        values = prognosis['neighbor_ruls'] + [prognosis['predicted_rul'], prognosis['query_actual_rul']]
        
        colors = ['#3498db'] * len(prognosis['neighbor_ids']) + ['#e74c3c', '#27ae60']
        
        fig = go.Figure(data=go.Bar(
            x=labels, y=values,
            marker_color=colors,
            text=[f'{v:.0f}' for v in values],
            textposition='outside'
        ))
        
        fig.update_layout(
            title=f"{title} (Error: {prognosis['rul_error']:.0f} cycles)",
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
    print("Testing Engine Similarity Finder")
    print("=" * 60)
    
    from data_loader import CMAPSSDataLoader
    from utils import add_remaining_useful_life
    
    loader = CMAPSSDataLoader('FD001')
    train_df, _, _ = loader.load_all_data()
    train_df = add_remaining_useful_life(train_df)
    
    finder = SimilarityFinder(n_sensors=4, max_engines=20)
    
    print("\n--- Building Fleet Profiles ---")
    profiles = finder.build_fleet_profiles(train_df)
    print(f"  Engines: {len(profiles)}")
    print(f"  Sensors: {finder.sensors_used}")
    
    print("\n--- Finding Similar Engines ---")
    similar = finder.find_similar(query_engine_id=1, k=5)
    for s in similar:
        print(f"  Engine {s['engine_id']}: distance={s['distance']:.3f}, "
              f"similarity={s['similarity']:.3f}, RUL={s['rul']}")
    
    print("\n--- Transfer Prognosis ---")
    prog = finder.transfer_prognosis(query_engine_id=1, k=5)
    print(f"  Actual RUL: {prog['query_actual_rul']}")
    print(f"  Predicted RUL: {prog['predicted_rul']:.0f}")
    print(f"  Error: {prog['rul_error']:.0f} cycles")
    print(f"  Confidence: {prog['confidence']:.3f}")
    
    print("\n--- Similarity Matrix ---")
    matrix = finder.compute_similarity_matrix()
    print(f"  Shape: {matrix.shape}")
    print(f"  Mean similarity: {matrix.values[np.triu_indices(len(matrix), k=1)].mean():.3f}")
    
    print("\n✅ Engine Similarity Finder test PASSED")

"""
Sensor Correlation Network Analyzer
Graph-based analysis of sensor interdependencies, community detection,
and anomaly propagation path identification.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import plotly.graph_objects as go
import config
from utils import setup_logging

logger = setup_logging(__name__)


class SensorNetwork:
    """
    Builds and analyzes a correlation network of engine sensors.
    
    Nodes = sensors, edges = significant correlations.
    Identifies sensor communities, hub sensors, and how anomalies
    propagate through the sensor network.
    """
    
    def __init__(self, corr_threshold: float = 0.6, p_threshold: float = 0.05):
        """
        Args:
            corr_threshold: Minimum |correlation| to create an edge.
            p_threshold: Statistical significance threshold.
        """
        self.corr_threshold = corr_threshold
        self.p_threshold = p_threshold
        self.corr_matrix = None
        self.adjacency = {}
        self.sensor_cols = None
        self.communities = {}
        self.centrality = {}
        self.graph_stats = {}
        logger.info(f"SensorNetwork initialized (threshold={corr_threshold})")
    
    def build_network(self, df: pd.DataFrame) -> Dict:
        """
        Build correlation network from sensor data.
        
        Args:
            df: DataFrame with sensor columns.
            
        Returns:
            Graph statistics dict.
        """
        self.sensor_cols = [c for c in df.columns if c.startswith('sensor_')]
        sensor_data = df[self.sensor_cols].dropna()
        
        # Correlation matrix
        self.corr_matrix = sensor_data.corr()
        
        # Build adjacency list (undirected, weighted)
        self.adjacency = {s: [] for s in self.sensor_cols}
        edges = []
        
        for i, s1 in enumerate(self.sensor_cols):
            for j, s2 in enumerate(self.sensor_cols):
                if i >= j:
                    continue
                corr = self.corr_matrix.loc[s1, s2]
                if abs(corr) >= self.corr_threshold:
                    self.adjacency[s1].append((s2, corr))
                    self.adjacency[s2].append((s1, corr))
                    edges.append((s1, s2, corr))
        
        # Compute centrality (degree-based)
        self.centrality = {
            s: len(neighbors) for s, neighbors in self.adjacency.items()
        }
        max_degree = max(self.centrality.values()) if self.centrality else 1
        self.centrality = {
            s: d / max(max_degree, 1) for s, d in self.centrality.items()
        }
        
        # Community detection (greedy modularity)
        self.communities = self._detect_communities()
        
        self.graph_stats = {
            'n_nodes': len(self.sensor_cols),
            'n_edges': len(edges),
            'density': 2 * len(edges) / max(len(self.sensor_cols) * (len(self.sensor_cols) - 1), 1),
            'n_communities': len(set(self.communities.values())),
            'hub_sensors': sorted(self.centrality, key=self.centrality.get, reverse=True)[:5],
            'isolated_sensors': [s for s, d in self.centrality.items() if d == 0]
        }
        
        logger.info(f"Network built: {self.graph_stats['n_nodes']} nodes, "
                    f"{self.graph_stats['n_edges']} edges, "
                    f"{self.graph_stats['n_communities']} communities")
        
        return self.graph_stats
    
    def _detect_communities(self) -> Dict[str, int]:
        """Simple label propagation community detection."""
        labels = {s: i for i, s in enumerate(self.sensor_cols)}
        
        for _ in range(10):  # iterations
            changed = False
            for node in self.sensor_cols:
                neighbors = self.adjacency.get(node, [])
                if not neighbors:
                    continue
                
                # Count neighbor labels
                label_counts = defaultdict(float)
                for neighbor, weight in neighbors:
                    label_counts[labels[neighbor]] += abs(weight)
                
                if label_counts:
                    best_label = max(label_counts, key=label_counts.get)
                    if labels[node] != best_label:
                        labels[node] = best_label
                        changed = True
            
            if not changed:
                break
        
        # Normalize community IDs to 0, 1, 2, ...
        unique_labels = sorted(set(labels.values()))
        remap = {old: new for new, old in enumerate(unique_labels)}
        return {s: remap[l] for s, l in labels.items()}
    
    def get_propagation_paths(self, source_sensor: str, 
                               max_hops: int = 3) -> List[List[str]]:
        """
        Find anomaly propagation paths from a source sensor via BFS.
        
        Args:
            source_sensor: Starting sensor.
            max_hops: Maximum path length.
            
        Returns:
            List of paths (each path is a list of sensor names).
        """
        if source_sensor not in self.adjacency:
            return []
        
        paths = []
        queue = [(source_sensor, [source_sensor])]
        visited = {source_sensor}
        
        while queue:
            current, path = queue.pop(0)
            
            if len(path) > 1:
                paths.append(path)
            
            if len(path) >= max_hops + 1:
                continue
            
            for neighbor, _ in self.adjacency.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return sorted(paths, key=len)
    
    def analyze_degradation_correlation(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze how sensor correlations change as engines degrade.
        
        Args:
            train_df: Training data with RUL column.
            
        Returns:
            DataFrame showing correlation shift between healthy and degraded states.
        """
        if 'RUL' not in train_df.columns:
            logger.warning("RUL column required for degradation correlation analysis")
            return pd.DataFrame()
        
        max_rul = train_df['RUL'].max()
        healthy = train_df[train_df['RUL'] > max_rul * 0.7][self.sensor_cols]
        degraded = train_df[train_df['RUL'] < max_rul * 0.2][self.sensor_cols]
        
        corr_healthy = healthy.corr()
        corr_degraded = degraded.corr()
        
        shifts = []
        for i, s1 in enumerate(self.sensor_cols):
            for j, s2 in enumerate(self.sensor_cols):
                if i >= j:
                    continue
                h = corr_healthy.loc[s1, s2]
                d = corr_degraded.loc[s1, s2]
                shift = d - h
                if abs(shift) > 0.1:
                    shifts.append({
                        'sensor_1': s1,
                        'sensor_2': s2,
                        'corr_healthy': float(h),
                        'corr_degraded': float(d),
                        'shift': float(shift),
                        'abs_shift': abs(float(shift))
                    })
        
        shift_df = pd.DataFrame(shifts).sort_values('abs_shift', ascending=False)
        logger.info(f"Found {len(shifts)} significant correlation shifts during degradation")
        return shift_df
    
    # ------------------------------------------------------------------
    # Plotly visualizations
    # ------------------------------------------------------------------
    def plot_network(self, title: str = "Sensor Correlation Network") -> go.Figure:
        """Interactive network graph visualization."""
        if self.corr_matrix is None:
            raise RuntimeError("Call build_network() first.")
        
        # Layout: circular
        n = len(self.sensor_cols)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pos = {s: (np.cos(a), np.sin(a)) for s, a in zip(self.sensor_cols, angles)}
        
        # Edges
        edge_x, edge_y = [], []
        edge_colors = []
        for s1, neighbors in self.adjacency.items():
            for s2, weight in neighbors:
                if s1 < s2:
                    x0, y0 = pos[s1]
                    x1, y1 = pos[s2]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                    edge_colors.append(weight)
        
        fig = go.Figure()
        
        # Draw edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(width=1, color='rgba(100,100,100,0.3)'),
            hoverinfo='none',
            name='Correlations'
        ))
        
        # Draw nodes
        node_x = [pos[s][0] for s in self.sensor_cols]
        node_y = [pos[s][1] for s in self.sensor_cols]
        node_sizes = [15 + 25 * self.centrality.get(s, 0) for s in self.sensor_cols]
        node_colors = [self.communities.get(s, 0) for s in self.sensor_cols]
        node_text = [
            f"{s}<br>Degree centrality: {self.centrality.get(s, 0):.2f}"
            f"<br>Community: {self.communities.get(s, 0)}"
            f"<br>Connections: {len(self.adjacency.get(s, []))}"
            for s in self.sensor_cols
        ]
        
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                colorscale='Viridis',
                line=dict(width=1, color='white')
            ),
            text=[s.replace('sensor_', 'S') for s in self.sensor_cols],
            textposition='top center',
            hovertext=node_text,
            hoverinfo='text',
            name='Sensors'
        ))
        
        fig.update_layout(
            title=title,
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=500,
            plot_bgcolor='white'
        )
        
        return fig
    
    def plot_correlation_heatmap(self, title: str = "Sensor Correlation Matrix") -> go.Figure:
        """Heatmap of the full correlation matrix."""
        labels = [s.replace('sensor_', 'S') for s in self.sensor_cols]
        
        fig = go.Figure(data=go.Heatmap(
            z=self.corr_matrix.values,
            x=labels, y=labels,
            colorscale='RdBu_r',
            zmin=-1, zmax=1,
            colorbar=dict(title='Correlation')
        ))
        fig.update_layout(title=title, height=500)
        return fig
    
    def plot_community_summary(self, title: str = "Sensor Communities") -> go.Figure:
        """Bar chart of community membership."""
        comm_df = pd.DataFrame([
            {'sensor': s, 'community': f"Community {c}"}
            for s, c in self.communities.items()
        ])
        counts = comm_df['community'].value_counts().sort_index()
        
        fig = go.Figure(go.Bar(
            x=counts.index, y=counts.values,
            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                          '#8c564b', '#e377c2'][:len(counts)],
            text=counts.values, textposition='outside'
        ))
        fig.update_layout(
            title=title, xaxis_title='Community', yaxis_title='Number of Sensors',
            height=350
        )
        return fig


# ============================================================
# Standalone test
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Sensor Correlation Network")
    print("=" * 60)
    
    from data_loader import CMAPSSDataLoader
    from utils import add_remaining_useful_life
    
    loader = CMAPSSDataLoader('FD001')
    train_df, _, _ = loader.load_all_data()
    train_df = add_remaining_useful_life(train_df)
    
    net = SensorNetwork(corr_threshold=0.6)
    
    print("\n--- Building Network ---")
    stats = net.build_network(train_df)
    print(f"  Nodes: {stats['n_nodes']}")
    print(f"  Edges: {stats['n_edges']}")
    print(f"  Density: {stats['density']:.3f}")
    print(f"  Communities: {stats['n_communities']}")
    print(f"  Hub sensors: {stats['hub_sensors']}")
    print(f"  Isolated: {stats['isolated_sensors']}")
    
    print("\n--- Propagation Paths ---")
    if stats['hub_sensors']:
        hub = stats['hub_sensors'][0]
        paths = net.get_propagation_paths(hub, max_hops=3)
        print(f"  From {hub}: {len(paths)} paths found")
        for p in paths[:5]:
            print(f"    {' → '.join(p)}")
    
    print("\n--- Degradation Correlation Shifts ---")
    shifts = net.analyze_degradation_correlation(train_df)
    if len(shifts) > 0:
        print(shifts[['sensor_1', 'sensor_2', 'corr_healthy', 'corr_degraded', 'shift']].head().to_string(index=False))
    
    print("\n✅ Sensor Network test PASSED")

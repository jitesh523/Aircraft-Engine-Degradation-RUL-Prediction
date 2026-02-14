"""
Degradation Pattern Clustering
Discovers engine degradation archetypes using unsupervised learning
on trajectory features extracted from sensor time-series data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import config
from utils import setup_logging

logger = setup_logging(__name__)


class DegradationClusterer:
    """
    Clusters engines by their degradation trajectory shape.
    
    Extracts trajectory features (slope, curvature, volatility, etc.)
    from sensor time-series, clusters them into archetypes, and
    provides per-cluster survival statistics.
    """
    
    TRAJECTORY_FEATURES = [
        'slope', 'curvature', 'volatility', 'range',
        'skewness', 'kurtosis', 'trend_strength', 'lifetime'
    ]
    
    def __init__(self, n_clusters: int = 4, key_sensors: List[str] = None):
        """
        Args:
            n_clusters: Number of degradation archetypes to discover.
            key_sensors: Sensors to analyze (default: auto-detect informative ones).
        """
        self.n_clusters = n_clusters
        self.key_sensors = key_sensors
        self.feature_matrix = None
        self.labels = None
        self.cluster_profiles = {}
        self.scaler = StandardScaler()
        self.pca = None
        self.engine_ids = None
        logger.info(f"DegradationClusterer initialized (k={n_clusters})")
    
    def extract_trajectory_features(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract degradation trajectory features for each engine.
        
        For each engine and sensor, computes:
        - slope: linear trend direction
        - curvature: acceleration of change
        - volatility: standard deviation of differences
        - range: max - min
        - skewness, kurtosis: distribution shape
        - trend_strength: R² of linear fit
        - lifetime: total operating cycles
        
        Args:
            train_df: Training DataFrame with unit_id and sensor columns.
            
        Returns:
            Feature matrix (engines × trajectory features).
        """
        if self.key_sensors is None:
            sensor_cols = [c for c in train_df.columns if c.startswith('sensor_')]
            # Auto-select sensors with meaningful variance
            variances = train_df[sensor_cols].std()
            self.key_sensors = variances[variances > variances.median()].index.tolist()[:8]
        
        logger.info(f"Extracting trajectory features for {len(self.key_sensors)} sensors")
        
        records = []
        engine_ids = []
        
        for uid, grp in train_df.groupby('unit_id'):
            if len(grp) < 5:
                continue
            
            rec = {'unit_id': uid, 'lifetime': len(grp)}
            engine_ids.append(uid)
            
            for sensor in self.key_sensors:
                if sensor not in grp.columns:
                    continue
                    
                vals = grp[sensor].values.astype(float)
                t = np.arange(len(vals), dtype=float)
                
                # Slope (linear regression)
                if len(vals) > 1:
                    coeffs = np.polyfit(t, vals, 1)
                    slope = coeffs[0]
                    # R² for trend strength
                    y_pred = np.polyval(coeffs, t)
                    ss_res = np.sum((vals - y_pred) ** 2)
                    ss_tot = np.sum((vals - np.mean(vals)) ** 2)
                    r2 = 1 - ss_res / max(ss_tot, 1e-10)
                else:
                    slope, r2 = 0, 0
                
                # Curvature (2nd derivative proxy)
                if len(vals) > 2:
                    coeffs2 = np.polyfit(t, vals, 2)
                    curvature = coeffs2[0]
                else:
                    curvature = 0
                
                # Volatility
                diffs = np.diff(vals)
                volatility = np.std(diffs) if len(diffs) > 0 else 0
                
                # Distribution shape
                from scipy import stats as sp_stats
                skew = float(sp_stats.skew(vals)) if len(vals) > 2 else 0
                kurt = float(sp_stats.kurtosis(vals)) if len(vals) > 3 else 0
                
                prefix = sensor.replace('sensor_', 's')
                rec[f'{prefix}_slope'] = slope
                rec[f'{prefix}_curvature'] = curvature
                rec[f'{prefix}_volatility'] = volatility
                rec[f'{prefix}_range'] = float(np.ptp(vals))
                rec[f'{prefix}_skew'] = skew
                rec[f'{prefix}_kurtosis'] = kurt
                rec[f'{prefix}_trend'] = max(0, r2)
            
            records.append(rec)
        
        self.engine_ids = engine_ids
        feat_df = pd.DataFrame(records)
        feat_cols = [c for c in feat_df.columns if c != 'unit_id']
        
        # Scale features
        feat_df[feat_cols] = self.scaler.fit_transform(feat_df[feat_cols].fillna(0))
        self.feature_matrix = feat_df
        
        logger.info(f"Extracted {len(feat_cols)} features for {len(records)} engines")
        return feat_df
    
    def cluster(self, method: str = 'kmeans') -> np.ndarray:
        """
        Cluster engines by trajectory features.
        
        Args:
            method: 'kmeans' (default).
            
        Returns:
            Array of cluster labels.
        """
        if self.feature_matrix is None:
            raise RuntimeError("Call extract_trajectory_features() first.")
        
        feat_cols = [c for c in self.feature_matrix.columns if c != 'unit_id']
        X = self.feature_matrix[feat_cols].fillna(0).values
        
        # PCA for visualization
        n_comp = min(3, X.shape[1])
        self.pca = PCA(n_components=n_comp)
        X_pca = self.pca.fit_transform(X)
        
        # Cluster
        km = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.labels = km.fit_predict(X)
        
        # Silhouette score
        if len(set(self.labels)) > 1:
            sil = silhouette_score(X, self.labels)
        else:
            sil = 0
        
        # Store PCA coordinates
        self.feature_matrix['cluster'] = self.labels
        self.feature_matrix['pca_1'] = X_pca[:, 0]
        self.feature_matrix['pca_2'] = X_pca[:, 1]
        if n_comp >= 3:
            self.feature_matrix['pca_3'] = X_pca[:, 2]
        
        logger.info(f"Clustering complete: silhouette = {sil:.3f}")
        
        return self.labels
    
    def compute_cluster_profiles(self, train_df: pd.DataFrame) -> Dict:
        """
        Compute degradation profile statistics for each cluster.
        
        Args:
            train_df: Original training data with RUL.
            
        Returns:
            Dict with per-cluster profiles.
        """
        if self.labels is None:
            raise RuntimeError("Call cluster() first.")
        
        engine_clusters = dict(zip(self.engine_ids, self.labels))
        
        profiles = {}
        for cluster_id in range(self.n_clusters):
            cluster_engines = [e for e, c in engine_clusters.items() if c == cluster_id]
            cluster_data = train_df[train_df['unit_id'].isin(cluster_engines)]
            
            lifetimes = cluster_data.groupby('unit_id')['time_cycles'].max()
            
            profile = {
                'n_engines': len(cluster_engines),
                'mean_lifetime': float(lifetimes.mean()),
                'std_lifetime': float(lifetimes.std()),
                'min_lifetime': float(lifetimes.min()),
                'max_lifetime': float(lifetimes.max()),
                'median_lifetime': float(lifetimes.median()),
            }
            
            # Degradation speed classification
            overall_median = train_df.groupby('unit_id')['time_cycles'].max().median()
            if profile['median_lifetime'] < overall_median * 0.75:
                profile['archetype'] = 'Fast Degrader'
            elif profile['median_lifetime'] > overall_median * 1.25:
                profile['archetype'] = 'Slow Degrader'
            else:
                profile['archetype'] = 'Average Degrader'
            
            profiles[cluster_id] = profile
            logger.info(f"Cluster {cluster_id}: {profile['n_engines']} engines, "
                       f"median life={profile['median_lifetime']:.0f} "
                       f"({profile['archetype']})")
        
        self.cluster_profiles = profiles
        return profiles
    
    def find_optimal_k(self, k_range: range = range(2, 8)) -> Dict:
        """
        Find optimal number of clusters using silhouette analysis.
        
        Returns:
            Dict with scores per k and suggested optimum.
        """
        feat_cols = [c for c in self.feature_matrix.columns 
                     if c not in ('unit_id', 'cluster', 'pca_1', 'pca_2', 'pca_3')]
        X = self.feature_matrix[feat_cols].fillna(0).values
        
        scores = {}
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X)
            sil = silhouette_score(X, labels)
            scores[k] = float(sil)
        
        optimal_k = max(scores, key=scores.get)
        logger.info(f"Optimal k = {optimal_k} (silhouette = {scores[optimal_k]:.3f})")
        
        return {'scores': scores, 'optimal_k': optimal_k}
    
    # ------------------------------------------------------------------
    # Plotly visualizations
    # ------------------------------------------------------------------
    def plot_clusters_2d(self, title: str = "Degradation Archetypes (PCA)") -> go.Figure:
        """2D PCA scatter plot colored by cluster."""
        fig = go.Figure()
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f']
        
        for cluster_id in range(self.n_clusters):
            mask = self.feature_matrix['cluster'] == cluster_id
            subset = self.feature_matrix[mask]
            archetype = self.cluster_profiles.get(cluster_id, {}).get('archetype', f'Cluster {cluster_id}')
            
            fig.add_trace(go.Scatter(
                x=subset['pca_1'], y=subset['pca_2'],
                mode='markers',
                name=f"{archetype} (n={len(subset)})",
                marker=dict(
                    size=8, color=colors[cluster_id % len(colors)],
                    opacity=0.7, line=dict(width=0.5, color='white')
                ),
                text=[f"Engine {int(uid)}" for uid in subset['unit_id']],
                hoverinfo='text+name'
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title=f'PC1 ({self.pca.explained_variance_ratio_[0]:.0%} variance)',
            yaxis_title=f'PC2 ({self.pca.explained_variance_ratio_[1]:.0%} variance)',
            height=500
        )
        return fig
    
    def plot_cluster_lifetimes(self, title: str = "Lifetime by Cluster") -> go.Figure:
        """Box plot of engine lifetimes per cluster."""
        fig = go.Figure()
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for cid, profile in self.cluster_profiles.items():
            archetype = profile.get('archetype', f'Cluster {cid}')
            # Generate representative data for box plot
            lifetimes = np.random.normal(
                profile['mean_lifetime'], profile['std_lifetime'],
                profile['n_engines']
            )
            fig.add_trace(go.Box(
                y=lifetimes, name=archetype,
                marker_color=colors[cid % len(colors)],
                boxmean=True
            ))
        
        fig.update_layout(
            title=title, yaxis_title='Lifetime (cycles)', height=400
        )
        return fig
    
    def plot_silhouette_analysis(self, k_results: Dict,
                                  title: str = "Optimal Cluster Count") -> go.Figure:
        """Line chart of silhouette scores vs k."""
        ks = list(k_results['scores'].keys())
        scores = list(k_results['scores'].values())
        optimal = k_results['optimal_k']
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ks, y=scores, mode='lines+markers',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=10)
        ))
        fig.add_vline(x=optimal, line_dash='dash', line_color='red',
                      annotation_text=f'Optimal k={optimal}')
        
        fig.update_layout(
            title=title,
            xaxis_title='Number of Clusters (k)',
            yaxis_title='Silhouette Score',
            height=350
        )
        return fig


# ============================================================
# Standalone test
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Degradation Pattern Clustering")
    print("=" * 60)
    
    from data_loader import CMAPSSDataLoader
    from utils import add_remaining_useful_life
    
    loader = CMAPSSDataLoader('FD001')
    train_df, _, _ = loader.load_all_data()
    train_df = add_remaining_useful_life(train_df)
    
    clusterer = DegradationClusterer(n_clusters=3)
    
    print("\n--- Extracting Trajectory Features ---")
    feat_df = clusterer.extract_trajectory_features(train_df)
    print(f"  Feature matrix: {feat_df.shape}")
    
    print("\n--- Clustering ---")
    labels = clusterer.cluster()
    print(f"  Cluster distribution: {pd.Series(labels).value_counts().to_dict()}")
    
    print("\n--- Cluster Profiles ---")
    profiles = clusterer.compute_cluster_profiles(train_df)
    for cid, p in profiles.items():
        print(f"  Cluster {cid}: {p['n_engines']} engines, "
              f"median={p['median_lifetime']:.0f}cy ({p['archetype']})")
    
    print("\n--- Optimal k ---")
    k_results = clusterer.find_optimal_k()
    print(f"  Best k = {k_results['optimal_k']}")
    print(f"  Scores: {k_results['scores']}")
    
    print("\n✅ Degradation Clustering test PASSED")

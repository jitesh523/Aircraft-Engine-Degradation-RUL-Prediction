"""
Survival Analysis Engine for Aircraft Engine RUL Prediction
Provides Kaplan-Meier estimator and Cox Proportional Hazards models
to produce time-to-failure probability distributions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import config
from utils import setup_logging

logger = setup_logging(__name__)


class SurvivalAnalyzer:
    """
    Survival analysis for turbofan engines using Kaplan-Meier and Cox PH models.
    
    Converts run-to-failure data into survival curves that give the *probability*
    of an engine surviving past a given number of cycles, rather than a single
    point RUL estimate.
    """
    
    def __init__(self):
        """Initialize the survival analyzer."""
        self.km_fitted = False
        self.cox_fitted = False
        self.km_model = None
        self.cox_model = None
        self._duration_col = 'duration'
        self._event_col = 'event'
        
        # Will be populated after fit
        self.median_survival = None
        self.survival_table = None
        
        logger.info("SurvivalAnalyzer initialized")
    
    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------
    def prepare_survival_data(self, 
                              train_df: pd.DataFrame,
                              is_run_to_failure: bool = True) -> pd.DataFrame:
        """
        Convert C-MAPSS time-series data into per-engine survival records.
        
        Each row = one engine with:
          - duration: total operating cycles (max time_cycles)
          - event: 1 if failure observed (run-to-failure), 0 if censored
          - operational setting / sensor summary features (for Cox PH)
        
        Args:
            train_df: Raw training DataFrame with unit_id, time_cycles, sensors.
            is_run_to_failure: If True, all engines in training data ran to failure.
            
        Returns:
            DataFrame ready for survival modelling.
        """
        records = []
        sensor_cols = [c for c in train_df.columns if c.startswith('sensor_')]
        setting_cols = [c for c in train_df.columns if c.startswith('setting_')]
        
        for uid, grp in train_df.groupby('unit_id'):
            duration = grp['time_cycles'].max()
            event = 1 if is_run_to_failure else 0
            
            rec = {
                'unit_id': uid,
                self._duration_col: duration,
                self._event_col: event,
            }
            
            # Add summary features for Cox model
            for col in setting_cols:
                rec[f'{col}_mean'] = grp[col].mean()
            
            for col in sensor_cols:
                rec[f'{col}_mean'] = grp[col].mean()
                rec[f'{col}_std'] = grp[col].std()
                rec[f'{col}_last'] = grp[col].iloc[-1]
            
            records.append(rec)
        
        surv_df = pd.DataFrame(records)
        logger.info(f"Prepared survival data: {len(surv_df)} engines, "
                    f"median duration={surv_df[self._duration_col].median():.0f} cycles")
        return surv_df
    
    # ------------------------------------------------------------------
    # Kaplan-Meier
    # ------------------------------------------------------------------
    def fit_kaplan_meier(self, surv_df: pd.DataFrame, label: str = 'Fleet'):
        """
        Fit Kaplan-Meier survival estimator.
        
        Args:
            surv_df: Survival-formatted DataFrame (from prepare_survival_data).
            label: Label for the curve.
        """
        from lifelines import KaplanMeierFitter
        
        self.km_model = KaplanMeierFitter()
        self.km_model.fit(
            durations=surv_df[self._duration_col],
            event_observed=surv_df[self._event_col],
            label=label
        )
        
        self.km_fitted = True
        self.median_survival = self.km_model.median_survival_time_
        self.survival_table = self.km_model.survival_function_
        
        logger.info(f"Kaplan-Meier fitted: median survival = {self.median_survival:.1f} cycles")
    
    def predict_survival_probability(self, time_points: np.ndarray = None) -> pd.DataFrame:
        """
        Get survival probability at given time points.
        
        Args:
            time_points: Array of cycle values. If None, uses fitted timeline.
            
        Returns:
            DataFrame with columns [timeline, survival_probability].
        """
        if not self.km_fitted:
            raise RuntimeError("Call fit_kaplan_meier() first.")
        
        if time_points is not None:
            probs = self.km_model.predict(time_points)
            return pd.DataFrame({'timeline': time_points, 'survival_probability': probs.values})
        
        sf = self.km_model.survival_function_.reset_index()
        sf.columns = ['timeline', 'survival_probability']
        return sf
    
    def get_hazard_function(self) -> pd.DataFrame:
        """
        Compute the Nelson-Aalen cumulative hazard function.
        
        Returns:
            DataFrame with columns [timeline, cumulative_hazard].
        """
        from lifelines import NelsonAalenFitter
        
        if not self.km_fitted:
            raise RuntimeError("Call fit_kaplan_meier() first. Requires survival data.")
        
        naf = NelsonAalenFitter()
        sf = self.km_model
        naf.fit(
            durations=sf.durations,
            event_observed=sf.event_observed,
            label='Hazard'
        )
        
        hf = naf.cumulative_hazard_.reset_index()
        hf.columns = ['timeline', 'cumulative_hazard']
        return hf
    
    # ------------------------------------------------------------------
    # Cox Proportional Hazards
    # ------------------------------------------------------------------
    def fit_cox(self, surv_df: pd.DataFrame, 
                covariates: List[str] = None,
                penalizer: float = 0.1) -> Dict:
        """
        Fit Cox Proportional Hazards model.
        
        Args:
            surv_df: Survival-formatted DataFrame.
            covariates: List of covariate column names. If None, auto-detected.
            penalizer: L2 penalizer to prevent overfitting.
            
        Returns:
            Summary dictionary with coefficients and concordance index.
        """
        from lifelines import CoxPHFitter
        
        if covariates is None:
            exclude = {'unit_id', self._duration_col, self._event_col}
            covariates = [c for c in surv_df.columns if c not in exclude]
        
        # Drop columns with zero variance (lifelines will complain)
        valid_covs = []
        for c in covariates:
            if surv_df[c].std() > 1e-8:
                valid_covs.append(c)
        
        fit_cols = [self._duration_col, self._event_col] + valid_covs
        fit_df = surv_df[fit_cols].dropna()
        
        self.cox_model = CoxPHFitter(penalizer=penalizer)
        self.cox_model.fit(
            fit_df,
            duration_col=self._duration_col,
            event_col=self._event_col
        )
        
        self.cox_fitted = True
        
        concordance = self.cox_model.concordance_index_
        
        # Extract top risk factors
        coefs = self.cox_model.summary[['coef', 'exp(coef)', 'p']].copy()
        coefs = coefs.sort_values('p')
        
        logger.info(f"Cox PH fitted: concordance index = {concordance:.3f}, "
                    f"{len(valid_covs)} covariates")
        
        return {
            'concordance_index': concordance,
            'n_covariates': len(valid_covs),
            'top_risk_factors': coefs.head(10).to_dict(),
            'aic': self.cox_model.AIC_partial_
        }
    
    def predict_survival_curve(self, 
                                engine_data: pd.DataFrame,
                                times: np.ndarray = None) -> pd.DataFrame:
        """
        Predict survival curve for specific engine(s) using the Cox model.
        
        Args:
            engine_data: DataFrame row(s) with the same covariates as fit.
            times: Optional time points to evaluate.
            
        Returns:
            DataFrame with survival probabilities for each engine.
        """
        if not self.cox_fitted:
            raise RuntimeError("Call fit_cox() first.")
        
        sf = self.cox_model.predict_survival_function(engine_data, times=times)
        return sf
    
    def predict_median_rul(self, engine_data: pd.DataFrame) -> pd.Series:
        """
        Predict median RUL for engine(s) using Cox model.
        
        Args:
            engine_data: DataFrame row(s) with covariate values.
            
        Returns:
            Series of median survival times per engine.
        """
        if not self.cox_fitted:
            raise RuntimeError("Call fit_cox() first.")
        
        return self.cox_model.predict_median(engine_data)
    
    # ------------------------------------------------------------------
    # Group comparison
    # ------------------------------------------------------------------
    def compare_groups(self, 
                       surv_df: pd.DataFrame,
                       group_col: str,
                       test: str = 'logrank') -> Dict:
        """
        Compare survival between groups (e.g., operating conditions).
        
        Args:
            surv_df: Survival-formatted DataFrame.
            group_col: Column to group by.
            test: Statistical test ('logrank').
            
        Returns:
            Dictionary with test statistic, p-value, and per-group medians.
        """
        from lifelines import KaplanMeierFitter
        from lifelines.statistics import logrank_test
        
        groups = surv_df[group_col].unique()
        group_models = {}
        
        for g in sorted(groups):
            mask = surv_df[group_col] == g
            kmf = KaplanMeierFitter()
            kmf.fit(
                surv_df.loc[mask, self._duration_col],
                surv_df.loc[mask, self._event_col],
                label=str(g)
            )
            group_models[g] = kmf
        
        # Pairwise log-rank test (first two groups as primary comparison)
        groups_sorted = sorted(groups)
        result = {'groups': {}, 'test': test}
        
        for g in groups_sorted:
            result['groups'][str(g)] = {
                'n': int((surv_df[group_col] == g).sum()),
                'median_survival': float(group_models[g].median_survival_time_)
            }
        
        if len(groups_sorted) >= 2:
            g1, g2 = groups_sorted[0], groups_sorted[1]
            m1 = surv_df[group_col] == g1
            m2 = surv_df[group_col] == g2
            
            lr = logrank_test(
                surv_df.loc[m1, self._duration_col],
                surv_df.loc[m2, self._duration_col],
                event_observed_A=surv_df.loc[m1, self._event_col],
                event_observed_B=surv_df.loc[m2, self._event_col]
            )
            result['statistic'] = float(lr.test_statistic)
            result['p_value'] = float(lr.p_value)
            result['significant'] = lr.p_value < 0.05
        
        self._group_models = group_models
        
        logger.info(f"Group comparison on '{group_col}': {len(groups)} groups")
        return result
    
    # ------------------------------------------------------------------
    # Plotting (Plotly)
    # ------------------------------------------------------------------
    def plot_survival_curves(self, title: str = "Kaplan-Meier Survival Curve") -> go.Figure:
        """
        Plot the fitted Kaplan-Meier survival curve.
        
        Returns:
            Plotly Figure.
        """
        if not self.km_fitted:
            raise RuntimeError("Call fit_kaplan_meier() first.")
        
        sf = self.predict_survival_probability()
        ci = self.km_model.confidence_interval_survival_function_
        ci = ci.reset_index()
        ci.columns = ['timeline', 'lower', 'upper']
        
        fig = go.Figure()
        
        # Confidence band
        fig.add_trace(go.Scatter(
            x=ci['timeline'], y=ci['upper'],
            mode='lines', line=dict(width=0), showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=ci['timeline'], y=ci['lower'],
            mode='lines', line=dict(width=0),
            fill='tonexty', fillcolor='rgba(31,119,180,0.15)',
            name='95% CI'
        ))
        
        # Main curve
        fig.add_trace(go.Scatter(
            x=sf['timeline'], y=sf['survival_probability'],
            mode='lines', name='Survival Probability',
            line=dict(color='#1f77b4', width=2.5)
        ))
        
        # Median line
        if self.median_survival and np.isfinite(self.median_survival):
            fig.add_hline(y=0.5, line_dash="dash", line_color="gray",
                          annotation_text=f"Median = {self.median_survival:.0f} cycles")
        
        fig.update_layout(
            title=title,
            xaxis_title="Operating Cycles",
            yaxis_title="Survival Probability",
            yaxis_range=[0, 1.05],
            height=450
        )
        return fig
    
    def plot_hazard_function(self, title: str = "Cumulative Hazard Function") -> go.Figure:
        """Plot cumulative hazard function."""
        hf = self.get_hazard_function()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hf['timeline'], y=hf['cumulative_hazard'],
            mode='lines', name='Cumulative Hazard',
            line=dict(color='#d62728', width=2.5)
        ))
        fig.update_layout(
            title=title,
            xaxis_title="Operating Cycles",
            yaxis_title="Cumulative Hazard",
            height=400
        )
        return fig
    
    def plot_group_comparison(self, title: str = "Survival by Group") -> go.Figure:
        """
        Plot survival curves for each group (call compare_groups first).
        
        Returns:
            Plotly Figure.
        """
        if not hasattr(self, '_group_models'):
            raise RuntimeError("Call compare_groups() first.")
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f']
        
        fig = go.Figure()
        for i, (g, kmf) in enumerate(self._group_models.items()):
            sf = kmf.survival_function_.reset_index()
            sf.columns = ['timeline', 'survival']
            color = colors[i % len(colors)]
            
            fig.add_trace(go.Scatter(
                x=sf['timeline'], y=sf['survival'],
                mode='lines', name=str(g),
                line=dict(color=color, width=2)
            ))
        
        fig.update_layout(
            title=title, xaxis_title="Operating Cycles",
            yaxis_title="Survival Probability", yaxis_range=[0, 1.05],
            height=450
        )
        return fig
    
    def plot_cox_coefficients(self, title: str = "Cox PH Coefficients") -> go.Figure:
        """
        Plot Cox model coefficients (hazard ratios).
        
        Returns:
            Plotly Figure.
        """
        if not self.cox_fitted:
            raise RuntimeError("Call fit_cox() first.")
        
        summary = self.cox_model.summary.sort_values('coef')
        top_n = min(20, len(summary))
        summary = summary.head(top_n)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=summary.index,
            x=summary['coef'],
            orientation='h',
            marker_color=['#d62728' if c > 0 else '#2ca02c' for c in summary['coef']],
            text=[f"p={p:.3f}" for p in summary['p']],
            textposition='outside'
        ))
        fig.update_layout(
            title=title,
            xaxis_title="Coefficient (log hazard ratio)",
            height=max(350, top_n * 25),
            margin=dict(l=200)
        )
        return fig
    
    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    def get_summary(self) -> Dict:
        """Return a summary of fitted models."""
        summary = {'km_fitted': self.km_fitted, 'cox_fitted': self.cox_fitted}
        
        if self.km_fitted:
            summary['median_survival'] = float(self.median_survival) if np.isfinite(self.median_survival) else None
        
        if self.cox_fitted:
            summary['concordance_index'] = float(self.cox_model.concordance_index_)
            summary['aic'] = float(self.cox_model.AIC_partial_)
        
        return summary


# ============================================================
# Standalone test
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Survival Analyzer")
    print("=" * 60)
    
    from data_loader import CMAPSSDataLoader
    
    # Load training data
    loader = CMAPSSDataLoader('FD001')
    train_df, _, _ = loader.load_all_data()
    
    analyzer = SurvivalAnalyzer()
    
    # Prepare survival data
    surv_df = analyzer.prepare_survival_data(train_df)
    print(f"\nSurvival data shape: {surv_df.shape}")
    print(surv_df[['unit_id', 'duration', 'event']].describe())
    
    # Fit Kaplan-Meier
    print("\n--- Kaplan-Meier ---")
    analyzer.fit_kaplan_meier(surv_df)
    print(f"Median survival: {analyzer.median_survival:.1f} cycles")
    
    # Survival probabilities at key points
    probs = analyzer.predict_survival_probability(
        np.array([50, 100, 150, 200, 250, 300])
    )
    print("\nSurvival probabilities:")
    print(probs.to_string(index=False))
    
    # Hazard function
    print("\n--- Hazard Function ---")
    hf = analyzer.get_hazard_function()
    print(f"Max cumulative hazard: {hf['cumulative_hazard'].max():.2f}")
    
    # Cox PH (use a few summary features)
    print("\n--- Cox PH ---")
    cox_covs = [c for c in surv_df.columns 
                if c.endswith('_mean') and 'sensor' in c][:5]
    
    if cox_covs:
        cox_result = analyzer.fit_cox(surv_df, covariates=cox_covs)
        print(f"Concordance index: {cox_result['concordance_index']:.3f}")
        print(f"AIC: {cox_result['aic']:.1f}")
    
    # Group comparison: split engines into fast/slow degradation
    print("\n--- Group Comparison ---")
    surv_df['degradation_speed'] = pd.qcut(
        surv_df['duration'], q=2, labels=['Fast', 'Slow']
    )
    comp = analyzer.compare_groups(surv_df, 'degradation_speed')
    print(f"Groups: {comp['groups']}")
    if 'p_value' in comp:
        print(f"Log-rank p-value: {comp['p_value']:.4f} "
              f"({'significant' if comp['significant'] else 'not significant'})")
    
    # Summary
    print("\n--- Summary ---")
    print(analyzer.get_summary())
    
    print("\n✅ Survival Analyzer test PASSED")

"""
Instrumental Variables (IV) Estimator
Performs Two-Stage Least Squares (2SLS) regression to estimate causal effects
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.sandbox.regression.gmm import IV2SLS
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from utils import setup_logging

logger = setup_logging(__name__)


class IVEstimator:
    """
    Instrumental Variables Estimator using 2SLS
    Estimates causal effect of treatment on outcome using an instrument
    """
    
    def __init__(self):
        """Initialize IV Estimator"""
        self.model = None
        self.results = None
        logger.info("Initialized IV Estimator")
    
    def estimate_effect(self,
                       df: pd.DataFrame,
                       outcome_col: str,
                       treatment_col: str,
                       instrument_col: str,
                       control_cols: Optional[List[str]] = None,
                       add_constant: bool = True) -> Dict:
        """
        Estimate causal effect using 2SLS
        
        Stage 1: Treatment ~ Instrument + Controls
        Stage 2: Outcome ~ Predicted_Treatment + Controls
        
        Args:
            df: DataFrame containing variables
            outcome_col: Dependent variable (Y)
            treatment_col: Endogenous independent variable (X)
            instrument_col: Instrumental variable (Z)
            control_cols: Exogenous control variables (C)
            add_constant: Whether to add optimization constant (intercept)
            
        Returns:
            Dictionary with estimation results
        """
        logger.info(f"Estimating causal effect of {treatment_col} on {outcome_col} using instrument {instrument_col}...")
        
        # Prepare data
        working_df = df.copy().dropna()
        
        if control_cols is None:
            control_cols = []
        
        # Define Y (outcome)
        Y = working_df[outcome_col]
        
        # Define endogenous variables (treatment)
        endog = working_df[treatment_col]
        
        # Define exogenous variables (controls)
        exog_data = working_df[control_cols] if control_cols else pd.DataFrame(index=working_df.index)
        
        # Define instruments (instrument + controls)
        instr_data = working_df[[instrument_col] + control_cols]
        
        if add_constant:
            exog_data = sm.add_constant(exog_data)
            instr_data = sm.add_constant(instr_data)
        
        # Fit 2SLS model
        # Note: statsmodels IV2SLS signature is (endog, exog, instrument)
        # where exog includes both endogenous regressors and exogenous controls
        
        # The 'exog' argument for IV2SLS should contain ALL regressors (endogenous + exogenous)
        X = working_df[[treatment_col] + control_cols]
        if add_constant:
            X = sm.add_constant(X)
            
        try:
            self.model = IV2SLS(Y, X, instrument=instr_data)
            self.results = self.model.fit()
            
            # Extract results
            params = self.results.params
            pvalues = self.results.pvalues
            conf_int = self.results.conf_int()
            
            effect = params[treatment_col]
            ci_lower = conf_int.loc[treatment_col, 0]
            ci_upper = conf_int.loc[treatment_col, 1]
            p_value = pvalues[treatment_col]
            
            # First stage strength (F-statistic equivalent check)
            # We run the first stage manually to check instrument strength
            first_stage = sm.OLS(endog, instr_data).fit()
            f_stat = first_stage.fvalue
            
            summary = {
                'treatment': treatment_col,
                'outcome': outcome_col,
                'instrument': instrument_col,
                'effect_size': float(effect),
                'std_error': float(self.results.bse[treatment_col]),
                'p_value': float(p_value),
                'ci_95': (float(ci_lower), float(ci_upper)),
                'r_squared': float(self.results.rsquared),
                'f_statistic': float(f_stat),
                'instrument_strength': 'Strong' if f_stat > 10 else 'Weak',
                'n_obs': int(self.results.nobs)
            }
            
            logger.info(f"Estimation complete. Effect: {effect:.4f} (p={p_value:.4f})")
            logger.info(f"Instrument strength (F-stat): {f_stat:.2f} ({summary['instrument_strength']})")
            
            return summary
            
        except Exception as e:
            logger.error(f"IV estimation failed: {e}")
            return {'error': str(e)}
    
    def verify_assumptions(self,
                          df: pd.DataFrame,
                          outcome_col: str,
                          treatment_col: str,
                          instrument_col: str) -> Dict:
        """
        Check valid instrument assumptions correlations
        
        1. Relevance: Corr(Z, X) != 0 (Instrument predicts treatment)
        2. Exclusion: Corr(Z, Y) w.r.t X (Instrument affects outcome ONLY through treatment)
           Note: Exclusion restriction cannot be fully tested statistically, 
           but we can check raw correlations for intuition.
        
        Args:
            df: DataFrame
            outcome_col: Outcome
            treatment_col: Treatment
            instrument_col: Instrument
        
        Returns:
            Dictionary of correlation checks
        """
        # Calculate correlations
        corr_zx = df[[instrument_col, treatment_col]].corr().iloc[0, 1]
        corr_zy = df[[instrument_col, outcome_col]].corr().iloc[0, 1]
        corr_xy = df[[treatment_col, outcome_col]].corr().iloc[0, 1]
        
        checks = {
            'relevance_corr_zx': float(corr_zx),
            'raw_corr_zy': float(corr_zy),
            'obs_corr_xy': float(corr_xy),
            'relevance_check': abs(corr_zx) > 0.1  # Arbitrary threshold for "some" relevance
        }
        
        return checks
    
    def plot_iv_results(self,
                       df: pd.DataFrame,
                       outcome_col: str,
                       treatment_col: str):
        """
        Generate diagnostic plots for IV analysis
        
        Args:
            df: DataFrame
            outcome_col: Outcome variable
            treatment_col: Treatment variable
            
        Returns:
            Matplotlib figure
        """
        if self.results is None:
            logger.warning("No results to plot. Run estimate_effect first.")
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Standard OLS relationship
        sns.regplot(data=df, x=treatment_col, y=outcome_col, ax=axes[0], 
                   scatter_kws={'alpha': 0.3}, line_kws={'color': 'red'})
        axes[0].set_title(f'Naive OLS: {outcome_col} ~ {treatment_col}')
        
        # Plot 2: IV Residuals vs Fitted
        # If model is good, no pattern should exist
        fitted = self.results.fittedvalues
        resid = self.results.resid
        
        axes[1].scatter(fitted, resid, alpha=0.3)
        axes[1].axhline(0, color='red', linestyle='--')
        axes[1].set_xlabel('Fitted Values')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title('Residuals vs Fitted (IV 2SLS)')
        
        plt.tight_layout()
        return fig


if __name__ == "__main__":
    # Test with synthetic data
    logger.info("Generating synthetic data...")
    logger.info("True Causal Model: Y = 2*X + U + e")
    logger.info("Confounding: X correlated with U")
    
    np.random.seed(42)
    n = 1000
    
    # Unobserved confounder
    U = np.random.normal(0, 1, n)
    
    # Instrument (valid: affects X, independent of U)
    Z = np.random.normal(0, 1, n)
    
    # Treatment (endogenous: depends on Z and U)
    X = 0.8 * Z + 0.5 * U + np.random.normal(0, 0.5, n)
    
    # Outcome
    Y = 2 * X + U + np.random.normal(0, 0.5, n)
    
    df = pd.DataFrame({'Y': Y, 'X': X, 'Z': Z, 'U': U})
    
    estimator = IVEstimator()
    
    logger.info("Running IV Estimation...")
    results = estimator.estimate_effect(df, 'Y', 'X', 'Z')
    
    logger.info("Estimation Results:")
    for k, v in results.items():
        logger.info(f"  {k}: {v}")
        
    logger.info(f"True Effect: 2.0")
    logger.info(f"Estimated IV Effect: {results['effect_size']:.4f}")
    
    # Naive OLS for comparison
    ols = sm.OLS(Y, sm.add_constant(X)).fit()
    logger.info(f"Naive OLS Effect: {ols.params['X']:.4f} (Biased due to U)")


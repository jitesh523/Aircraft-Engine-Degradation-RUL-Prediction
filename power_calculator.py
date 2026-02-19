"""
Power Calculator
Calculates statistical power and sample size for A/B testing and experimentation
"""

import numpy as np
import pandas as pd
from statsmodels.stats.power import TTestIndPower
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from utils import setup_logging

logger = setup_logging(__name__)


class PowerCalculator:
    """
    Statistical Power Calculator
    Estimates required sample size or power for experiments
    """
    
    def __init__(self):
        """Initialize Power Calculator"""
        self.power_analysis = TTestIndPower()
        logger.info("Initialized Power Calculator")
    
    def calculate_sample_size(self,
                            effect_size: float,
                            alpha: float = 0.05,
                            power: float = 0.8,
                            ratio: float = 1.0) -> int:
        """
        Calculate required sample size per group
        
        Args:
            effect_size: Cohen's d (standardized mean difference)
            alpha: Significance level (Type I error rate)
            power: Statistical power (1 - Type II error rate)
            ratio: Ratio of sample sizes in group 2 to group 1
            
        Returns:
            Required sample size per group (rounded up)
        """
        if effect_size == 0:
            return float('inf')
            
        try:
            n = self.power_analysis.solve_power(
                effect_size=effect_size,
                power=power,
                alpha=alpha,
                ratio=ratio,
                alternative='two-sided'
            )
            return int(np.ceil(n))
        except Exception as e:
            logger.error(f"Sample size calculation failed: {e}")
            return 0
    
    def calculate_power(self,
                       nobs1: int,
                       effect_size: float,
                       alpha: float = 0.05,
                       ratio: float = 1.0) -> float:
        """
        Calculate statistical power given sample size
        
        Args:
            nobs1: Sample size of group 1
            effect_size: Cohen's d
            alpha: Significance level
            ratio: Ratio of sample sizes
            
        Returns:
            Statistical power (0.0 to 1.0)
        """
        try:
            power = self.power_analysis.solve_power(
                effect_size=effect_size,
                nobs1=nobs1,
                alpha=alpha,
                ratio=ratio,
                alternative='two-sided'
            )
            return float(power)
        except Exception as e:
            logger.error(f"Power calculation failed: {e}")
            return 0.0
            
    def calculate_minimum_detectable_effect(self,
                                          nobs1: int,
                                          power: float = 0.8,
                                          alpha: float = 0.05,
                                          ratio: float = 1.0) -> float:
        """
        Calculate Minimum Detectable Effect (MDE)
        
        Args:
            nobs1: Sample size of group 1
            power: Desired power
            alpha: Significance level
            ratio: Ratio of sample sizes
            
        Returns:
            Minimum detectable effect size (Cohen's d)
        """
        try:
            mde = self.power_analysis.solve_power(
                nobs1=nobs1,
                power=power,
                alpha=alpha,
                ratio=ratio,
                alternative='two-sided'
            )
            return float(mde)
        except Exception as e:
            logger.error(f"MDE calculation failed: {e}")
            return 0.0
    
    def plot_power_curve(self,
                        effect_sizes: List[float] = None,
                        n_range: Tuple[int, int] = (10, 500),
                        alpha: float = 0.05) -> plt.Figure:
        """
        Generate power curves for different effect sizes
        
        Args:
            effect_sizes: List of effect sizes to plot
            n_range: Range of sample sizes (min, max)
            alpha: Significance level
            
        Returns:
            Matplotlib figure
        """
        if effect_sizes is None:
            effect_sizes = [0.2, 0.5, 0.8]  # Small, Medium, Large
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sample_sizes = np.linspace(n_range[0], n_range[1], 100)
        
        for d in effect_sizes:
            powers = [self.calculate_power(n, d, alpha) for n in sample_sizes]
            ax.plot(sample_sizes, powers, label=f'Effect Size (d={d})', linewidth=2)
            
        ax.axhline(0.8, color='red', linestyle='--', alpha=0.5, label='Target Power (0.8)')
        
        ax.set_title(f'Power Curve (alpha={alpha})', fontsize=14)
        ax.set_xlabel('Sample Size (per group)', fontsize=12)
        ax.set_ylabel('Power (1 - beta)', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        
        return fig
        
    def estimate_cohens_d(self,
                         group1: np.ndarray,
                         group2: np.ndarray) -> float:
        """
        Estimate Cohen's d from two existing data groups
        
        Args:
            group1: Data for group 1
            group2: Data for group 2
            
        Returns:
            Cohen's d estimate
        """
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        mean_diff = np.mean(group1) - np.mean(group2)
        
        return abs(mean_diff / pooled_std)


if __name__ == "__main__":
    # Test calculator
    logger.info("--- Power Calculator Test ---")
    
    calc = PowerCalculator()
    
    # Scene 1: Calculate sample size for small effect
    d = 0.2  # Small effect
    alpha = 0.05
    power = 0.8
    
    n_req = calc.calculate_sample_size(d, alpha, power)
    logger.info(f"Required sample size for d={d}, alpha={alpha}, power={power}: {n_req} per group")
    
    # Scene 2: Calculate power for existing sample
    n = 50
    d = 0.5  # Medium effect
    
    achieved_power = calc.calculate_power(n, d, alpha)
    logger.info(f"Achieved power for n={n}, d={d}: {achieved_power:.4f}")
    
    # Scene 3: Calculate MDE
    mde = calc.calculate_minimum_detectable_effect(n, power=0.8)
    logger.info(f"MDE for n={n}, power={power}: {mde:.4f}")


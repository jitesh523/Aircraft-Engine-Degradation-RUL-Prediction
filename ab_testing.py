"""
A/B Testing Framework for Model Comparison
Statistical testing and champion/challenger framework
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
from datetime import datetime
from pathlib import Path


class ABTesting:
    """A/B testing framework for comparing model performance"""
    
    def __init__(self, model_a, model_b, name_a: str = "Model_A", name_b: str = "Model_B"):
        """
        Initialize A/B test
        
        Args:
            model_a: First model (champion)
            model_b: Second model (challenger)
            name_a: Name for model A
            name_b: Name for model B
        """
        self.model_a = model_a
        self.model_b = model_b
        self.name_a = name_a
        self.name_b = name_b
        self.test_results = {}
    
    def run_comparison(self, X_test, y_test, metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run comparison between two models
        
        Args:
            X_test: Test features
            y_test: Test targets
            metrics: List of metrics to compare (default: ['rmse', 'mae', 'r2'])
            
        Returns:
            Dictionary of comparison results
        """
        if metrics is None:
            metrics = ['rmse', 'mae', 'r2']
        
        # Get predictions
        pred_a = self.model_a.predict(X_test)
        pred_b = self.model_b.predict(X_test)
        
        # Calculate metrics
        results = {
            'timestamp': datetime.now().isoformat(),
            'n_samples': len(y_test),
            self.name_a: {},
            self.name_b: {},
            'comparison': {},
            'statistical_tests': {}
        }
        
        # Compute metrics for both models
        for metric in metrics:
            if metric == 'rmse':
                results[self.name_a]['rmse'] = float(np.sqrt(mean_squared_error(y_test, pred_a)))
                results[self.name_b]['rmse'] = float(np.sqrt(mean_squared_error(y_test, pred_b)))
            elif metric == 'mae':
                results[self.name_a]['mae'] = float(mean_absolute_error(y_test, pred_a))
                results[self.name_b]['mae'] = float(mean_absolute_error(y_test, pred_b))
            elif metric == 'r2':
                results[self.name_a]['r2'] = float(r2_score(y_test, pred_a))
                results[self.name_b]['r2'] = float(r2_score(y_test, pred_b))
        
        # Statistical significance testing
        errors_a = np.abs(y_test - pred_a)
        errors_b = np.abs(y_test - pred_b)
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(errors_a, errors_b)
        results['statistical_tests']['paired_t_test'] = {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': p_value < 0.05
        }
        
        # Wilcoxon signed-rank test (non-parametric alternative)
        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(errors_a, errors_b)
        results['statistical_tests']['wilcoxon_test'] = {
            'statistic': float(wilcoxon_stat),
            'p_value': float(wilcoxon_p),
            'significant': wilcoxon_p < 0.05
        }
        
        # Effect size (Cohen's d)
        cohens_d = (np.mean(errors_a) - np.mean(errors_b)) / np.sqrt((np.std(errors_a)**2 + np.std(errors_b)**2) / 2)
        results['statistical_tests']['cohens_d'] = float(cohens_d)
        
        # Confidence intervals for difference in means
        ci = stats.t.interval(0.95, 
                             len(errors_a) - 1,
                             loc=np.mean(errors_a - errors_b),
                             scale=stats.sem(errors_a - errors_b))
        results['statistical_tests']['ci_95_difference'] = {
            'lower': float(ci[0]),
            'upper': float(ci[1])
        }
        
        # Determine winner
        rmse_a = results[self.name_a].get('rmse', float('inf'))
        rmse_b = results[self.name_b].get('rmse', float('inf'))
        
        if results['statistical_tests']['paired_t_test']['significant']:
            winner = self.name_a if rmse_a < rmse_b else self.name_b
            improvement = abs((rmse_a - rmse_b) / max(rmse_a, rmse_b) * 100)
        else:
            winner = "No significant difference"
            improvement = 0.0
        
        results['comparison']['winner'] = winner
        results['comparison']['improvement_percent'] = float(improvement)
        results['comparison']['recommendation'] = self._get_recommendation(
            results['statistical_tests']['paired_t_test']['p_value'],
            improvement,
            winner
        )
        
        self.test_results = results
        return results
    
    def _get_recommendation(self, p_value: float, improvement: float, winner: str) -> str:
        """Generate deployment recommendation"""
        if p_value >= 0.05:
            return "No significant difference detected. Keep current model (Model A)."
        elif winner == self.name_b:
            if improvement > 10:
                return f"Model B shows significant improvement ({improvement:.1f}%). Strongly recommend deployment."
            elif improvement > 5:
                return f"Model B shows moderate improvement ({improvement:.1f}%). Recommend deployment with monitoring."
            else:
                return f"Model B shows small improvement ({improvement:.1f}%). Consider additional testing."
        else:
            return "Model A performs better. Keep current model."
    
    def power_analysis(self, 
                      effect_size: float,
                      alpha: float = 0.05,
                      power: float = 0.8) -> int:
        """
        Calculate required sample size for A/B test
        
        Args:
            effect_size: Expected effect size (Cohen's d)
            alpha: Significance level
            power: Statistical power
            
        Returns:
            Required sample size per group
        """
        from scipy.stats import norm
        
        z_alpha = norm.ppf(1 - alpha / 2)
        z_beta = norm.ppf(power)
        
        n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
        
        return int(np.ceil(n))
    
    def bootstrap_comparison(self, 
                            X_test, 
                            y_test,
                            n_bootstrap: int = 1000,
                            metric: str = 'rmse') -> Dict[str, Any]:
        """
        Bootstrap comparison for robust confidence intervals
        
        Args:
            X_test: Test features
            y_test: Test targets
            n_bootstrap: Number of bootstrap samples
            metric: Metric to compare
            
        Returns:
            Bootstrap results with confidence intervals
        """
        metrics_a = []
        metrics_b = []
        
        n_samples = len(y_test)
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot = X_test[indices]
            y_boot = y_test[indices]
            
            # Get predictions
            pred_a = self.model_a.predict(X_boot)
            pred_b = self.model_b.predict(X_boot)
            
            # Calculate metric
            if metric == 'rmse':
                metrics_a.append(np.sqrt(mean_squared_error(y_boot, pred_a)))
                metrics_b.append(np.sqrt(mean_squared_error(y_boot, pred_b)))
            elif metric == 'mae':
                metrics_a.append(mean_absolute_error(y_boot, pred_a))
                metrics_b.append(mean_absolute_error(y_boot, pred_b))
            elif metric == 'r2':
                metrics_a.append(r2_score(y_boot, pred_a))
                metrics_b.append(r2_score(y_boot, pred_b))
        
        metrics_a = np.array(metrics_a)
        metrics_b = np.array(metrics_b)
        differences = metrics_a - metrics_b
        
        return {
            'metric': metric,
            'n_bootstrap': n_bootstrap,
            self.name_a: {
                'mean': float(np.mean(metrics_a)),
                'ci_95': (float(np.percentile(metrics_a, 2.5)), 
                         float(np.percentile(metrics_a, 97.5)))
            },
            self.name_b: {
                'mean': float(np.mean(metrics_b)),
                'ci_95': (float(np.percentile(metrics_b, 2.5)), 
                         float(np.percentile(metrics_b, 97.5)))
            },
            'difference': {
                'mean': float(np.mean(differences)),
                'ci_95': (float(np.percentile(differences, 2.5)), 
                         float(np.percentile(differences, 97.5))),
                'p_value': float(np.mean(differences > 0))  # Empirical p-value
            }
        }
    
    def save_results(self, filepath: str):
        """Save test results to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.test_results, f, indent=2)
    
    def print_summary(self):
        """Print formatted summary of results"""
        if not self.test_results:
            print("No test results available. Run run_comparison() first.")
            return
        
        print("\n" + "="*60)
        print("A/B TEST RESULTS SUMMARY")
        print("="*60)
        
        print(f"\nModel A: {self.name_a}")
        print(f"Model B: {self.name_b}")
        print(f"Sample Size: {self.test_results['n_samples']}")
        
        print("\n" + "-"*60)
        print("PERFORMANCE METRICS")
        print("-"*60)
        
        for metric in ['rmse', 'mae', 'r2']:
            if metric in self.test_results[self.name_a]:
                val_a = self.test_results[self.name_a][metric]
                val_b = self.test_results[self.name_b][metric]
                print(f"\n{metric.upper()}:")
                print(f"  {self.name_a}: {val_a:.4f}")
                print(f"  {self.name_b}: {val_b:.4f}")
                diff = ((val_b - val_a) / val_a * 100) if metric != 'r2' else ((val_b - val_a) / abs(val_a) * 100)
                print(f"  Difference: {diff:+.2f}%")
        
        print("\n" + "-"*60)
        print("STATISTICAL TESTS")
        print("-"*60)
        
        st = self.test_results['statistical_tests']
        print(f"\nPaired T-Test:")
        print(f"  p-value: {st['paired_t_test']['p_value']:.4f}")
        print(f"  Significant: {st['paired_t_test']['significant']}")
        
        print(f"\nWilcoxon Test:")
        print(f"  p-value: {st['wilcoxon_test']['p_value']:.4f}")
        print(f"  Significant: {st['wilcoxon_test']['significant']}")
        
        print(f"\nEffect Size (Cohen's d): {st['cohens_d']:.4f}")
        
        ci = st['ci_95_difference']
        print(f"95% CI for difference: [{ci['lower']:.4f}, {ci['upper']:.4f}]")
        
        print("\n" + "-"*60)
        print("CONCLUSION")
        print("-"*60)
        
        comp = self.test_results['comparison']
        print(f"\nWinner: {comp['winner']}")
        if comp['improvement_percent'] > 0:
            print(f"Improvement: {comp['improvement_percent']:.2f}%")
        print(f"\nRecommendation: {comp['recommendation']}")
        
        print("\n" + "="*60 + "\n")


class ChampionChallengerFramework:
    """Framework for managing champion/challenger model deployment"""
    
    def __init__(self, champion_model, champion_name: str = "Champion"):
        """
        Initialize framework with champion model
        
        Args:
            champion_model: Current production model
            champion_name: Name of champion model
        """
        self.champion = champion_model
        self.champion_name = champion_name
        self.challengers = {}
        self.test_history = []
    
    def add_challenger(self, model, name: str):
        """Add a challenger model"""
        self.challengers[name] = model
        print(f"Added challenger: {name}")
    
    def test_challenger(self, 
                       challenger_name: str,
                       X_test, 
                       y_test,
                       auto_promote: bool = False) -> Dict[str, Any]:
        """
        Test a challenger against the champion
        
        Args:
            challenger_name: Name of challenger to test
            X_test: Test features
            y_test: Test targets
            auto_promote: Automatically promote if significantly better
            
        Returns:
            Test results
        """
        if challenger_name not in self.challengers:
            raise ValueError(f"Challenger {challenger_name} not found")
        
        challenger = self.challengers[challenger_name]
        
        ab_test = ABTesting(self.champion, challenger, 
                           self.champion_name, challenger_name)
        results = ab_test.run_comparison(X_test, y_test)
        ab_test.print_summary()
        
        # Store in history
        self.test_history.append({
            'timestamp': datetime.now().isoformat(),
            'challenger': challenger_name,
            'results': results
        })
        
        # Auto-promote if configured and significantly better
        if auto_promote and results['comparison']['winner'] == challenger_name:
            if results['statistical_tests']['paired_t_test']['significant']:
                self.promote_challenger(challenger_name)
        
        return results
    
    def promote_challenger(self, challenger_name: str):
        """Promote a challenger to champion"""
        if challenger_name not in self.challengers:
            raise ValueError(f"Challenger {challenger_name} not found")
        
        print(f"\nPromoting {challenger_name} to Champion!")
        print(f"Previous champion ({self.champion_name}) archived.")
        
        # Archive old champion
        old_champion_name = f"{self.champion_name}_archived_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.challengers[old_champion_name] = self.champion
        
        # Promote challenger
        self.champion = self.challengers[challenger_name]
        self.champion_name = challenger_name
        
        # Remove from challengers
        del self.challengers[challenger_name]
    
    def get_test_history(self) -> pd.DataFrame:
        """Get history of all challenger tests"""
        if not self.test_history:
            return pd.DataFrame()
        
        history_data = []
        for test in self.test_history:
            history_data.append({
                'timestamp': test['timestamp'],
                'challenger': test['challenger'],
                'winner': test['results']['comparison']['winner'],
                'improvement_pct': test['results']['comparison']['improvement_percent'],
                'p_value': test['results']['statistical_tests']['paired_t_test']['p_value']
            })
        
        return pd.DataFrame(history_data)

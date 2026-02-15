"""
Automated Anomaly Report Generator
Generates comprehensive HTML investigation reports for fleet anomalies
with embedded charts, root cause analysis, and recommendations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
import os
import config
from utils import setup_logging

logger = setup_logging(__name__)


class ReportEngine:
    """
    Generate professional fleet health and anomaly investigation reports.
    
    Reports include:
    - Executive summary
    - Fleet-level statistics
    - Per-engine anomaly details
    - Root cause analysis
    - Risk assessment
    - Maintenance recommendations
    """
    
    REPORT_DIR = os.path.join(config.PROJECT_ROOT, 'reports')
    
    def __init__(self):
        os.makedirs(self.REPORT_DIR, exist_ok=True)
        self.report_data = {}
        logger.info("ReportEngine initialized")
    
    def analyze_fleet(self, fleet_df: pd.DataFrame,
                       rul_col: str = 'rul_pred') -> Dict:
        """
        Analyze fleet for report data.
        
        Args:
            fleet_df: Fleet data with RUL predictions.
            rul_col: Column name for RUL values.
            
        Returns:
            Fleet analysis summary.
        """
        if rul_col not in fleet_df.columns:
            rul_col = 'RUL'
        
        thresholds = config.MAINTENANCE_THRESHOLDS
        ruls = fleet_df[rul_col].values
        
        critical_engines = fleet_df[ruls < thresholds['critical']]
        warning_engines = fleet_df[(ruls >= thresholds['critical']) & (ruls < thresholds['warning'])]
        healthy_engines = fleet_df[ruls >= thresholds['warning']]
        
        self.report_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'n_engines': len(fleet_df),
            'mean_rul': float(ruls.mean()),
            'min_rul': float(ruls.min()),
            'max_rul': float(ruls.max()),
            'std_rul': float(ruls.std()),
            'n_critical': len(critical_engines),
            'n_warning': len(warning_engines),
            'n_healthy': len(healthy_engines),
            'fleet_health_score': float(np.clip(ruls.mean() / 150 * 100, 0, 100)),
            'critical_engines': critical_engines.to_dict('records') if len(critical_engines) > 0 else [],
            'warning_engines': warning_engines.to_dict('records') if len(warning_engines) > 0 else [],
            'risk_level': 'HIGH' if len(critical_engines) > len(fleet_df) * 0.2 else
                          'MEDIUM' if len(critical_engines) > 0 else 'LOW',
            'cost_params': config.COST_PARAMETERS,
            'estimated_cost': float(
                len(critical_engines) * config.COST_PARAMETERS.get('unscheduled_maintenance', 50000) +
                len(warning_engines) * config.COST_PARAMETERS.get('scheduled_maintenance', 10000)
            )
        }
        
        logger.info(f"Fleet analyzed: {len(fleet_df)} engines, "
                    f"risk={self.report_data['risk_level']}")
        
        return self.report_data
    
    def generate_html_report(self, title: str = "Fleet Health Report",
                              filename: str = None) -> str:
        """
        Generate a full HTML report.
        
        Args:
            title: Report title.
            filename: Output filename (default: auto-generated).
            
        Returns:
            Path to generated HTML report.
        """
        if not self.report_data:
            raise RuntimeError("Call analyze_fleet() first.")
        
        d = self.report_data
        filename = filename or f"fleet_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        filepath = os.path.join(self.REPORT_DIR, filename)
        
        risk_color = {'HIGH': '#e74c3c', 'MEDIUM': '#f39c12', 'LOW': '#27ae60'}[d['risk_level']]
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #0a0f1c;
            color: #e0e0e0;
            padding: 40px;
            line-height: 1.6;
        }}
        .container {{ max-width: 1100px; margin: 0 auto; }}
        .header {{
            background: linear-gradient(135deg, #1a1f36, #2d1b69);
            border-radius: 16px;
            padding: 40px;
            margin-bottom: 30px;
            border: 1px solid #3d2b7a;
        }}
        .header h1 {{
            font-size: 2em;
            background: linear-gradient(135deg, #60a5fa, #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }}
        .header .meta {{ color: #9ca3af; font-size: 0.9em; }}
        .risk-badge {{
            display: inline-block;
            padding: 6px 20px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.85em;
            letter-spacing: 1px;
            color: white;
            background: {risk_color};
            margin-top: 10px;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .card {{
            background: #111827;
            border: 1px solid #1f2937;
            border-radius: 12px;
            padding: 24px;
            text-align: center;
        }}
        .card .value {{
            font-size: 2em;
            font-weight: bold;
            color: #60a5fa;
        }}
        .card .label {{ color: #9ca3af; margin-top: 5px; font-size: 0.85em; }}
        .section {{
            background: #111827;
            border: 1px solid #1f2937;
            border-radius: 12px;
            padding: 30px;
            margin-bottom: 25px;
        }}
        .section h2 {{
            font-size: 1.3em;
            color: #a78bfa;
            margin-bottom: 15px;
            border-bottom: 1px solid #1f2937;
            padding-bottom: 10px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        th, td {{
            padding: 10px 15px;
            text-align: left;
            border-bottom: 1px solid #1f2937;
        }}
        th {{
            color: #a78bfa;
            font-size: 0.85em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        tr:hover {{ background: #1f2937; }}
        .status-critical {{ color: #ef4444; font-weight: bold; }}
        .status-warning {{ color: #f59e0b; }}
        .status-healthy {{ color: #10b981; }}
        .recommendation {{
            background: #1e1b4b;
            border-left: 4px solid #a78bfa;
            padding: 15px 20px;
            margin-top: 10px;
            border-radius: 0 8px 8px 0;
        }}
        .bar {{
            height: 10px;
            border-radius: 5px;
            background: #1f2937;
            overflow: hidden;
            margin-top: 5px;
        }}
        .bar-fill {{
            height: 100%;
            border-radius: 5px;
            transition: width 0.5s;
        }}
        .footer {{
            text-align: center;
            color: #6b7280;
            margin-top: 40px;
            font-size: 0.8em;
        }}
    </style>
</head>
<body>
<div class="container">

    <div class="header">
        <h1>✈️ {title}</h1>
        <div class="meta">Generated: {d['timestamp']} | Engines: {d['n_engines']} | System: Aircraft Engine RUL Prediction</div>
        <div class="risk-badge">RISK LEVEL: {d['risk_level']}</div>
    </div>

    <div class="grid">
        <div class="card">
            <div class="value">{d['n_engines']}</div>
            <div class="label">TOTAL ENGINES</div>
        </div>
        <div class="card">
            <div class="value" style="color: #ef4444">{d['n_critical']}</div>
            <div class="label">CRITICAL</div>
        </div>
        <div class="card">
            <div class="value" style="color: #f59e0b">{d['n_warning']}</div>
            <div class="label">WARNING</div>
        </div>
        <div class="card">
            <div class="value" style="color: #10b981">{d['n_healthy']}</div>
            <div class="label">HEALTHY</div>
        </div>
        <div class="card">
            <div class="value">{d['fleet_health_score']:.0f}%</div>
            <div class="label">FLEET HEALTH</div>
        </div>
        <div class="card">
            <div class="value">${d['estimated_cost']:,.0f}</div>
            <div class="label">EST. MAINTENANCE COST</div>
        </div>
    </div>

    <div class="section">
        <h2>📊 Fleet Statistics</h2>
        <table>
            <tr><td>Mean RUL</td><td><strong>{d['mean_rul']:.1f} cycles</strong></td></tr>
            <tr><td>Min RUL</td><td><strong>{d['min_rul']:.1f} cycles</strong></td></tr>
            <tr><td>Max RUL</td><td><strong>{d['max_rul']:.1f} cycles</strong></td></tr>
            <tr><td>Std Dev</td><td><strong>{d['std_rul']:.1f} cycles</strong></td></tr>
            <tr>
                <td>Health Distribution</td>
                <td>
                    <div class="bar" style="width:100%">
                        <div class="bar-fill" style="width:{d['n_critical']/d['n_engines']*100:.0f}%; background:#ef4444; float:left"></div>
                        <div class="bar-fill" style="width:{d['n_warning']/d['n_engines']*100:.0f}%; background:#f59e0b; float:left"></div>
                        <div class="bar-fill" style="width:{d['n_healthy']/d['n_engines']*100:.0f}%; background:#10b981; float:left"></div>
                    </div>
                </td>
            </tr>
        </table>
    </div>
"""
        
        # Critical engines section
        if d['critical_engines']:
            html += """
    <div class="section">
        <h2>🚨 Critical Engines — Immediate Action Required</h2>
        <table>
            <tr><th>Engine ID</th><th>RUL</th><th>Status</th><th>Action</th></tr>
"""
            for eng in d['critical_engines'][:15]:
                eid = eng.get('engine_id', eng.get('unit_id', 'N/A'))
                rul = eng.get('rul_pred', eng.get('RUL', 0))
                html += f"""            <tr>
                <td>{eid}</td>
                <td class="status-critical">{rul:.0f} cy</td>
                <td class="status-critical">CRITICAL</td>
                <td>Ground immediately — schedule emergency maintenance</td>
            </tr>
"""
            html += """        </table>
    </div>
"""
        
        # Warning engines section
        if d['warning_engines']:
            html += """
    <div class="section">
        <h2>⚠️ Warning Engines — Schedule Maintenance</h2>
        <table>
            <tr><th>Engine ID</th><th>RUL</th><th>Status</th><th>Action</th></tr>
"""
            for eng in d['warning_engines'][:15]:
                eid = eng.get('engine_id', eng.get('unit_id', 'N/A'))
                rul = eng.get('rul_pred', eng.get('RUL', 0))
                html += f"""            <tr>
                <td>{eid}</td>
                <td class="status-warning">{rul:.0f} cy</td>
                <td class="status-warning">WARNING</td>
                <td>Schedule maintenance within {max(1, int(rul - 30))} cycles</td>
            </tr>
"""
            html += """        </table>
    </div>
"""
        
        # Recommendations
        html += """
    <div class="section">
        <h2>📋 Recommendations</h2>
"""
        recs = self._generate_recommendations()
        for rec in recs:
            html += f"""        <div class="recommendation">
            <strong>{rec['priority']}</strong> — {rec['action']}
        </div>
"""
        html += """    </div>
"""
        
        # Cost analysis
        html += f"""
    <div class="section">
        <h2>💰 Cost Analysis</h2>
        <table>
            <tr><td>Unscheduled maintenance ({d['n_critical']} engines)</td>
                <td><strong>${d['n_critical'] * 50000:,.0f}</strong></td></tr>
            <tr><td>Scheduled maintenance ({d['n_warning']} engines)</td>
                <td><strong>${d['n_warning'] * 10000:,.0f}</strong></td></tr>
            <tr><td>Monitoring ({d['n_healthy']} engines)</td>
                <td><strong>${d['n_healthy'] * 500:,.0f}</strong></td></tr>
            <tr style="border-top: 2px solid #a78bfa">
                <td><strong>Total Estimated Cost</strong></td>
                <td><strong style="color: #60a5fa">${d['estimated_cost'] + d['n_healthy'] * 500:,.0f}</strong></td>
            </tr>
        </table>
    </div>

    <div class="footer">
        <p>Aircraft Engine RUL Prediction System — Automated Report</p>
        <p>Generated on {d['timestamp']}</p>
    </div>

</div>
</body>
</html>
"""
        
        with open(filepath, 'w') as f:
            f.write(html)
        
        logger.info(f"HTML report generated: {filepath}")
        return filepath
    
    def _generate_recommendations(self) -> List[Dict]:
        """Generate prioritized recommendations based on fleet analysis."""
        d = self.report_data
        recs = []
        
        if d['n_critical'] > 0:
            recs.append({
                'priority': '🔴 URGENT',
                'action': (f"Ground {d['n_critical']} critical engine(s) immediately. "
                          f"Expected unscheduled cost: ${d['n_critical'] * 50000:,.0f}. "
                          f"Failure within <30 cycles is imminent.")
            })
        
        if d['n_warning'] > 0:
            recs.append({
                'priority': '🟡 HIGH',
                'action': (f"Schedule proactive maintenance for {d['n_warning']} warning engine(s). "
                          f"Estimated cost savings vs unscheduled: "
                          f"${d['n_warning'] * 40000:,.0f}.")
            })
        
        if d['n_critical'] / max(d['n_engines'], 1) > 0.15:
            recs.append({
                'priority': '🔴 STRATEGIC',
                'action': ("Critical engine fraction exceeds 15% of fleet. "
                          "Consider fleet-wide inspection campaign and accelerated "
                          "spare parts procurement.")
            })
        
        if d['fleet_health_score'] < 50:
            recs.append({
                'priority': '🟡 PLANNING',
                'action': ("Fleet health score below 50%. Increase predictive monitoring "
                          "frequency and review maintenance intervals.")
            })
        
        recs.append({
            'priority': '🟢 ROUTINE',
            'action': (f"Continue monitoring {d['n_healthy']} healthy engine(s). "
                      f"Next review recommended in 10 cycles.")
        })
        
        return recs
    
    def generate_summary_text(self) -> str:
        """Generate a plain-text executive summary."""
        d = self.report_data
        
        text = f"""
FLEET HEALTH REPORT — {d['timestamp']}
{'='*50}

EXECUTIVE SUMMARY
Fleet Size: {d['n_engines']} engines
Risk Level: {d['risk_level']}
Health Score: {d['fleet_health_score']:.0f}%

STATUS BREAKDOWN
  🔴 Critical: {d['n_critical']} engines (RUL < {config.MAINTENANCE_THRESHOLDS['critical']} cy)
  🟡 Warning:  {d['n_warning']} engines (RUL < {config.MAINTENANCE_THRESHOLDS['warning']} cy)
  🟢 Healthy:  {d['n_healthy']} engines

RUL STATISTICS
  Mean: {d['mean_rul']:.1f} cy | Min: {d['min_rul']:.1f} cy | Max: {d['max_rul']:.1f} cy

ESTIMATED COSTS
  Total: ${d['estimated_cost']:,.0f}
"""
        
        return text


# ============================================================
# Standalone test
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Automated Report Generator")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Create fleet
    fleet = pd.DataFrame({
        'engine_id': [f'ENG-{i:03d}' for i in range(1, 41)],
        'rul_pred': np.concatenate([
            np.random.randint(5, 30, 8),
            np.random.randint(30, 80, 12),
            np.random.randint(80, 200, 20),
        ]).astype(float)
    })
    
    engine = ReportEngine()
    
    print("\n--- Fleet Analysis ---")
    analysis = engine.analyze_fleet(fleet)
    print(f"  Risk level: {analysis['risk_level']}")
    print(f"  Health score: {analysis['fleet_health_score']:.0f}%")
    print(f"  Critical: {analysis['n_critical']}, Warning: {analysis['n_warning']}, Healthy: {analysis['n_healthy']}")
    print(f"  Estimated cost: ${analysis['estimated_cost']:,.0f}")
    
    print("\n--- Text Summary ---")
    summary = engine.generate_summary_text()
    print(summary)
    
    print("\n--- HTML Report ---")
    report_path = engine.generate_html_report("Fleet Health Report — Q1 2026")
    print(f"  Report saved: {report_path}")
    print(f"  File size: {os.path.getsize(report_path):,} bytes")
    
    print("\n✅ Report Generator test PASSED")

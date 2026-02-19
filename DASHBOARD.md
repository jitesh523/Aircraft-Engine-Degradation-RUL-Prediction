# Streamlit Dashboard

## Quick Start

```bash
# Using Makefile
make run-dashboard

# Or directly
streamlit run dashboard.py
```

Dashboard will be available at **http://localhost:8501**

## Dashboard Tabs (21 Total)

| # | Tab | Module | Description |
|---|-----|--------|-------------|
| 1 | 📊 Quick Prediction | `dashboard.py` | Manual sensor input with instant RUL gauge |
| 2 | 📁 Batch Upload | `dashboard.py` | CSV upload, multi-engine processing |
| 3 | 📈 Model Analytics | `model_comparison.py` | Compare LSTM, RF, XGBoost, Ensemble |
| 4 | 🔧 Maintenance Planner | `maintenance_planner.py` | Schedule optimization |
| 5 | 📅 Maintenance Scheduler | `maintenance_scheduler.py` | Fleet-wide scheduling |
| 6 | 🧪 A/B Testing | `ab_testing.py` | Champion-challenger framework |
| 7 | 🧬 Feature Engineering | `feature_engineer.py` | Automated feature creation |
| 8 | 🌀 Drift Monitor | `model_monitor.py` | PSI / KS drift detection |
| 9 | 📐 IV Estimator | `iv_estimator.py` | Causal effect estimation (2SLS) |
| 10 | ⚡ Power Calculator | `power_calculator.py` | Sample size & MDE analysis |
| 11 | 🔍 SHAP Explainer | `shap_explainer.py` | Feature importance explanations |
| 12 | 🤖 Auto Model Selector | `auto_model_selector.py` | Automated model selection |
| 13 | 🔮 Uncertainty | `uncertainty_quantifier.py` | MC Dropout confidence intervals |
| 14 | 🏗️ Digital Twin | `digital_twin.py` | Engine simulation |
| 15 | 🎲 What-If Simulator | `whatif_simulator.py` | Scenario analysis |
| 16 | 📡 Sensor Network | `sensor_network.py` | Sensor health monitoring |
| 17 | 🧬 Survival Analysis | `survival_analyzer.py` | Kaplan-Meier & Cox PH |
| 18 | ✈️ Operational Envelope | `envelope_analyzer.py` | Operating limits analysis |
| 19 | 🔗 Similarity Finder | `similarity_finder.py` | DTW-based engine matching |
| 20 | 💰 Cost Optimizer | `cost_optimizer.py` | Pareto multi-objective optimization |
| 21 | 🎯 Fleet Risk | `fleet_risk_simulator.py` | Monte Carlo risk assessment |

## Sidebar

- **Version footer**: v2.0, 48 modules, 21 tabs, Python version
- **Navigation**: Select any tab from the sidebar dropdown
- **AI Assistant**: LLM-powered maintenance chat (requires `GEMINI_API_KEY`)

## Deployment

### Local
```bash
streamlit run dashboard.py --server.port 8501
```

### Docker
```bash
docker build -t rul-dashboard -f Dockerfile.dashboard .
docker run -p 8501:8501 rul-dashboard
```

### Cloud
Push to GitHub → Deploy on [Streamlit Cloud](https://share.streamlit.io)

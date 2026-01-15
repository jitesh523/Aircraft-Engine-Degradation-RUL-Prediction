# Streamlit Dashboard

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run dashboard.py

# Dashboard will open at http://localhost:8501
```

## Features

### 📊 Quick Prediction
- Manual sensor input
- Instant RUL prediction with gauge visualization
- Health status (Critical/Warning/Healthy)
- Maintenance recommendations
- Uncertainty bounds (95% confidence intervals)

### 📁 Batch Upload
- Upload CSV files with sensor data
- Process multiple engines simultaneously
- Download results as CSV
- Interactive visualizations
- Summary statistics

### 📈 Model Analytics
- **Model Comparison**: Compare LSTM, RF, LR, and Ensemble
- **Feature Importance**: SHAP-based sensor rankings
- **Performance Metrics**: RMSE, MAE, R² tracking
- **Cost Impact Analysis**: Savings and availability improvements

## Dashboard Screenshots

### Main Interface
- Clean, modern design with Plotly interactive charts
- Sidebar configuration
- Real-time predictions
- Responsive layout

### Quick Prediction Mode
```
✈️ Aircraft Engine RUL Prediction
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 Quick RUL Prediction
┌─────────────────────────────┐
│  Sensor Readings           │
│  ├─ Sensor 4: 1589.7       │
│  ├─ Sensor 7: 554.36       │
│  ├─ Sensor 11: 47.47       │
│  └─ ...                    │
└─────────────────────────────┘

        RUL: 112 cycles
    [████████████░░░] Healthy
    
 Continue routine monitoring
```

### Batch Upload Mode
- Drag & drop CSV upload
- Real-time processing progress
- Interactive result table
- Bar charts by engine
- Export to CSV

### Model Analytics
- Side-by-side model comparison
- SHAP feature importance bar charts
- Ensemble weight pie chart
- Performance metrics dashboard

## Usage Examples

### 1. Quick Prediction
```python
1. Select "📊 Quick Prediction"
2. Enter sensor values
3. Check "Use Ensemble" for best accuracy
4. Click "🚀 Predict RUL"
5. View gauge chart and recommendations
```

### 2. Batch Processing
```csv
# Sample CSV format:
unit_id,time_cycle,sensor_2,sensor_3,sensor_4,...
1,1,642.3,1589.7,1400.6,...
1,2,642.5,1591.2,1402.1,...
...
```

Upload → Predict → Download Results

### 3. Model Insights
```python
1. Navigate to "📈 Model Analytics"
2. Compare model performance
3. View SHAP feature importance
4. Analyze cost savings
```

## Configuration

### Sidebar Options
- **Mode Selection**: Quick/Batch/Analytics
- **Model Status**: Real-time model loading status
- **Settings**: Ensemble toggle, uncertainty display

### Customization

Edit `dashboard.py` to customize:
```python
# Theme colors
st.set_page_config(
    page_title="Your Title",
    page_icon="🚁",
    layout="wide"
)

# Thresholds
CRITICAL_THRESHOLD = 30
WARNING_THRESHOLD = 80
```

## Deployment

### Local Development
```bash
streamlit run dashboard.py --server.port 8501
```

### Production Deployment

#### Streamlit Cloud (Easiest)
```bash
1. Push to GitHub
2. Go to share.streamlit.io
3. Connect repository
4. Deploy!
```

#### Docker
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t rul-dashboard .
docker run -p 8501:8501 rul-dashboard
```

#### Heroku
```bash
# Create Procfile
web: streamlit run dashboard.py --server.port=$PORT --server.address=0.0.0.0

# Deploy
heroku create rul-prediction-dashboard
git push heroku main
```

## Tips

**Performance**:
- Models are cached with `@st.cache_resource`
- First load takes ~5 seconds, then instant
- Use batch mode for multiple engines

**Data Format**:
- CSV must have `unit_id` column
- At least 30 timesteps per engine for LSTM
- Column names should match: `sensor_2`, `sensor_3`, etc.

**Troubleshooting**:
- Clear cache: Press 'C' in dashboard
- Reload models: Restart streamlit
- Check logs: Console shows detailed errors

## Advanced Features

### Custom Callbacks
Add real-time updates:
```python
@st.cache_data
def process_engine(data):
    # Your processing logic
    return results
```

### Session State
Preserve user data across interactions:
```python
if 'predictions' not in st.session_state:
    st.session_state.predictions = []
```

### Multi-page Apps
```bash
pages/
├── 1_📊_Predictions.py
├── 2_📈_Analytics.py
└── 3_⚙️_Settings.py
```

## Requirements

- Python 3.8+
- Streamlit 1.29.0+
- Plotly 5.18.0+
- All dependencies in requirements.txt

## Support

For issues:
1. Check console for error messages
2. Verify models are in `models/saved/`
3. Ensure CSV format matches schema
4. Review Streamlit docs: docs.streamlit.io

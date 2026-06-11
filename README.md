# Forest Fire Prediction & Monitoring System

![Python Version](https://img.shields.io/badge/Python-3.12%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Development-yellow)

An intelligent wildfire detection and monitoring system that processes satellite data from MODIS, applies machine learning clustering and predictions, and provides real-time alerts through an interactive web dashboard.

## Features

- **Satellite Data Processing**: Automatically processes MODIS fire detection data with spatial and temporal features
- **Intelligent Clustering**: Uses HDBSCAN for spatial anomaly detection and clustering of fire events
- **Dual ML Models**: 
  - Regression model for Fire Radiative Power (FRP) prediction
  - Classification model for fire type detection
- **Real-Time Inference**: Predicts fire characteristics on new satellite anomalies
- **Autonomous Monitoring**: Background agent system for continuous fire detection with configurable alerts
- **Interactive Dashboard**: Streamlit web interface for visualization and real-time monitoring
- **Discord Integration**: Automated alerts via Discord webhooks for high-severity fire events

## Quick Start

### Prerequisites

- Python 3.12 or higher
- Git

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/prediction-feux-forets.git
cd prediction-feux-forets
```

2. **Set up Python environment**:
```bash
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On macOS/Linux:
source .venv/bin/activate
```

3. **Install dependencies**:
```bash
pip install -e .
```

4. **Configure environment variables**:
```bash
cp .env.example .env
# Edit .env to add your Discord webhook URL if using the monitoring agent
```

5. **Verify installation**:
```bash
python main.py
```

## Usage

### Command-Line Pipeline

Run the complete data processing and model training pipeline:

```bash
python main.py
```

This executes 5 sequential phases:
1. **Preprocessing**: Normalizes features from MODIS satellite data
2. **Clustering**: Applies HDBSCAN for spatial fire event grouping
3. **Model Training**: Trains regression model for FRP prediction
4. **Classification Training**: Trains model for fire type detection
5. **Inference**: Simulates live prediction on a sample fire event

### Interactive Web Dashboard

Launch the Streamlit interface for visualization and monitoring:

```bash
streamlit run app.py
```

The dashboard provides:
- Historical fire event visualization on interactive maps
- Real-time fire severity classification
- FRP (Fire Radiative Power) predictions
- Cluster analysis and statistics
- Integration with the autonomous monitoring agent

### Autonomous Monitoring Agent

Run the background monitoring agent for continuous detection:

```bash
python -m src.agent
```

The agent:
- Continuously scans for new fire anomalies
- Classifies fire severity (Low/Medium/High/Critical)
- Sends Discord alerts for medium and above severity events
- Maintains an alert log for audit trails

## Project Structure

```
prediction-feux-forets/
├── app.py                          # Streamlit web interface
├── main.py                         # CLI pipeline orchestrator
├── pyproject.toml                  # Project dependencies and config
│
├── src/
│   ├── __init__.py
│   ├── config.py                   # Configuration and path management
│   ├── preprocess.py               # Data cleaning and feature engineering
│   ├── clustering.py               # HDBSCAN spatial clustering
│   ├── train_models.py             # Model training (regression & classification)
│   ├── inference.py                # Prediction on new data
│   └── agent.py                    # Autonomous monitoring agent
│
├── data/
│   ├── raw/                        # MODIS satellite fire data
│   │   └── MODIS_fires.csv
│   └── processed/                  # Processed data and alert logs
│       └── agent_alert_log.json
│
├── models/                         # Trained Keras models
│   ├── reg_frp_model.keras         # FRP regression model
│   ├── cls_type_model.keras        # Fire type classifier
│   ├── scaler.pkl                  # Feature scaler
│   └── spatial_clusterer.pkl       # HDBSCAN clusterer
│
└── .github/
    └── prompts/                    # Project generation prompts
```

## Data Format

The system expects MODIS fire data in CSV format with these columns:

| Column | Type | Description |
|--------|------|-------------|
| latitude | float | Fire event latitude |
| longitude | float | Fire event longitude |
| brightness | float | Fire radiative intensity |
| bright_t31 | float | Brightness temperature at 31 µm |
| frp | float | Fire Radiative Power (MW) - **target for regression** |
| type | int | Fire type classification - **target for classification** |
| acq_date | datetime | Acquisition date |
| acq_time | int | Acquisition time (HHMM format) |
| daynight | str | 'D' for day or 'N' for night |
| satellite | str | 'Terra' or 'Aqua' |
| confidence | int | Detection confidence (0-100) |
| scan / track | float | Pixel dimensions |

## Dependencies

Key libraries:
- **TensorFlow 2.21+**: Deep learning models
- **scikit-learn 1.8+**: ML preprocessing and utilities
- **HDBSCAN 0.8.43+**: Spatial clustering
- **Streamlit 1.58+**: Web interface
- **Pandas 3.0+**: Data manipulation
- **Plotly & Pydeck**: Interactive visualizations

See [pyproject.toml](pyproject.toml) for the complete dependency list.

## Configuration

Edit `src/config.py` to customize:
- Data paths (raw/processed directories)
- Model paths and names
- Feature scaling parameters

## Troubleshooting

**Missing MODIS data file?**
- Ensure `data/raw/MODIS_fires.csv` exists before running `main.py`

**Models not found on inference?**
- Run `python main.py` first to train and save models to `models/`

**Streamlit app not loading?**
- Verify all models are trained: `python main.py`
- Check that port 8501 is available

**Discord alerts not sending?**
- Add `DISCORD_WEBHOOK_URL` to your `.env` file
- Verify webhook URL is valid in Discord settings

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure:
- Code follows PEP 8 style guidelines
- New features include appropriate tests
- Documentation is updated
- Commit messages are clear and descriptive

## Support & Resources

- **Issues & Bug Reports**: [GitHub Issues](https://github.com/yourusername/prediction-feux-forets/issues)
- **Documentation**: See project README sections above
- **Data Source**: MODIS satellite fire detection products
- **Model Architecture**: Custom TensorFlow neural networks

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Maintainers

- **Project Lead**: [Your Name](https://github.com/yourusername)

## Acknowledgments

- MODIS satellite fire data from NASA Earth Observatories
- HDBSCAN clustering algorithm for spatial analysis
- Streamlit community for the excellent dashboard framework

---

**Last Updated**: June 2026 | **Version**: 0.1.0

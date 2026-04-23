# 🌿 Plant Health Analyzer

A multi-sensor plant health classification and correlation analysis web app built with Streamlit.

**Live App:** [Deploy link will appear here after deployment]

## Features

- **Data Entry** — Interactive table editor, upload Excel files, download results
- **Formula Classifier** — Scientific threshold voting system (NDVI, GNDVI, NDRE, SPAD, Temperature) compared against human labels
- **Correlation Analysis** — R² heatmap, scatter plots, index cross-correlations, SPAD validation
- **Spatial Map** — Geographic plant distribution with health score overlays
- **R² Rankings** — All pairwise correlations ranked and downloadable

## Sensors Supported

| Sensor | Device | Indices |
|--------|--------|---------|
| Spectral | Plant-o-Meter | NDVI, GNDVI, NDRE |
| Chlorophyll | SPAD-502 Meter | SPAD |
| Thermal | FLIR E5-XT | Canopy Temperature |

## Classification Thresholds

| Index | Healthy | Unhealthy | Dry | Reference |
|-------|---------|-----------|-----|-----------|
| NDVI | ≥ 0.60 | 0.40–0.59 | < 0.40 | Tucker (1979) |
| GNDVI | ≥ 0.55 | 0.35–0.54 | < 0.35 | Gitelson et al. (1996) |
| NDRE | ≥ 0.15 | 0.10–0.14 | < 0.10 | Barnes et al. (2000) |
| SPAD | ≥ 45 | 30–44 | < 30 | Peng et al. (1993) |
| Temperature | ≤ 32°C | 32–35°C | > 35°C | CWSI / Idso (1981) |

## Quick Start (Local)

```bash
git clone https://github.com/YOUR_USERNAME/plant-health-analyzer
cd plant-health-analyzer
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repo → set `app.py` as main file
4. Click **Deploy** — your app is live in ~2 minutes

## Input Data Format

Your Excel file should have these columns:

| Column | Description | Range |
|--------|-------------|-------|
| PLANTS | Plant name/ID | text |
| Latitude | GPS latitude | decimal degrees |
| Longtitude | GPS longitude | decimal degrees |
| NDVI | Vegetation index | 0.0 – 1.0 |
| GNDVI | Green NDVI | 0.0 – 1.0 |
| NDRE | Red edge index | 0.0 – 1.0 |
| SPAD Meter | Chlorophyll content | 0 – 100 |
| Temperature | Canopy temperature | °C |
| Personal_assumption_classification | Health label | Healthy / Unhealthy / Dry |

## Project Structure

```
plant-health-analyzer/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── .streamlit/
    └── config.toml         # Dark theme configuration
```

## Built With

- [Streamlit](https://streamlit.io) — Web framework
- [Matplotlib](https://matplotlib.org) — Visualization
- [SciPy](https://scipy.org) — Statistical analysis
- [scikit-learn](https://scikit-learn.org) — Data normalization
- [pandas](https://pandas.pydata.org) — Data handling

## Citation

If you use this tool in academic work, please cite the threshold references listed in the table above.

## License

MIT License — free to use, modify, and distribute.

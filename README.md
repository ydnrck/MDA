# Analyzing Cycling Traffic Resilience to Weather Conditions in Flanders

**Modern Data Analytics (GOZ39A) — Group 7**  
KU Leuven, MSc Statistics, Year 1

**Authors:** Aramayis Gharibyan · Diana Ashim · Jane Shadrina · Jules Vanderkerken

---

## Project overview

This project investigates the relationship between weather conditions and cycling traffic across Flanders, using automatic bicycle count data from the Agentschap Wegen en Verkeer (AWV) combined with meteorological observations from the Royal Meteorological Institute (RMI).

The analysis has two goals:
1. Estimate the effect of rainfall on cycling counts, controlling for other weather and temporal factors
2. Identify common weather-resilience profiles across cycling sites to support infrastructure planning

---

## Repository structure

```
├── MDA_EDA.ipynb                          # Exploratory data analysis
├── linear_regression_random_forest.ipynb  # Lasso regression and Random Forest models
├── clustering.ipynb                       # Per-site Ridge regression and K-means clustering
├── load_cycling_weather_data.py           # Helper functions to load the merged dataset
├── cycling_weather_full.parquet           # Merged AWV + RMI dataset 
└── README.md
```

---

## Data

### Sources
- **AWV fietstellingen** — automatic bicycle counts at fixed monitoring stations across Flanders, publicly available at [opendata.apps.mow.vlaanderen.be](https://opendata.apps.mow.vlaanderen.be/fietstellingen/index.html). Monthly CSV files covering August 2019 to March 2026.
- **RMI weather observations** — 10-minute weather readings from 17 stations across Belgium, available at [opendata.meteo.be](https://opendata.meteo.be). Variables include precipitation, temperature, wind speed, wind gusts, humidity, pressure, and solar radiation.

### Merged dataset
The two datasets were joined by matching each AWV cycling site to its nearest RMI weather station (spatial nearest-neighbour matching) and aligning timestamps to the nearest 15-minute interval. The result is `cycling_weather_full.parquet` (~271 MB, ~40 million rows). This file is not included in the repository due to size. To reproduce the analysis, download both source datasets and run the merging pipeline or contact the authors.

### Key variables used

| Variable | Source | Description |
|---|---|---|
| `count` | AWV | Cyclists counted in a 15-min interval |
| `ts` | AWV | Timestamp (UTC) |
| `site_id` | AWV | Unique counting station identifier |
| `precip_quantity` | RMI | Precipitation in mm |
| `temp_dry_shelter_avg` | RMI | Air temperature in °C |
| `wind_speed_10m` | RMI | Wind speed at 10m in m/s |
| `wind_gusts_speed` | RMI | Maximum wind gust in m/s |
| `humidity_rel_shelter_avg` | RMI | Relative humidity in % |
| `short_wave_from_sky_avg` | RMI | Solar radiation in W/m² |
| `nearest_station_km` | Derived | Distance from cycling site to matched weather station |

---

## Notebooks

### 1. `MDA_EDA.ipynb`
Exploratory data analysis on a reproducible 5% random sample of the merged dataset. Run this first.

Covers:
- Missing value analysis
- Temporal cycling patterns (hourly, daily, seasonal)
- Weather effects on cycling counts across rainfall intensity categories
- Spatial heterogeneity across sites
- Correlation matrix of weather variables

Key libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `geopandas`, `contextily`, `missingno`

### 2. `linear_regression_random_forest.ipynb`
Predictive modelling on a 50% random sample of the merged dataset.

Covers:
- Feature engineering (temporal dummies, log transformation of target)
- Lasso regression with 5-fold GridSearchCV over alpha values
- Random Forest with hyperparameter tuning on a 50k subsample, retrained on 200k rows
- Coefficient and feature importance analysis

Key libraries: `pandas`, `numpy`, `sklearn`, `matplotlib`

**Note on computational constraints:** The full dataset (~20M rows after cleaning) was too large to fit a Random Forest in available RAM. GridSearchCV was run on a 50,000-row subsample to find optimal hyperparameters (n_estimators=100, max_depth=20), then the final model was retrained on 200,000 rows and evaluated on the full test set (~4M rows).

### 3. `clustering.ipynb`
Site-level profiling and clustering. Run after the regression notebook.

Covers:
- Aggregation to daily site-level summaries
- Per-site Ridge regression to extract weather sensitivity coefficients for each of 148 sites
- Outlier removal using IsolationForest (8% contamination threshold)
- PCA on the 9 clustering features (first 5 components retain 81.9% of variance)
- K-means clustering on principal components, evaluated with elbow plots and silhouette scores
- Final 3-cluster solution with interpretation and visualisation

Key libraries: `pandas`, `numpy`, `sklearn`, `matplotlib`, `seaborn`

---

### Steps

1. Place `cycling_weather_full.parquet` in your project directory
2. Update `PROJECT_DIR` in each notebook to point to your project directory
3. Run notebooks in order: `MDA_EDA.ipynb` → `linear_regression_random_forest.ipynb` → `clustering.ipynb`

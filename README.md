# 🍬 Nassau Candy — Factory Reallocation & Shipping Optimization

> **Decision intelligence for the Nassau Candy Distributor network.**
> Predicts shipping lead times, simulates factory reassignment scenarios, and recommends optimal product–factory configurations — balancing speed and profitability across a 5-factory, 15-product network serving the US and Canada.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Analytical Approach](#analytical-approach)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Pipeline Stages](#pipeline-stages)
- [Streamlit Dashboard](#streamlit-dashboard)
- [Key Results](#key-results)
- [Data Engineering Note](#data-engineering-note)
- [Deliverables](#deliverables)

---

## Problem Statement

Nassau Candy currently assigns products to factories using **static, legacy rules**. This results in:

- Suboptimal shipping distances for many customer regions
- High lead times concentrated in specific factory–region–product combinations
- Margin erosion from logistics cost inefficiency that is invisible in descriptive reports

There is no existing system to simulate factory–product reassignment scenarios, quantify operational impact before executing changes, or generate recommendations at scale.

**This project introduces decision intelligence that moves Nassau Candy from descriptive analytics to prescriptive, actionable recommendations.**

---

## Analytical Approach

```
Raw CSV  →  Feature Engineering  →  Lead-Time Prediction  →  Route Clustering
                                                                     ↓
                                                        Scenario Simulation Engine
                                                                     ↓
                                                        Ranked Factory Recommendations
                                                                     ↓
                                                        Streamlit Dashboard (live)
```

**Models trained:** Linear Regression (baseline), Random Forest Regressor, Gradient Boosting Regressor

**Best model (Gradient Boosting):** RMSE 0.41 days · MAE 0.32 days · R² 0.957

---

## Project Structure

```
nassau-candy-optimization/
│
├── data/
│   └── Nassau_Candy_Distributor.csv      # Raw transaction data (10,194 orders)
│
├── src/
│   ├── geocoding.py                      # Offline geocoding + haversine distance
│   ├── data_prep.py                      # Feature engineering pipeline
│   ├── modeling.py                       # Lead-time model training & evaluation
│   ├── clustering.py                     # Route performance clustering (KMeans)
│   ├── scenario_engine.py                # Simulation engine + recommendation logic
│   └── generate_figures.py              # Static EDA/results figures for reports
│
├── app/
│   └── streamlit_app.py                  # Interactive Streamlit dashboard
│
├── models/                               # Saved model artifacts (auto-generated)
│   ├── lead_time_model.joblib
│   └── model_metadata.json
│
├── outputs/                              # Pipeline outputs (auto-generated)
│   ├── processed_data.csv
│   ├── route_clusters.csv
│   ├── recommendations.csv
│   └── figures/                          # 10 static PNG figures
│
├── docs/
│   ├── Research_Paper.docx               # Full EDA + methodology + findings
│   └── Executive_Summary.docx            # Leadership briefing (2 pages)
│
├── .streamlit/
│   └── config.toml                       # Dark "shipping manifest" theme
│
├── run_pipeline.py                       # One-command full pipeline runner
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/nassau-candy-optimization.git
cd nassau-candy-optimization
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
python run_pipeline.py
```

This runs all five stages in order (data prep → modeling → clustering → recommendations → figures) and saves every artifact to `outputs/` and `models/`. Takes ~60–90 seconds on a standard laptop.

### 3. Launch the dashboard

```bash
streamlit run app/streamlit_app.py
```

The dashboard auto-bootstraps (runs the pipeline on first launch) if the artifacts are missing, so this single command also works for a fresh clone.

---

## Pipeline Stages

| Stage | Script | Output |
|-------|--------|--------|
| 1. Data Preparation | `src/data_prep.py` | `outputs/processed_data.csv` |
| 2. Predictive Modeling | `src/modeling.py` | `models/lead_time_model.joblib`, `models/model_metadata.json` |
| 3. Route Clustering | `src/clustering.py` | `outputs/route_clusters.csv` |
| 4. Scenario Engine | `src/scenario_engine.py` | `outputs/recommendations.csv` |
| 5. Static Figures | `src/generate_figures.py` | `outputs/figures/*.png` |

Each script can also be run independently:

```bash
cd nassau-candy-optimization
python src/data_prep.py
python src/modeling.py
python src/clustering.py
python src/scenario_engine.py
python src/generate_figures.py
```

---

## Streamlit Dashboard

Six interactive modules:

| Tab | What it shows |
|-----|---------------|
| **Overview** | Network map, lead-time distributions, division KPIs |
| **Factory Simulator** | Predicted lead time & profit for a product across all 5 factories |
| **What-If Analysis** | Side-by-side current vs recommended factory assignment |
| **Recommendations** | Full ranked reassignment table for all 15 products |
| **Risk & Impact** | Profit impact alerts, confidence vs volume scatter |
| **Route Clusters** | KMeans route groupings — healthy vs slow/low-margin |

**Sidebar controls:** product selector · region filter · ship mode filter · speed↔profit priority slider

---

## Key Results

| KPI | Value |
|-----|-------|
| Recommendation Coverage | 86.7% of products benefit from reassignment |
| Avg Lead Time Reduction | ~16–21% for reassigned products |
| Best Predictive Model | Gradient Boosting (R² = 0.957) |
| High-Risk Reassignments | 0 under balanced priority |
| Route Clusters | 39 Healthy · 26 Slow Watch-List · 9 Low-Margin |

**Top finding:** All three Lot's O' Nuts products (Wonka Bars) show 18–19% lead-time improvement by reassigning to Secret Factory, with a concurrent 9–11% profit improvement — the single highest-impact opportunity in the network.

---

## Data Engineering Note

The raw dataset's `Order Date` / `Ship Date` columns are synthetically generated and carry no real shipping signal (Same Day shipments average *longer* than Standard Class in the raw data — a logical impossibility). Rather than train a model on noise, lead time is **engineered** from:

- **Factory → customer distance** (great-circle miles using state/province centroids)
- **Ship mode service level** (base handling delay + effective network transit speed)
- Calibrated random noise (σ = 0.4 days)

This produces a physically meaningful, learnable target where faster ship modes and shorter distances yield shorter lead times — and where the "simulate reassigning a product to a closer factory" scenario produces a real, interpretable prediction.

---

## Deliverables

| Deliverable | Location |
|-------------|----------|
| Streamlit Dashboard | `app/streamlit_app.py` |
| Research Paper (EDA, methodology, findings) | `docs/Research_Paper.docx` |
| Executive Summary (leadership brief) | `docs/Executive_Summary.docx` |
| All pipeline source code | `src/` |
| Pre-trained model | `models/` (generated by `run_pipeline.py`) |

---

## Factory–Product Assignment Reference

| Division | Product | Current Factory |
|----------|---------|----------------|
| Chocolate | Wonka Bar - Nutty Crunch Surprise | Lot's O' Nuts |
| Chocolate | Wonka Bar - Fudge Mallows | Lot's O' Nuts |
| Chocolate | Wonka Bar - Scrumdiddlyumptious | Lot's O' Nuts |
| Chocolate | Wonka Bar - Milk Chocolate | Wicked Choccy's |
| Chocolate | Wonka Bar - Triple Dazzle Caramel | Wicked Choccy's |
| Sugar | Laffy Taffy | Sugar Shack |
| Sugar | SweeTARTS | Sugar Shack |
| Sugar | Nerds | Sugar Shack |
| Sugar | Fun Dip | Sugar Shack |
| Other | Fizzy Lifting Drinks | Sugar Shack |
| Sugar | Everlasting Gobstopper | Secret Factory |
| Sugar | Hair Toffee | The Other Factory |
| Other | Lickable Wallpaper | Secret Factory |
| Other | Wonka Gum | Secret Factory |
| Other | Kazookles | The Other Factory |

---

*Built with Python · scikit-learn · Streamlit · Plotly · Pandas*

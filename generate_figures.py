"""
generate_figures.py
--------------------
Generates static PNG figures (for the research paper / executive
summary) from the processed dataset, model results, clusters, and
recommendations. Run after data_prep.py, modeling.py, clustering.py,
and scenario_engine.py have all produced their outputs.
"""

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

sns.set_theme(style="whitegrid", rc={
    "axes.edgecolor": "#3C2C22", "figure.facecolor": "white", "axes.facecolor": "white",
})
PALETTE = ["#B5651D", "#4FA8A5", "#D9534F", "#8C7A6B", "#6F8FAF", "#C97B4A"]
sns.set_palette(PALETTE)

FIG_DIR = "outputs/figures"


def fig_lead_time_dist(df):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(df["Lead Time Days"], bins=30, color=PALETTE[0], ax=ax)
    ax.set_title("Distribution of Engineered Lead Time (days)")
    ax.set_xlabel("Lead Time (days)")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/01_lead_time_distribution.png", dpi=150)
    plt.close(fig)


def fig_lead_time_by_shipmode(df):
    fig, ax = plt.subplots(figsize=(7, 4))
    order = df.groupby("Ship Mode")["Lead Time Days"].mean().sort_values().index
    sns.barplot(data=df, x="Ship Mode", y="Lead Time Days", order=order, ax=ax, errorbar=None)
    ax.set_title("Average Lead Time by Ship Mode")
    ax.set_ylabel("Avg Lead Time (days)")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/02_lead_time_by_shipmode.png", dpi=150)
    plt.close(fig)


def fig_distance_vs_leadtime(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    sample = df.sample(min(2000, len(df)), random_state=42)
    sns.scatterplot(data=sample, x="Distance Miles", y="Lead Time Days", hue="Ship Mode",
                     alpha=0.5, s=20, ax=ax)
    ax.set_title("Distance vs Lead Time, by Ship Mode")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/03_distance_vs_leadtime.png", dpi=150)
    plt.close(fig)


def fig_division_sales_profit(df):
    grp = df.groupby("Division").agg(Sales=("Sales", "sum"), Profit=("Adjusted Gross Profit", "sum")).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.barplot(data=grp, x="Division", y="Sales", ax=axes[0], color=PALETTE[0])
    axes[0].set_title("Total Sales by Division")
    sns.barplot(data=grp, x="Division", y="Profit", ax=axes[1], color=PALETTE[1])
    axes[1].set_title("Total Adjusted Gross Profit by Division")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/04_division_sales_profit.png", dpi=150)
    plt.close(fig)


def fig_region_volume(df):
    fig, ax = plt.subplots(figsize=(7, 4))
    order = df["Region"].value_counts().index
    sns.countplot(data=df, x="Region", order=order, ax=ax, color=PALETTE[2])
    ax.set_title("Order Volume by Region")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/05_region_volume.png", dpi=150)
    plt.close(fig)


def fig_model_comparison(metadata_path="models/model_metadata.json"):
    meta = json.load(open(metadata_path))
    results = meta["metrics"]
    names = list(results.keys())
    rmse = [results[n]["RMSE"] for n in names]
    r2 = [results[n]["R2"] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.barplot(x=names, y=rmse, ax=axes[0], color=PALETTE[3])
    axes[0].set_title("Model RMSE (lower is better)")
    axes[0].set_ylabel("RMSE (days)")
    axes[0].tick_params(axis="x", rotation=15)

    sns.barplot(x=names, y=r2, ax=axes[1], color=PALETTE[4])
    axes[1].set_title("Model R\u00b2 (higher is better)")
    axes[1].set_ylabel("R\u00b2")
    axes[1].tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/06_model_comparison.png", dpi=150)
    plt.close(fig)


def fig_feature_importance(model_path="models/lead_time_model.joblib", metadata_path="models/model_metadata.json"):
    model = joblib.load(model_path)
    meta = json.load(open(metadata_path))
    cols = meta["feature_cols"]
    if not hasattr(model, "feature_importances_"):
        return
    importances = pd.Series(model.feature_importances_, index=cols).sort_values(ascending=False).head(12)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(x=importances.values, y=importances.index, ax=ax, color=PALETTE[0])
    ax.set_title(f"Top Feature Importances ({meta['best_model']})")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/07_feature_importance.png", dpi=150)
    plt.close(fig)


def fig_route_clusters(clusters_path="outputs/route_clusters.csv"):
    clusters = pd.read_csv(clusters_path)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(
        data=clusters, x="Avg_Distance", y="Avg_Lead_Time", size="Order_Volume",
        hue="Cluster Label", sizes=(30, 300), alpha=0.7, ax=ax,
    )
    ax.set_title("Route Clusters: Distance vs Lead Time")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/08_route_clusters.png", dpi=150)
    plt.close(fig)


def fig_recommendations(recs_path="outputs/recommendations.csv"):
    recs = pd.read_csv(recs_path)
    reassign = recs[recs["Action"] == "Reassign"].sort_values("Lead Time Change %")
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = reassign["Risk Flag"].map({"Low Risk": PALETTE[1], "Caution": PALETTE[0], "High Risk": PALETTE[2]})
    ax.barh(reassign["Product Name"], reassign["Lead Time Change %"], color=colors)
    ax.set_title("Recommended Lead Time Improvement by Product")
    ax.set_xlabel("Lead Time Change %")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/09_recommendations.png", dpi=150)
    plt.close(fig)


def fig_factory_network():
    import sys
    sys.path.insert(0, "src")
    from geocoding import FACTORY_COORDINATES
    fig, ax = plt.subplots(figsize=(7, 5))
    names = list(FACTORY_COORDINATES.keys())
    lats = [FACTORY_COORDINATES[n][0] for n in names]
    lons = [FACTORY_COORDINATES[n][1] for n in names]
    ax.scatter(lons, lats, s=200, color=PALETTE[2], edgecolor="black", zorder=3)
    for n, lo, la in zip(names, lons, lats):
        ax.annotate(n, (lo, la), textcoords="offset points", xytext=(6, 6), fontsize=9)
    ax.set_title("Factory Network (approximate locations)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/10_factory_network.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    import os
    os.makedirs(FIG_DIR, exist_ok=True)
    df = pd.read_csv("outputs/processed_data.csv")
    fig_lead_time_dist(df)
    fig_lead_time_by_shipmode(df)
    fig_distance_vs_leadtime(df)
    fig_division_sales_profit(df)
    fig_region_volume(df)
    fig_model_comparison()
    fig_feature_importance()
    fig_route_clusters()
    fig_recommendations()
    fig_factory_network()
    print("All figures generated in outputs/figures/")

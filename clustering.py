"""
clustering.py
-------------
Route & Product Clustering stage.

Groups (Current Factory x Region x Ship Mode) "routes" by performance
similarity -- average lead time, average distance, average adjusted
profit margin, and order volume -- using KMeans. This surfaces:
    - Consistently slow routes (high lead time cluster)
    - Congested / low-margin region-product combinations

Output: outputs/route_clusters.csv, with a human-readable cluster label.
"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

N_CLUSTERS = 4
RANDOM_SEED = 42


def build_route_summary(df):
    grp = (
        df.groupby(["Current Factory", "Region", "Ship Mode"])
        .agg(
            Avg_Lead_Time=("Lead Time Days", "mean"),
            Avg_Distance=("Distance Miles", "mean"),
            Avg_Profit_Margin=("Profit Margin %", "mean"),
            Order_Volume=("Row ID", "count"),
            Total_Units=("Units", "sum"),
        )
        .reset_index()
    )
    return grp


def cluster_routes(route_summary, n_clusters=N_CLUSTERS, seed=RANDOM_SEED):
    features = ["Avg_Lead_Time", "Avg_Distance", "Avg_Profit_Margin", "Order_Volume"]
    X = route_summary[features]
    X_scaled = StandardScaler().fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    route_summary = route_summary.copy()
    route_summary["Cluster"] = km.fit_predict(X_scaled)

    # Label clusters by their characteristics relative to the overall mean
    cluster_stats = route_summary.groupby("Cluster")[features].mean()
    overall_lead = route_summary["Avg_Lead_Time"].mean()
    overall_margin = route_summary["Avg_Profit_Margin"].mean()

    def label_cluster(row):
        slow = row["Avg_Lead_Time"] > overall_lead
        low_margin = row["Avg_Profit_Margin"] < overall_margin
        if slow and low_margin:
            return "Consistently Slow & Low-Margin (High Risk)"
        elif slow:
            return "Consistently Slow (Watch List)"
        elif low_margin:
            return "Low-Margin but On-Time"
        else:
            return "Healthy Route"

    cluster_labels = cluster_stats.apply(label_cluster, axis=1)
    route_summary["Cluster Label"] = route_summary["Cluster"].map(cluster_labels)

    return route_summary, cluster_stats


def run(processed_path="outputs/processed_data.csv", out_path="outputs/route_clusters.csv"):
    df = pd.read_csv(processed_path)
    route_summary = build_route_summary(df)
    clustered, cluster_stats = cluster_routes(route_summary)
    clustered.to_csv(out_path, index=False)
    print(f"Clustered {len(clustered)} routes -> {out_path}")
    print("\nCluster profile (mean values):")
    print(cluster_stats.round(2))
    print("\nRoutes per label:")
    print(clustered["Cluster Label"].value_counts())
    return clustered


if __name__ == "__main__":
    run()

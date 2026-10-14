"""Customer analytics and clustering pipeline.

Run from the repository root:
    python -m src.eda_pipeline
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

KMEANS_FEATURES = [
    "recency", "frequency", "monetary", "interaction_score",
    "discount_sensitivity", "active_customer_score", "return_rate",
    "wishlist_conversion", "total_spend_usd_log", "avg_order_value_usd_log",
    "session_duration_minutes", "pages_viewed_before_purchase",
    "tenure", "reviews_given", "wishlist_items",
]


def load_data():
    customers = pd.read_csv(DATA_DIR / "customers.csv")
    orders = pd.read_csv(DATA_DIR / "orders.csv")
    return customers, orders


def build_customer_dataset(customers: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    customers = customers.copy()
    orders = orders.copy()

    orders["order_date"] = pd.to_datetime(orders["order_date"])
    customers["registration_date"] = pd.to_datetime(customers["registration_date"])

    orders_agg = orders.groupby("customer_id").agg(
        order_date=("order_date", "max"),
        category=("category", lambda x: ", ".join(x.dropna().unique())),
        quantity=("quantity", "sum"),
        discount_pct=("discount_pct", "mean"),
        shipping_fee_usd=("shipping_fee_usd", "sum"),
        tax_pct=("tax_pct", "mean"),
        total_amount_usd=("total_amount_usd", "sum"),
        delivery_days=("delivery_days", "mean"),
        returned=("returned", "sum"),
        customer_rating=("customer_rating", "mean"),
        session_duration_minutes=("session_duration_minutes", "sum"),
        pages_viewed_before_purchase=("pages_viewed_before_purchase", "sum"),
        is_repeat_customer=("is_repeat_customer", "max"),
    ).reset_index()

    df = customers.merge(orders_agg, on="customer_id", how="left")

    df["total_spend_usd_raw"] = df["total_spend_usd"]
    df["avg_order_value_usd_raw"] = df["avg_order_value_usd"]
    df["total_spend_usd_log"] = np.log1p(df["total_spend_usd"].clip(lower=0))
    df["avg_order_value_usd_log"] = np.log1p(df["avg_order_value_usd"].clip(lower=0))

    behavior_cols = [
        "quantity", "shipping_fee_usd", "total_amount_usd",
        "session_duration_minutes", "pages_viewed_before_purchase",
    ]
    for col in behavior_cols:
        df[col] = df[col].fillna(df[col].median())

    df["rating_missing"] = df["customer_rating"].isna().astype(int)

    for col in ["discount_pct", "tax_pct", "delivery_days", "customer_rating"]:
        df[col] = df[col].fillna(df[col].median())

    for col in ["returned", "is_repeat_customer"]:
        df[col] = df[col].fillna(df[col].mode().iloc[0])

    df["category"] = df["category"].replace("", np.nan).fillna("Missing_Order_Data")
    df["order_date"] = df["order_date"].fillna(df["registration_date"])

    analysis_date = orders["order_date"].max() + pd.Timedelta(days=1)
    df["tenure"] = (analysis_date - df["registration_date"]).dt.days
    df["recency"] = df["days_since_last_purchase"]
    df["frequency"] = df["total_orders"]
    df["monetary"] = df["total_spend_usd_raw"]

    df["age_segment"] = pd.cut(
        df["age"], bins=[17, 25, 35, 45, 60, 90],
        labels=["18-25", "26-35", "36-45", "46-60", "61-90"],
    )

    df["wishlist_conversion"] = df["total_orders"] / (df["wishlist_items"] + 1)
    df["discount_sensitivity"] = df["discount_pct"] * df["total_orders"]
    df["return_rate"] = df["returned"] / (df["total_orders"] + 1)
    df["interaction_score"] = (
        df["session_duration_minutes"] * df["pages_viewed_before_purchase"]
    )
    df["active_customer_score"] = (
        df["total_orders"] / (df["days_since_last_purchase"] + 1)
    )

    df["recency_score"] = pd.qcut(
        df["recency"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]
    )
    df["frequency_score"] = pd.qcut(
        df["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]
    )
    df["monetary_score"] = pd.qcut(
        df["monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]
    )
    df["rfm_score"] = (
        df["recency_score"].astype(str)
        + df["frequency_score"].astype(str)
        + df["monetary_score"].astype(str)
    )

    return df


def cluster_customers(df: pd.DataFrame, n_clusters: int = 5):
    features = df[KMEANS_FEATURES].replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df = df.copy()
    df["kmeans_cluster"] = kmeans.fit_predict(X_scaled) + 1

    # Alternative models are retained for methodological comparison.
    agg = AgglomerativeClustering(n_clusters=4)
    df["agg_cluster"] = agg.fit_predict(X_scaled) + 1

    gmm = GaussianMixture(n_components=4, random_state=42)
    df["gmm_cluster"] = gmm.fit_predict(X_scaled) + 1

    dbscan = DBSCAN(eps=2, min_samples=5)
    df["dbscan_cluster"] = dbscan.fit_predict(X_scaled)

    model_scores = {
        "KMeans": silhouette_score(X_scaled, df["kmeans_cluster"]),
        "Agglomerative": silhouette_score(X_scaled, df["agg_cluster"]),
        "GMM": silhouette_score(X_scaled, df["gmm_cluster"]),
    }
    mask = df["dbscan_cluster"] != -1
    if mask.sum() > 1 and df.loc[mask, "dbscan_cluster"].nunique() > 1:
        model_scores["DBSCAN"] = silhouette_score(
            X_scaled[mask], df.loc[mask, "dbscan_cluster"]
        )

    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(components, columns=["PC1", "PC2"])
    pca_df["kmeans_cluster"] = df["kmeans_cluster"].values

    return df, scaler, kmeans, model_scores, pca_df


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    customers, orders = load_data()
    customer_df = build_customer_dataset(customers, orders)
    clustered_df, _, _, scores, pca_df = cluster_customers(customer_df)

    clustered_df.to_csv(OUTPUT_DIR / "persona_ready_dataset.csv", index=False)
    pca_df.to_csv(OUTPUT_DIR / "pca_components.csv", index=False)
    pd.DataFrame(
        [{"model": name, "silhouette_score": score} for name, score in scores.items()]
    ).sort_values("silhouette_score", ascending=False).to_csv(
        OUTPUT_DIR / "clustering_model_comparison.csv", index=False
    )

    print(f"Customers: {len(customer_df):,}")
    print(f"Orders: {len(orders):,}")
    print("K-Means silhouette:", round(scores["KMeans"], 4))
    print(f"Saved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

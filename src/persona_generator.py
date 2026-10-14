"""Create business-friendly customer personas from clustering output."""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"

CLUSTER_NAMES = {
    1: "VIP Champions",
    2: "Low Value Customers",
    3: "Loyal Customers",
    4: "Highly Engaged Users",
    5: "Risky Returners",
}

PERSONA_PROFILES = {
    "VIP Champions": {
        "persona_name": "Premium Loyal Buyers",
        "shopping_style": "Premium and loyal shopper",
        "motivation": "Exclusivity, loyalty rewards, fast delivery",
        "pain_point": "Stock issues, delivery delays, and weak personalization",
        "marketing_strategy": "VIP access, premium bundles, loyalty rewards, and personalized offers",
    },
    "Low Value Customers": {
        "persona_name": "Occasional Deal Seekers",
        "shopping_style": "Low-frequency and price-sensitive shopper",
        "motivation": "Discounts, convenience, and low-risk offers",
        "pain_point": "Low engagement and weak purchase urgency",
        "marketing_strategy": "Coupons, free shipping, reactivation campaigns, and simple value-focused messaging",
    },
    "Loyal Customers": {
        "persona_name": "Reliable Repeat Buyers",
        "shopping_style": "Consistent and practical shopper",
        "motivation": "Trust, convenience, and reliable shopping experience",
        "pain_point": "Limited personalization and lack of fresh recommendations",
        "marketing_strategy": "Cross-sell campaigns, loyalty points, and personalized recommendations",
    },
    "Highly Engaged Users": {
        "persona_name": "Digital Window Shoppers",
        "shopping_style": "Highly active browsing-oriented shopper",
        "motivation": "Product discovery, trends, and personalized recommendations",
        "pain_point": "Too many choices and decision fatigue",
        "marketing_strategy": "Wishlist reminders, personalized recommendations, push notifications, and limited-time offers",
    },
    "Risky Returners": {
        "persona_name": "High-Risk Return Shoppers",
        "shopping_style": "Return-heavy and uncertain shopper",
        "motivation": "Discounts, flexibility, reassurance, and easy returns",
        "pain_point": "Product dissatisfaction, trust issues, and mismatch between expectation and product experience",
        "marketing_strategy": "Quality-focused messaging, size guides, review-based recommendations, and return-risk reduction",
    },
}


def get_mode(series):
    mode = series.mode()
    return mode.iloc[0] if not mode.empty else np.nan


def assign_risk_level(churn_rate, avg_return_rate, avg_recency):
    if churn_rate >= 0.20 or avg_return_rate >= 0.25 or avg_recency >= 180:
        return "High"
    if churn_rate >= 0.10 or avg_return_rate >= 0.15 or avg_recency >= 90:
        return "Medium"
    return "Low"


def create_personas(df):
    df = df.copy()
    df["segment_name"] = df["kmeans_cluster"].map(CLUSTER_NAMES)
    if df["segment_name"].isna().any():
        raise ValueError("Cluster mapping is incomplete.")

    summary = (
        df.groupby("segment_name")
        .agg(
            cluster_id=("kmeans_cluster", "first"),
            segment_size=("customer_id", "nunique"),
            avg_age=("age", "mean"),
            avg_monetary=("monetary", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_recency=("recency", "mean"),
            avg_interaction_score=("interaction_score", "mean"),
            avg_return_rate=("return_rate", "mean"),
            avg_wishlist_items=("wishlist_items", "mean"),
            avg_session_duration=("session_duration_minutes", "mean"),
            avg_pages_viewed=("pages_viewed_before_purchase", "mean"),
            churn_rate=("churned", "mean"),
            top_membership_tier=("membership_tier", get_mode),
            top_preferred_category=("preferred_category", get_mode),
            top_device=("preferred_device", get_mode),
            top_payment_method=("preferred_payment_method", get_mode),
            top_acquisition_channel=("acquisition_channel", get_mode),
        )
        .reset_index()
    )

    summary["segment_percentage"] = (
        summary["segment_size"] / summary["segment_size"].sum() * 100
    )

    rows = []
    for _, m in summary.iterrows():
        segment = m["segment_name"]
        profile = PERSONA_PROFILES[segment]
        risk = assign_risk_level(m["churn_rate"], m["avg_return_rate"], m["avg_recency"])

        rows.append({
            "cluster_id": int(m["cluster_id"]),
            "segment_name": segment,
            "persona_name": profile["persona_name"],
            "segment_size": int(m["segment_size"]),
            "segment_percentage": round(m["segment_percentage"], 2),
            "avg_age": round(m["avg_age"], 2),
            "avg_monetary": round(m["avg_monetary"], 2),
            "avg_frequency": round(m["avg_frequency"], 2),
            "avg_recency": round(m["avg_recency"], 2),
            "avg_interaction_score": round(m["avg_interaction_score"], 2),
            "avg_return_rate": round(m["avg_return_rate"], 4),
            "avg_wishlist_items": round(m["avg_wishlist_items"], 2),
            "avg_session_duration": round(m["avg_session_duration"], 2),
            "avg_pages_viewed": round(m["avg_pages_viewed"], 2),
            "churn_rate": round(m["churn_rate"], 4),
            "top_membership_tier": m["top_membership_tier"],
            "top_preferred_category": m["top_preferred_category"],
            "top_device": m["top_device"],
            "top_payment_method": m["top_payment_method"],
            "top_acquisition_channel": m["top_acquisition_channel"],
            "shopping_style": profile["shopping_style"],
            "motivation": profile["motivation"],
            "pain_point": profile["pain_point"],
            "marketing_strategy": profile["marketing_strategy"],
            "risk_level": risk,
            "business_summary": (
                f"{profile['persona_name']} represent {m['segment_percentage']:.1f}% of the "
                f"customer base, with average monetary value {m['avg_monetary']:.2f}, "
                f"frequency {m['avg_frequency']:.2f}, and recency {m['avg_recency']:.1f} days."
            ),
        })

    return pd.DataFrame(rows).sort_values("cluster_id").reset_index(drop=True)


def main():
    input_path = OUTPUT_DIR / "persona_ready_dataset.csv"
    if not input_path.exists():
        raise FileNotFoundError("Run `python -m src.eda_pipeline` first.")

    df = pd.read_csv(input_path)
    personas = create_personas(df)
    personas.to_csv(OUTPUT_DIR / "persona_output_df.csv", index=False)
    print(personas[["cluster_id", "segment_name", "segment_size", "segment_percentage"]].to_string(index=False))


if __name__ == "__main__":
    main()

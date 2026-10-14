"""Rule-based campaign simulation.

The response score is a heuristic business-fit score, not a calibrated probability.
LLM explanations are intentionally separated from the deterministic score.
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"

CAMPAIGNS = [
    {"campaign_id": 1, "campaign_name": "Premium Electronics Push", "campaign_description": "20% discount on premium electronics, free same-day shipping, and VIP-only early access.", "target_category": "Electronics", "discount_pct": 20, "free_shipping": 1, "vip_access": 1, "premium_positioning": 1, "reactivation_focus": 0, "loyalty_focus": 1},
    {"campaign_id": 2, "campaign_name": "Win-Back Free Shipping Offer", "campaign_description": "Free shipping and a limited-time coupon for inactive or at-risk customers.", "target_category": "All", "discount_pct": 10, "free_shipping": 1, "vip_access": 0, "premium_positioning": 0, "reactivation_focus": 1, "loyalty_focus": 0},
    {"campaign_id": 3, "campaign_name": "Wishlist Reminder Discount", "campaign_description": "Personalized reminder for wishlist items with a limited-time 10% discount.", "target_category": "All", "discount_pct": 10, "free_shipping": 0, "vip_access": 0, "premium_positioning": 0, "reactivation_focus": 0, "loyalty_focus": 0},
    {"campaign_id": 4, "campaign_name": "VIP Loyalty Rewards", "campaign_description": "Exclusive loyalty points and early access offers for loyal and high-value customers.", "target_category": "All", "discount_pct": 5, "free_shipping": 0, "vip_access": 1, "premium_positioning": 1, "reactivation_focus": 0, "loyalty_focus": 1},
    {"campaign_id": 5, "campaign_name": "Discount-Led Bundle Offer", "campaign_description": "Bundle deal with discount-focused messaging for price-sensitive customers.", "target_category": "All", "discount_pct": 20, "free_shipping": 0, "vip_access": 0, "premium_positioning": 0, "reactivation_focus": 0, "loyalty_focus": 0},
]


def normalize(value, low, high):
    if high == low:
        return 0.5
    return float(np.clip((value - low) / (high - low), 0, 1))


def calculate_campaign_score(persona, campaign, all_personas):
    score = 0.0
    reasons = []

    if campaign["target_category"] == "All" or campaign["target_category"] == persona["top_preferred_category"]:
        category_fit = 20
        reasons.append("Target category matches the persona profile")
    else:
        category_fit = 5
        reasons.append("Target category has limited alignment with the persona profile")
    score += category_fit

    monetary_score = normalize(
        persona["avg_monetary"], all_personas["avg_monetary"].min(), all_personas["avg_monetary"].max()
    )
    if campaign["premium_positioning"]:
        score += monetary_score * 15
    else:
        score += (1 - monetary_score) * 8 + 4

    frequency_score = normalize(
        persona["avg_frequency"], all_personas["avg_frequency"].min(), all_personas["avg_frequency"].max()
    )
    score += frequency_score * (15 if campaign["loyalty_focus"] else 8)

    engagement_score = normalize(
        persona["avg_interaction_score"], all_personas["avg_interaction_score"].min(), all_personas["avg_interaction_score"].max()
    )
    wishlist_score = normalize(
        persona["avg_wishlist_items"], all_personas["avg_wishlist_items"].min(), all_personas["avg_wishlist_items"].max()
    )
    if "Wishlist" in campaign["campaign_name"]:
        score += wishlist_score * 10 + engagement_score * 5
    else:
        score += engagement_score * 15

    discount_pct = campaign["discount_pct"]
    if discount_pct >= 20:
        discount_fit = 15 if persona["avg_monetary"] <= all_personas["avg_monetary"].median() else 9
    elif discount_pct >= 10:
        discount_fit = 10
    else:
        discount_fit = 5
    score += discount_fit

    churn = persona["churn_rate"]
    recency = persona["avg_recency"]
    if campaign["reactivation_focus"]:
        score += normalize(churn, 0, 0.30) * 5 + normalize(recency, 30, 180) * 5
    else:
        score += 2 if churn >= 0.20 or recency >= 180 else 8

    return_rate = persona["avg_return_rate"]
    penalty = -10 if return_rate >= 0.25 else -5 if return_rate >= 0.15 else 0
    score += penalty

    score = round(float(np.clip(score, 0, 100)), 1)
    likelihood = "High" if score >= 75 else "Medium" if score >= 50 else "Low"
    return score, likelihood, reasons


def run_simulation(personas):
    rows = []
    for _, persona in personas.iterrows():
        for campaign in CAMPAIGNS:
            score, likelihood, reasons = calculate_campaign_score(persona, campaign, personas)
            rows.append({
                "cluster_id": persona["cluster_id"],
                "segment_name": persona["segment_name"],
                "persona_name": persona["persona_name"],
                "campaign_id": campaign["campaign_id"],
                "campaign_name": campaign["campaign_name"],
                "response_score": score,
                "response_likelihood": likelihood,
                "segment_size": persona["segment_size"],
                "segment_percentage": persona["segment_percentage"],
                "risk_level": persona["risk_level"],
                "top_preferred_category": persona["top_preferred_category"],
                "avg_monetary": persona["avg_monetary"],
                "avg_frequency": persona["avg_frequency"],
                "avg_recency": persona["avg_recency"],
                "churn_rate": persona["churn_rate"],
                "avg_return_rate": persona["avg_return_rate"],
                "scoring_reasons": " | ".join(reasons),
            })

    return pd.DataFrame(rows)


def main():
    personas = pd.read_csv(OUTPUT_DIR / "persona_output_df.csv")
    results = run_simulation(personas)
    results.to_csv(OUTPUT_DIR / "campaign_simulation_results.csv", index=False)

    best = (
        results.sort_values(["persona_name", "response_score"], ascending=[True, False])
        .groupby("persona_name", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    best.to_csv(OUTPUT_DIR / "best_campaign_recommendations.csv", index=False)

    summary = (
        results.groupby("campaign_name")
        .agg(
            avg_response_score=("response_score", "mean"),
            high_response_segments=("response_likelihood", lambda x: (x == "High").sum()),
            medium_response_segments=("response_likelihood", lambda x: (x == "Medium").sum()),
            low_response_segments=("response_likelihood", lambda x: (x == "Low").sum()),
            total_reachable_customers=("segment_size", "sum"),
        )
        .reset_index()
    )
    summary.to_csv(OUTPUT_DIR / "campaign_performance_summary.csv", index=False)
    print(best[["persona_name", "campaign_name", "response_score", "response_likelihood"]].to_string(index=False))


if __name__ == "__main__":
    main()

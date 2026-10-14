import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

try:
    import google.generativeai as genai
except ImportError:
    genai = None

from src.campaign_simulation import CAMPAIGNS, calculate_campaign_score

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
load_dotenv(ROOT / ".env")

st.set_page_config(page_title="AI Marketing Campaign Simulator", layout="wide")

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
API_KEY = os.getenv("GOOGLE_API_KEY", "")

if genai and API_KEY:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel(MODEL_NAME)
else:
    model = None


@st.cache_data
def load_data():
    personas = pd.read_csv(OUTPUT_DIR / "persona_output_df.csv")
    recommendations = pd.read_csv(OUTPUT_DIR / "best_campaign_recommendations.csv")
    return personas, recommendations


try:
    persona_df, campaign_df = load_data()
except FileNotFoundError:
    st.error(
        "Required output files are missing. Run the pipeline first:\n\n"
        "`python -m src.eda_pipeline` → `python -m src.persona_generator` → "
        "`python -m src.campaign_simulation`"
    )
    st.stop()

with st.sidebar:
    st.title("Marketing Analytics")
    app_mode = st.radio("Mode", ["Campaign Scorer", "AI Strategy Agent"])
    selected_segment = st.selectbox("Target Segment", persona_df["segment_name"].unique())

p = persona_df[persona_df["segment_name"] == selected_segment].iloc[0]

st.title("AI Marketing Campaign Simulator")
st.caption("Customer segmentation, rule-based campaign simulation and GenAI-assisted strategy.")

if app_mode == "Campaign Scorer":
    st.subheader(f"Campaign comparison — {selected_segment}")

    campaign_names = [c["campaign_name"] for c in CAMPAIGNS]
    col_a, col_b = st.columns(2)

    with col_a:
        choice_a = st.selectbox("Campaign A", campaign_names, key="campaign_a")
    with col_b:
        choice_b = st.selectbox("Campaign B", campaign_names, index=1, key="campaign_b")

    camp_a = next(c for c in CAMPAIGNS if c["campaign_name"] == choice_a)
    camp_b = next(c for c in CAMPAIGNS if c["campaign_name"] == choice_b)

    score_a, likelihood_a, _ = calculate_campaign_score(p, camp_a, persona_df)
    score_b, likelihood_b, _ = calculate_campaign_score(p, camp_b, persona_df)

    st.dataframe(
        pd.DataFrame({
            "Metric": ["Response Score", "Likelihood"],
            "Campaign A": [score_a, likelihood_a],
            "Campaign B": [score_b, likelihood_b],
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.info("The response score is a heuristic business-fit score, not a calibrated purchase probability.")

    if st.button("Generate AI Explanation"):
        if not model:
            st.warning("Set GOOGLE_API_KEY in `.env` to enable Gemini explanations.")
        else:
            prompt = f"""
You are a senior marketing analytics assistant.
Explain the campaign comparison using only the provided data.

Segment: {selected_segment}
Motivation: {p['motivation']}
Pain point: {p['pain_point']}

Campaign A: {choice_a}
Score A: {score_a}/100 ({likelihood_a})

Campaign B: {choice_b}
Score B: {score_b}/100 ({likelihood_b})

Give a concise business explanation, key trade-offs, and suggested next test.
Do not describe the heuristic score as a real probability.
"""
            with st.spinner("Generating explanation..."):
                response = model.generate_content(prompt)
            st.markdown(response.text)

else:
    st.subheader(f"AI Strategy Agent — {selected_segment}")
    st.write(f"**Motivation:** {p['motivation']}")
    st.write(f"**Pain point:** {p['pain_point']}")
    st.write(f"**Current strategy:** {p['marketing_strategy']}")

    focus = st.text_input(
        "Optional focus",
        placeholder="e.g. increase loyalty or average order value",
    )

    if st.button("Generate Strategies"):
        if not model:
            st.warning("Set GOOGLE_API_KEY in `.env` to enable the AI Strategy Agent.")
        else:
            prompt = f"""
You are a senior e-commerce marketing strategy agent.

Segment: {selected_segment}
Motivation: {p['motivation']}
Pain point: {p['pain_point']}
Marketing strategy: {p['marketing_strategy']}
User focus: {focus or 'General growth and customer value'}

Generate 3 testable campaign ideas. For each include:
1. Name
2. Mechanism
3. Target behavior
4. Why it may work
5. A measurable KPI

Avoid claiming guaranteed outcomes.
"""
            with st.spinner("Generating strategies..."):
                response = model.generate_content(prompt)
            st.markdown(response.text)

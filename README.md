# End-to-End AI Marketing Campaign Simulator & Customer Analytics Platform

An end-to-end e-commerce customer analytics project that combines **RFM analysis, behavioral feature engineering, unsupervised customer segmentation, campaign simulation and GenAI-assisted marketing strategy**.

## Project flow

```text
Customer + Order Data
        ↓
Data Quality & Aggregation
        ↓
Feature Engineering + RFM
        ↓
Standardization
        ↓
K-Means Customer Segmentation
        ↓
Persona Profiles
        ↓
Rule-Based Campaign Simulation
        ↓
Streamlit Decision-Support App
        ↓
Optional Gemini AI Explanations & Strategy Generation
```

## Key capabilities

- Customer-level order aggregation
- RFM-style recency, frequency and monetary features
- Behavioral features such as interaction score, return rate, discount sensitivity and wishlist conversion
- K-Means customer segmentation with PCA visualization
- Method comparison using silhouette score
- Business-oriented customer personas
- Rule-based campaign response simulation
- Persona-level campaign recommendations
- Streamlit interactive dashboard
- Optional Gemini-powered campaign explanations and strategy generation

## Important modeling note

The campaign `response_score` is a **heuristic business-fit score** based on customer/persona characteristics and campaign rules. It is **not a calibrated probability of purchase** and should not be interpreted as an observed campaign response.

The five K-Means segments are retained for business interpretability. The repository also exports silhouette scores for alternative clustering approaches.

## Repository structure

```text
.
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
├── data/
│   ├── customers.csv
│   ├── orders.csv
│   └── README.md
├── outputs/
│   ├── persona_ready_dataset.csv
│   ├── persona_output_df.csv
│   ├── campaign_simulation_results.csv
│   ├── best_campaign_recommendations.csv
│   ├── campaign_performance_summary.csv
│   ├── clustering_model_comparison.csv
│   └── pca_components.csv
└── src/
    ├── eda_pipeline.py
    ├── persona_generator.py
    └── campaign_simulation.py
```

## Setup

```bash
git clone https://github.com/suleaydn6/AI-Powered-Marketing-Analytics.git
cd AI-Powered-Marketing-Analytics

python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt

```

## Run the pipeline

```bash
python -m src.eda_pipeline
python -m src.persona_generator
python -m src.campaign_simulation
```

Then launch the dashboard:

```bash
streamlit run app.py
```

## Gemini configuration

Copy `.env.example` to `.env` and add your own key:

```text
GOOGLE_API_KEY=your_key
GEMINI_MODEL=gemini-2.5-flash
```

`.env` is excluded from Git through `.gitignore`.

## Analytical methodology

### Feature engineering

The pipeline derives:

- `recency`
- `frequency`
- `monetary`
- `interaction_score`
- `discount_sensitivity`
- `active_customer_score`
- `return_rate`
- `wishlist_conversion`
- log-transformed spending variables
- tenure
- RFM quintile scores

### Segmentation

K-Means is used as the primary segmentation method. Features are standardized before clustering.

For methodological comparison, the pipeline also evaluates:

- Agglomerative Clustering
- Gaussian Mixture Model
- DBSCAN where a valid silhouette calculation is possible

PCA is used for two-dimensional visualization.

### Persona layer

Cluster-level metrics are translated into business-oriented personas, including:

- segment size and share
- monetary value
- purchase frequency
- recency
- engagement
- return risk
- churn rate
- preferred category/device
- marketing motivation and pain points

### Campaign simulation

Campaign fit is estimated using explicit business rules covering:

- category alignment
- spending power
- purchase frequency
- engagement
- wishlist behavior
- discount sensitivity
- reactivation signals
- return risk

The resulting score supports **scenario comparison**, not causal campaign measurement.

## Future improvements

- Train a supervised campaign-response model using historical treatment/outcome data
- Add A/B test and uplift modeling
- Calibrate response probabilities
- Automate cluster labeling from cluster statistics
- Add model/version tracking and experiment logging
- Introduce a formal evaluation dataset for GenAI outputs

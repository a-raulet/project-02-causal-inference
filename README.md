# Causal Inference Platform

A causal inference platform for A/B testing analysis using Bayesian methods and machine learning. Analyzes the Hillstrom email marketing dataset to estimate treatment effects and recommend optimal targeting strategies.

The Streamlit dashboard is available [here](https://causal-inference-platform.streamlit.app/).

## Overview

This project demonstrates a complete causal inference pipeline:

1. **Exploratory Analysis** - Covariate balance checks and heterogeneity exploration
2. **Bayesian A/B Testing** - Posterior estimation with PyMC (Beta-Binomial models)
3. **Causal ML** - Conditional Average Treatment Effects (CATE) with X-Learner
4. **Production API** - Real-time treatment recommendations via FastAPI
5. **Interactive Dashboard** - Results visualization with Streamlit

## Dataset

**Hillstrom Email Marketing Dataset**
- **64,000 customers** from an e-commerce company
- **3 treatment groups**: Mens E-Mail, Womens E-Mail, No E-Mail (control)
- **Outcomes**: Conversion (binary), Visit (binary), Spend (continuous)
- **Covariates**: Recency, history, channel, zip_code, newbie status

Source: [Kevin Hillstrom's MineThatData](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html)

## Quick Start

```bash
# Install dependencies
poetry install

# Run notebooks in order
poetry run jupyter lab

# Start the API
poetry run uvicorn src.api.main:app --reload

# Launch Streamlit dashboard
cd streamlit && poetry run streamlit run app.py
```

## Project Structure

```
project-02-causal-inference/
├── data/
│   ├── raw/hillstrom.csv           # Original dataset
│   └── processed/                   # Analysis outputs (JSON, CSV)
├── notebooks/
│   ├── 01_exploration.ipynb         # EDA, balance checks
│   ├── 02_bayesian_ab_testing.ipynb # Bayesian posterior estimation
│   └── 03_causal_ml.ipynb           # CATE estimation, policy learning
├── models/
│   ├── cate_model_mens.pkl          # Trained X-Learner (Mens)
│   └── cate_model_womens.pkl        # Trained X-Learner (Womens)
├── src/api/
│   ├── main.py                      # FastAPI endpoints
│   ├── schemas.py                   # Pydantic models
│   ├── models.py                    # Model loading
│   └── preprocessing.py             # Feature engineering
├── streamlit/
│   ├── app.py                       # Main dashboard
│   └── pages/                       # Multi-page app
│       ├── 1_Exploration.py
│       ├── 2_Bayesian_AB.py
│       ├── 3_CausalML.py
│       └── 4_Simulateur.py
├── reports/figures/                 # Generated visualizations
└── tests/
```

## Methodology

### Bayesian A/B Testing

```
Prior:      Beta(1, 1)  # Uninformative
Likelihood: Binomial(n, p)
Posterior:  Beta(1 + conversions, 1 + non-conversions)
```

Key metrics computed:
- **P(Treatment > Control)** - Probability of superiority
- **Lift** - Relative improvement vs control
- **95% HDI** - Highest Density Interval
- **Expected Loss** - Risk of choosing wrong treatment

### Causal ML (CATE Estimation)

Uses the **X-Learner** algorithm:
1. Train outcome models on treatment and control separately
2. Estimate individual treatment effects (ITE)
3. Train CATE models to predict ITE from covariates
4. Use SHAP for interpretability

## API Usage

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "recency": 5,
    "history": 150.0,
    "mens": 1,
    "womens": 0,
    "newbie": 0,
    "channel": "Web",
    "zip_code": "Urban"
  }'

# Response
{
  "cate_mens_email": 0.0234,
  "cate_womens_email": 0.0089,
  "optimal_treatment": "Mens E-Mail",
  "lift_vs_no_email": 0.0234
}
```

## Key Results

| Treatment | Conversion Rate | Lift vs Control | P(Best) |
|-----------|-----------------|-----------------|---------|
| No E-Mail (Control) | ~5.0% | - | - |
| Mens E-Mail | ~5.8% | +16% | 0.92 |
| Womens E-Mail | ~5.4% | +8% | 0.78 |

## Tech Stack

| Component | Technology |
|-----------|------------|
| Bayesian Modeling | PyMC, ArviZ |
| Causal ML | CausalML, SHAP |
| API | FastAPI, Pydantic |
| Dashboard | Streamlit |
| Data | pandas, scikit-learn |

## Author

Arnaud Raulet

---

*Part of a data science portfolio focused on causal inference and marketing analytics.*

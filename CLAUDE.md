# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A causal inference platform for A/B testing analysis using Bayesian methods. The project analyzes the Hillstrom email marketing dataset (64,000 customers) to estimate treatment effects of email campaigns on conversion and spend.

## Development Commands

```bash
# Install dependencies
poetry install

# Run Jupyter Lab for notebook development
poetry run jupyter lab

# Run a specific notebook
poetry run jupyter execute notebooks/01_exploration.ipynb
```

## Architecture

### Data Pipeline
- `data/raw/hillstrom.csv` - Source dataset (email marketing A/B/C test)
- `data/processed/` - Processed outputs (e.g., `bayesian_results.json`)
- `reports/figures/` - Generated visualizations from notebooks

### Analysis Notebooks
- `notebooks/01_exploration.ipynb` - Data exploration, balance checks, heterogeneity analysis
- `notebooks/02_bayesian_ab_testing.ipynb` - Bayesian A/B/C testing with PyMC (Beta-Binomial for conversion, Normal for spend)

### Planned Components (Empty Scaffolding)
- `src/api/` - API endpoints
- `src/ingestion/` - Data ingestion
- `src/models/` - Causal models
- `streamlit/` - Dashboard
- `dbt/` - Data transformations
- `docker/` - Containerization

## Key Libraries

- **PyMC** - Bayesian modeling (Beta-Binomial models, MCMC sampling)
- **ArviZ** - Posterior diagnostics and visualization
- **pandas/seaborn/matplotlib** - Data manipulation and plotting

## Dataset Schema (Hillstrom)

| Column | Description |
|--------|-------------|
| treatment | `Mens E-Mail`, `Womens E-Mail`, `No E-Mail` |
| conversion | Binary outcome (0/1) |
| visit | Binary website visit (0/1) |
| spend | Dollar amount spent |
| recency, history, mens, womens, newbie, channel, zip_code | Covariates for heterogeneity analysis |

## Bayesian Analysis Pattern

The project uses this pattern for A/B testing:
1. Beta(1,1) prior (uninformative) for conversion rates
2. Binomial likelihood
3. Compute posterior P(Treatment > Control), lift, and HDI intervals
4. Expected loss for decision-making

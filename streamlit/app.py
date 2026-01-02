import streamlit as st
from utils import load_bayesian_results, load_causal_ml_results

st.set_page_config(
    page_title="Causal Inference Platform",
    page_icon=":chart_with_upwards_trend:",
    layout="wide"
)

st.title("Causal Inference Platform")
st.markdown("### Analyse A/B Testing avec Inference Causale")

st.markdown("""
Cette plateforme présente les résultats d'une analyse causale sur le dataset **Hillstrom**
(campagne email marketing) avec :

- **64,000 clients** répartis en 3 groupes (Mens E-Mail, Womens E-Mail, No E-Mail)
- **Objectif** : Identifier le meilleur traitement pour maximiser les conversions
""")

st.divider()

# Key metrics
col1, col2, col3, col4 = st.columns(4)

# Load results
try:
    bayesian = load_bayesian_results()
    causal = load_causal_ml_results()

    with col1:
        st.metric("Clients", "64,000")

    with col2:
        mens_lift = bayesian.get("mens_email", {}).get("lift_vs_control", 0)
        st.metric("Lift Mens E-Mail", f"+{mens_lift:.0%}")

    with col3:
        womens_lift = bayesian.get("womens_email", {}).get("lift_vs_control", 0)
        st.metric("Lift Womens E-Mail", f"+{womens_lift:.0%}")

    with col4:
        match_rate = causal.get("match_rate", 0)
        st.metric("Politique Optimale", f"{match_rate:.1%} match")

except FileNotFoundError:
    st.warning("Exécutez les notebooks pour générer les résultats.")
    with col1:
        st.metric("Clients", "64,000")
    with col2:
        st.metric("Lift Mens E-Mail", "N/A")
    with col3:
        st.metric("Lift Womens E-Mail", "N/A")
    with col4:
        st.metric("Politique Optimale", "N/A")

st.divider()

# Navigation
st.markdown("### Naviguer")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.page_link("pages/1_Exploration.py", label="Exploration", icon="1️⃣")
    st.caption("Balance et hétérogénéité")

with col2:
    st.page_link("pages/2_Bayesian_AB.py", label="Bayesian A/B", icon="2️⃣")
    st.caption("Posteriors et lift")

with col3:
    st.page_link("pages/3_CausalML.py", label="CausalML", icon="3️⃣")
    st.caption("CATE et politique optimale")

with col4:
    st.page_link("pages/4_Simulateur.py", label="Simulateur", icon="🎯")
    st.caption("Prédiction interactive")

st.divider()

st.markdown("""
### Méthodologie

| Phase | Méthode | Librairie |
|-------|---------|-----------|
| 1. Exploration | Balance checks, hétérogénéité | pandas, seaborn |
| 2. Bayesian A/B | Beta-Binomial, MCMC | PyMC, ArviZ |
| 3. CausalML | X-Learner, SHAP | CausalML, SHAP |
| 4. API | Recommandations temps réel | FastAPI |
| 5. Dashboard | Visualisation interactive | Streamlit |
""")

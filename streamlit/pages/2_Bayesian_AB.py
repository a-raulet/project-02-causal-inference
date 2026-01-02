import streamlit as st
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import get_figure_path, load_bayesian_results

st.set_page_config(page_title="Bayesian A/B", layout="wide")

st.title("Phase 2 - Bayesian A/B Testing")

st.markdown("""
Analyse bayésienne des taux de conversion avec :
- **Prior** : Beta(1,1) non-informatif
- **Likelihood** : Binomial
- **Posterior** : Distribution complète des taux de conversion
""")

st.divider()

# Load results
try:
    results = load_bayesian_results()

    # Metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "P(Mens > Control)",
            f"{results.get('mens_email', {}).get('prob_superior', 0):.1%}"
        )

    with col2:
        st.metric(
            "P(Womens > Control)",
            f"{results.get('womens_email', {}).get('prob_superior', 0):.1%}"
        )

    with col3:
        st.metric(
            "Lift Mens vs Womens",
            f"+{(results.get('mens_email', {}).get('lift_vs_control', 0) - results.get('womens_email', {}).get('lift_vs_control', 0)):.0%}"
        )

except FileNotFoundError:
    st.warning("Exécutez le notebook 02_bayesian_ab_testing.ipynb pour générer les résultats.")

st.divider()

# Figure 5: Posteriors
st.subheader("1. Distributions Postérieures")
fig5 = get_figure_path("05_posteriors_conversion.png")
if fig5.exists():
    st.image(str(fig5), width="stretch")
else:
    st.warning("Figure non disponible.")

st.markdown("""
**Interprétation :**
- Les posteriors des traitements email sont décalés vers la droite vs Control
- Faible chevauchement = haute confiance dans l'effet
""")

st.divider()

# Figure 6: Forest Plot
st.subheader("2. Forest Plot - Lift vs Control")
fig6 = get_figure_path("06_forest_plot_conversion.png")
if fig6.exists():
    st.image(str(fig6), width="stretch")
else:
    st.warning("Figure non disponible.")

st.divider()

# Figure 7: Summary
st.subheader("3. Synthèse Bayésienne")
fig7 = get_figure_path("07_bayesian_summary.png")
if fig7.exists():
    st.image(str(fig7), width="stretch")
else:
    st.warning("Figure non disponible.")

st.divider()

# Results table
st.subheader("4. Résultats Détaillés")

try:
    results = load_bayesian_results()

    data = []
    for treatment in ["mens_email", "womens_email"]:
        if treatment in results:
            r = results[treatment]
            data.append({
                "Traitement": treatment.replace("_", " ").title(),
                "Taux Conversion": f"{r.get('conversion_rate', 0):.2%}",
                "Lift vs Control": f"+{r.get('lift_vs_control', 0):.0%}",
                "P(Supérieur)": f"{r.get('prob_superior', 0):.1%}",
                "HDI 94%": f"[{r.get('hdi_low', 0):.4f}, {r.get('hdi_high', 0):.4f}]"
            })

    if data:
        st.table(data)

except FileNotFoundError:
    pass

st.markdown("""
**Conclusion Phase 2 :**
- Les deux campagnes email augmentent significativement les conversions
- **Mens E-Mail** a un effet plus fort que Womens E-Mail
- P(Treatment > Control) > 99% pour les deux traitements
""")

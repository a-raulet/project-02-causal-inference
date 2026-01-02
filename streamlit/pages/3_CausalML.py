import streamlit as st
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import get_figure_path, load_causal_ml_results

st.set_page_config(page_title="CausalML", layout="wide")

st.title("Phase 3 - CausalML & CATE")

st.markdown("""
Estimation des **Conditional Average Treatment Effects (CATE)** pour personnaliser
l'allocation des traitements avec :
- **X-Learner** : Meta-learner hybride
- **SHAP** : Interprétabilité des effets hétérogènes
- **Politique optimale** : Assigner le meilleur traitement par client
""")

st.divider()

# Load results
try:
    results = load_causal_ml_results()

    col1, col2, col3 = st.columns(3)

    with col1:
        cate_mens = results.get("cate_stats", {}).get("mens_email", {}).get("mean", 0)
        st.metric("CATE moyen Mens", f"+{cate_mens:.2%}")

    with col2:
        cate_womens = results.get("cate_stats", {}).get("womens_email", {}).get("mean", 0)
        st.metric("CATE moyen Womens", f"+{cate_womens:.2%}")

    with col3:
        conv_matched = results.get("conversion_matched", 0)
        conv_unmatched = results.get("conversion_unmatched", 0)
        lift = (conv_matched - conv_unmatched) / conv_unmatched if conv_unmatched > 0 else 0
        st.metric("Lift politique optimale", f"+{lift:.0%}")

except FileNotFoundError:
    st.warning("Exécutez le notebook 03_causal_ml.ipynb")

st.divider()

# Figure 8: CATE Distributions
st.subheader("1. Distribution des CATE par Meta-Learner")
fig8 = get_figure_path("08_cate_distributions.png")
if fig8.exists():
    st.image(str(fig8), width="stretch")
else:
    st.warning("Figure non disponible.")

st.markdown("""
**Observations :**
- X-Learner produit des estimations plus stables que T-Learner
- Distribution centrée autour de l'ATE mais avec variance significative
""")

st.divider()

# Figure 9: Heterogeneity
st.subheader("2. Hétérogénéité des CATE")
fig9 = get_figure_path("09_cate_heterogeneity.png")
if fig9.exists():
    st.image(str(fig9), width="stretch")
else:
    st.warning("Figure non disponible.")

st.divider()

# SHAP Figures
col1, col2 = st.columns(2)

with col1:
    st.subheader("3. SHAP Summary")
    fig10 = get_figure_path("10_shap_summary.png")
    if fig10.exists():
        st.image(str(fig10), width="stretch")

with col2:
    st.subheader("4. SHAP Importance")
    fig11 = get_figure_path("11_shap_importance.png")
    if fig11.exists():
        st.image(str(fig11), width="stretch")

st.markdown("""
**Variables les plus importantes pour l'hétérogénéité :**
- `mens` / `womens` : historique d'achat
- `history` : montant total dépensé
- `recency` : récence du dernier achat
""")

st.divider()

# Figure 12: Optimal Policy
st.subheader("5. Politique d'Allocation Optimale")
fig12 = get_figure_path("12_optimal_policy.png")
if fig12.exists():
    st.image(str(fig12), width="stretch")
else:
    st.warning("Figure non disponible.")

# Policy distribution
try:
    results = load_causal_ml_results()
    optimal = results.get("optimal_distribution", {})
    current = results.get("current_distribution", {})

    st.markdown("**Distribution recommandée vs actuelle :**")

    col1, col2, col3 = st.columns(3)

    total = sum(optimal.values()) if optimal else 1

    with col1:
        opt_mens = optimal.get("Mens E-Mail", 0)
        cur_mens = current.get("Mens E-Mail", 0)
        st.metric(
            "Mens E-Mail",
            f"{opt_mens/total:.0%}",
            delta=f"{(opt_mens - cur_mens)/total:+.0%} vs actuel"
        )

    with col2:
        opt_womens = optimal.get("Womens E-Mail", 0)
        cur_womens = current.get("Womens E-Mail", 0)
        st.metric(
            "Womens E-Mail",
            f"{opt_womens/total:.0%}",
            delta=f"{(opt_womens - cur_womens)/total:+.0%} vs actuel"
        )

    with col3:
        opt_none = optimal.get("No E-Mail", 0)
        cur_none = current.get("No E-Mail", 0)
        st.metric(
            "No E-Mail",
            f"{opt_none/total:.0%}",
            delta=f"{(opt_none - cur_none)/total:+.0%} vs actuel"
        )

except FileNotFoundError:
    pass

st.divider()

# Figure 13: Qini Curves
st.subheader("6. Courbes Uplift (Qini & Gain)")
fig13 = get_figure_path("13_qini_gain_curves.png")
if fig13.exists():
    st.image(str(fig13), width="stretch")
else:
    st.warning("Figure non disponible.")

st.markdown("""
**Interprétation :**
- Courbe au-dessus de la diagonale = modèle meilleur que random
- AUUC > 0 confirme la valeur du ciblage personnalisé
""")

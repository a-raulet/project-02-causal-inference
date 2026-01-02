import streamlit as st
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import get_figure_path

st.set_page_config(page_title="Exploration", layout="wide")

st.title("Phase 1 - Exploration des Données")

st.markdown("""
Analyse exploratoire du dataset Hillstrom pour vérifier :
- La **répartition des traitements** (randomisation)
- L'**équilibre des covariables** entre groupes
- L'**hétérogénéité** potentielle des effets
""")

st.divider()

# Figure 1: Treatment Overview
st.subheader("1. Répartition des Traitements")
fig1 = get_figure_path("01_treatment_overview.png")
if fig1.exists():
    st.image(str(fig1), width="stretch")
else:
    st.warning("Figure non disponible. Exécutez le notebook 01_exploration.ipynb")

st.markdown("""
**Observations :**
- Répartition équilibrée ~21,000 clients par groupe
- Design A/B/C classique avec groupe contrôle
""")

st.divider()

# Figure 2: Covariate Balance
st.subheader("2. Équilibre des Covariables")
fig2 = get_figure_path("02_covariate_balance.png")
if fig2.exists():
    st.image(str(fig2), width="stretch")
else:
    st.warning("Figure non disponible. Exécutez le notebook 01_exploration.ipynb")

st.markdown("""
**Observations :**
- Bonne balance sur toutes les covariables
- Pas de biais de sélection détecté
- Randomisation réussie
""")

st.divider()

# Figure 3: Heterogeneity
st.subheader("3. Exploration de l'Hétérogénéité")
fig3 = get_figure_path("03_heterogeneity_exploration.png")
if fig3.exists():
    st.image(str(fig3), width="stretch")
else:
    st.warning("Figure non disponible. Exécutez le notebook 01_exploration.ipynb")

st.markdown("""
**Observations :**
- Hétérogénéité suspectée selon l'historique d'achat (mens/womens)
- Les clients avec historique "Mens" semblent mieux répondre aux Mens E-Mail
- Justifie l'estimation des CATE en Phase 3
""")

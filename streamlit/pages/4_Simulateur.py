import streamlit as st
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import (
    load_cate_models,
    preprocess_for_prediction,
    HISTORY_SEGMENTS,
    ZIP_CODES,
    CHANNELS
)

st.set_page_config(page_title="Simulateur", layout="wide")

st.title("Simulateur de Recommandation")

st.markdown("""
Entrez les caractéristiques d'un client pour obtenir la recommandation
de traitement optimale basée sur les modèles CATE.
""")

st.divider()

# Load models
try:
    mens_model, womens_model = load_cate_models()
    models_loaded = True
except FileNotFoundError:
    st.error("Modèles non trouvés. Exécutez le notebook 03_causal_ml.ipynb")
    models_loaded = False

if models_loaded:
    # Input form
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Profil Client")

        recency = st.slider(
            "Récence (mois depuis dernier achat)",
            min_value=1, max_value=12, value=6
        )

        history = st.number_input(
            "Historique d'achat ($)",
            min_value=0.0, max_value=5000.0, value=200.0, step=50.0
        )

        history_segment = st.selectbox(
            "Segment historique",
            options=HISTORY_SEGMENTS,
            index=2
        )

        zip_code = st.selectbox(
            "Zone géographique",
            options=ZIP_CODES,
            index=2  # Urban
        )

    with col2:
        st.subheader("Comportement")

        channel = st.selectbox(
            "Canal d'achat",
            options=CHANNELS,
            index=0  # Web
        )

        mens = st.checkbox("A acheté produits Mens", value=True)
        womens = st.checkbox("A acheté produits Womens", value=False)
        newbie = st.checkbox("Nouveau client", value=False)

    st.divider()

    # Predict button
    if st.button("Obtenir la recommandation", type="primary", width="stretch"):
        # Preprocess
        X = preprocess_for_prediction(
            recency=recency,
            history=history,
            history_segment=history_segment,
            mens=int(mens),
            womens=int(womens),
            newbie=int(newbie),
            zip_code=zip_code,
            channel=channel
        )

        # Predict
        cate_mens = float(mens_model.predict(X)[0])
        cate_womens = float(womens_model.predict(X)[0])

        # Determine optimal
        if cate_mens > cate_womens and cate_mens > 0:
            optimal = "Mens E-Mail"
            lift = cate_mens
            color = "blue"
        elif cate_womens > cate_mens and cate_womens > 0:
            optimal = "Womens E-Mail"
            lift = cate_womens
            color = "red"
        else:
            optimal = "No E-Mail"
            lift = 0.0
            color = "gray"

        # Display results
        st.divider()
        st.subheader("Recommandation")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("CATE Mens E-Mail", f"{cate_mens:+.2%}")

        with col2:
            st.metric("CATE Womens E-Mail", f"{cate_womens:+.2%}")

        with col3:
            st.metric("Lift attendu", f"{lift:+.2%}")

        st.divider()

        # Recommendation box
        if optimal == "Mens E-Mail":
            st.success(f"### Recommandation : {optimal}")
            st.markdown("""
            Ce client devrait recevoir la campagne **Mens E-Mail**.

            L'effet causal estimé (CATE) indique que ce traitement maximisera
            la probabilité de conversion pour ce profil client.
            """)
        elif optimal == "Womens E-Mail":
            st.info(f"### Recommandation : {optimal}")
            st.markdown("""
            Ce client devrait recevoir la campagne **Womens E-Mail**.

            L'effet causal estimé (CATE) indique que ce traitement maximisera
            la probabilité de conversion pour ce profil client.
            """)
        else:
            st.warning(f"### Recommandation : {optimal}")
            st.markdown("""
            Ce client ne devrait **pas** recevoir d'email marketing.

            Les CATE négatifs ou nuls indiquent que l'envoi d'email n'améliorerait
            pas (voire réduirait) la probabilité de conversion.
            """)

        # Comparison chart
        st.divider()
        st.subheader("Comparaison des traitements")

        import pandas as pd

        chart_data = pd.DataFrame({
            "Traitement": ["Mens E-Mail", "Womens E-Mail", "No E-Mail"],
            "CATE": [cate_mens, cate_womens, 0.0]
        })

        st.bar_chart(chart_data.set_index("Traitement"))

else:
    st.info("Le simulateur sera disponible après l'exécution du notebook 03_causal_ml.ipynb")

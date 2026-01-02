import json
from pathlib import Path
import joblib
import numpy as np

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
MODELS_DIR = PROJECT_ROOT / "models"


def load_bayesian_results() -> dict:
    """Load Bayesian A/B testing results."""
    path = DATA_DIR / "processed" / "bayesian_results.json"
    with open(path) as f:
        return json.load(f)


def load_causal_ml_results() -> dict:
    """Load CausalML results."""
    path = DATA_DIR / "processed" / "causal_ml_results.json"
    with open(path) as f:
        return json.load(f)


def get_figure_path(name: str) -> Path:
    """Get path to a figure."""
    return FIGURES_DIR / name


def load_cate_models():
    """Load CATE prediction models."""
    mens_model = joblib.load(MODELS_DIR / "cate_model_mens.pkl")
    womens_model = joblib.load(MODELS_DIR / "cate_model_womens.pkl")
    return mens_model, womens_model


# History segment mapping
HISTORY_SEGMENT_MAP = {
    '1) $0 - $100': 1,
    '2) $100 - $200': 2,
    '3) $200 - $350': 3,
    '4) $350 - $500': 4,
    '5) $500 - $750': 5,
    '6) $750 - $1,000': 6,
    '7) $1,000 +': 7
}

HISTORY_SEGMENTS = list(HISTORY_SEGMENT_MAP.keys())
ZIP_CODES = ["Rural", "Suburban", "Urban"]
CHANNELS = ["Web", "Phone", "Multichannel"]


def preprocess_for_prediction(
    recency: int,
    history: float,
    history_segment: str,
    mens: int,
    womens: int,
    newbie: int,
    zip_code: str,
    channel: str
) -> np.ndarray:
    """Convert inputs to feature array for model prediction."""
    features = np.zeros(12)

    features[0] = recency
    features[1] = history
    features[2] = HISTORY_SEGMENT_MAP.get(history_segment, 1)
    features[3] = mens
    features[4] = womens
    features[5] = newbie
    features[6] = 1 if zip_code == 'Rural' else 0
    features[7] = 1 if zip_code == 'Suburban' else 0
    features[8] = 1 if zip_code == 'Urban' else 0
    features[9] = 1 if channel == 'Multichannel' else 0
    features[10] = 1 if channel == 'Phone' else 0
    features[11] = 1 if channel == 'Web' else 0

    return features.reshape(1, -1)

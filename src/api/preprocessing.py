import numpy as np

# History segment ordinal mapping
HISTORY_SEGMENT_MAP = {
    '1) $0 - $100': 1,
    '2) $100 - $200': 2,
    '3) $200 - $350': 3,
    '4) $350 - $500': 4,
    '5) $500 - $750': 5,
    '6) $750 - $1,000': 6,
    '7) $1,000 +': 7
}

# Feature order expected by the model
FEATURE_NAMES = [
    'recency', 'history', 'history_segment_ord',
    'mens', 'womens', 'newbie',
    'zip_Rural', 'zip_Suburban', 'zip_Urban',
    'channel_Multichannel', 'channel_Phone', 'channel_Web'
]


def preprocess_customer(customer: dict) -> np.ndarray:
    """
    Convert customer input to feature array for model prediction.

    Args:
        customer: Dictionary with customer features

    Returns:
        numpy array of shape (1, 12) with encoded features
    """
    features = np.zeros(12)

    # Numeric features
    features[0] = customer['recency']
    features[1] = customer['history']
    features[2] = HISTORY_SEGMENT_MAP.get(customer['history_segment'], 1)

    # Binary features
    features[3] = customer['mens']
    features[4] = customer['womens']
    features[5] = customer['newbie']

    # One-hot encode zip_code
    zip_code = customer['zip_code']
    features[6] = 1 if zip_code == 'Rural' else 0
    features[7] = 1 if zip_code == 'Suburban' else 0
    features[8] = 1 if zip_code == 'Urban' else 0

    # One-hot encode channel
    channel = customer['channel']
    features[9] = 1 if channel == 'Multichannel' else 0
    features[10] = 1 if channel == 'Phone' else 0
    features[11] = 1 if channel == 'Web' else 0

    return features.reshape(1, -1)


def preprocess_batch(customers: list[dict]) -> np.ndarray:
    """
    Convert multiple customers to feature array.

    Args:
        customers: List of customer dictionaries

    Returns:
        numpy array of shape (n_customers, 12)
    """
    return np.vstack([preprocess_customer(c) for c in customers])

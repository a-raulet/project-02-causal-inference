import joblib
from pathlib import Path

# Path to models directory
MODELS_DIR = Path(__file__).parent.parent.parent / "models"


class CATEModels:
    """Singleton class to load and hold CATE models."""

    _instance = None
    _models_loaded = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.cate_model_mens = None
            cls._instance.cate_model_womens = None
        return cls._instance

    def load_models(self):
        """Load models from pickle files."""
        if self._models_loaded:
            return

        mens_path = MODELS_DIR / "cate_model_mens.pkl"
        womens_path = MODELS_DIR / "cate_model_womens.pkl"

        if not mens_path.exists() or not womens_path.exists():
            raise FileNotFoundError(
                f"Model files not found. Expected:\n"
                f"  - {mens_path}\n"
                f"  - {womens_path}\n"
                f"Run notebook 03_causal_ml.ipynb to generate them."
            )

        self.cate_model_mens = joblib.load(mens_path)
        self.cate_model_womens = joblib.load(womens_path)
        self._models_loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._models_loaded

    def predict(self, X):
        """
        Predict CATE for both treatments.

        Args:
            X: Feature array of shape (n_samples, 12)

        Returns:
            Tuple of (cate_mens, cate_womens) arrays
        """
        if not self._models_loaded:
            self.load_models()

        cate_mens = self.cate_model_mens.predict(X)
        cate_womens = self.cate_model_womens.predict(X)

        return cate_mens, cate_womens


# Global instance
cate_models = CATEModels()

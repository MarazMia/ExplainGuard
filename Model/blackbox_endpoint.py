import numpy as np
import pandas as pd

class BlackBoxModelEndpoint:
    """
    Bare-minimum Black-Box API Endpoint provided to the auditor.
    Only exposes single-class probability outputs needed for SHAP and LIME.
    """
    def __init__(self, model):
        self._model = model

    def predict_proba(self, X):
        """
        Returns a 1D array of probabilities for Class 1.
        Accepts DataFrame or NumPy array.
        """
        if isinstance(X, pd.DataFrame):
            X_input = X.values
        else:
            X_input = X
            
        # Extract Class 1 (positive class) probability array
        probs = self._model.predict_proba(X_input)
        if probs.ndim == 2 and probs.shape[1] == 2:
            return probs[:, 1]
        return probs.ravel()

    def __call__(self, X):
        """Callable interface compatibility for SHAP KernelExplainer."""
        return self.predict_proba(X)
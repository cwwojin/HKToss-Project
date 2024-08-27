from sklearn.base import BaseEstimator
import pickle
from skl2onnx import convert_sklearn


class BaseSKLearnModel:
    model: type[BaseEstimator]
    model_name: str

    def __init__(self, model_name: str, **kwargs):
        self.model = None
        self.model_name = model_name

    def fit(self, X, y=None, **kwargs):
        return self.model.fit(X, y)

    def predict(self, X, **kwargs):
        return self.model.predict(X)

    def predict_proba(self, X, **kwargs):
        return self.model.predict_proba(X)

    def export_pkl(self, path: str, **kwargs):
        """Export a model to file via Pickle."""
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def export_onnx(self, name: str, initial_types=None, options=None, **kwargs):
        """Export a model to ONNX."""
        return convert_sklearn(
            self.model, name=name, initial_types=initial_types, options=options
        )

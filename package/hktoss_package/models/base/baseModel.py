from sklearn.base import BaseEstimator
import os.path as path
import pickle
from skl2onnx import convert_sklearn


class BaseSKLearnModel:
    model: type[BaseEstimator]

    def __init__(self, model: type[BaseEstimator] = None, **kwargs):
        if model:
            self.model = model

    def fit(self, X, y=None, **kwargs):
        return self.model.fit(X, y)

    def predict(self, X, **kwargs):
        return self.model.predict(X)

    def export_pkl(self, save_dir=".", model_name="model.pkl", **kwargs):
        """Export a model to file via Pickle."""

        with open(path.join(save_dir, model_name), "wb") as f:
            pickle.dump(self.model, f)

    def export_onnx(self, name: str, initial_types=None, options=None, **kwargs):
        """Export a model to ONNX."""
        return convert_sklearn(
            self.model, name=name, initial_types=initial_types, options=options
        )

from sklearn.base import BaseEstimator
import os.path as path
import pickle
from skl2onnx import convert_sklearn
from yacs.config import CfgNode as CN


class BaseSKLearnModel:
    model: type[BaseEstimator]
    config: CN

    def __init__(self, config: CN, **kwargs):
        self.model = None
        self.config = config
        self.config["model_name"] = self._set_model_name()

    def _set_model_name(self):
        return f"{self.config.MODEL_TYPE}"

    def fit(self, X, y=None, **kwargs):
        return self.model.fit(X, y)

    def predict(self, X, **kwargs):
        return self.model.predict(X)

    def predict_proba(self, X, **kwargs):
        return self.model.predict_proba(X)

    def export_pkl(self, save_dir=".", model_name="model.pkl", **kwargs):
        """Export a model to file via Pickle."""

        with open(path.join(save_dir, model_name), "wb") as f:
            pickle.dump(self.model, f)

    def export_onnx(self, name: str, initial_types=None, options=None, **kwargs):
        """Export a model to ONNX."""
        return convert_sklearn(
            self.model, name=name, initial_types=initial_types, options=options
        )

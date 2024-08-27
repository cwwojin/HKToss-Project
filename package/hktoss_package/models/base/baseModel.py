import pickle
from abc import abstractmethod

from skl2onnx import convert_sklearn
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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


class BaseSKLearnPipeline:
    model: type[BaseEstimator]
    pipeline: Pipeline
    model_name: str

    def __init__(self, model_name: str, **kwargs):
        self.model = None
        self.pipeline = None
        self.model_name = model_name
        self.scaler = StandardScaler()
        self.pca = PCA()

    @abstractmethod
    def build_pipe(self):
        pass

    def fit(self, X, y=None, **kwargs):
        return self.pipeline.fit(X, y)

    def predict(self, X, **kwargs):
        return self.pipeline.predict(X)

    def predict_proba(self, X, **kwargs):
        return self.pipeline.predict_proba(X)

    def export_pkl(self, path: str, **kwargs):
        """Export a model to file via Pickle."""
        with open(path, "wb") as f:
            pickle.dump(self.pipeline, f)

    def export_onnx(self, name: str, initial_types=None, options=None, **kwargs):
        """Export a model to ONNX."""
        return convert_sklearn(
            self.pipeline, name=name, initial_types=initial_types, options=options
        )

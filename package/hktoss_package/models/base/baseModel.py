import pickle
from abc import abstractmethod

from skl2onnx import convert_sklearn
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline


class BaseSKLearnPipeline:
    model: type[BaseEstimator]
    pipeline: Pipeline
    model_name: str

    def __init__(self, **kwargs):
        self.model = None
        self.pipeline = None

    @abstractmethod
    def build_pipe(self, **kwargs):
        pass

    @abstractmethod
    def build_pipe_transformer(self, **kwargs):
        pass

    def get_params(self, **kwargs):
        return self.pipeline.get_params(**kwargs)

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

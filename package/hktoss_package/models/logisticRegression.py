from hktoss_package.models.base import BaseSKLearnModel, BaseSKLearnPipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class LogisticRegressionModel(BaseSKLearnModel):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.model = LogisticRegression()


class LogisticRegressionPipeline(BaseSKLearnPipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = LogisticRegression()
        self.pipeline = self.build_pipe()

    def build_pipe(self):
        self.scaler = StandardScaler()
        self.pca = PCA()
        return Pipeline(
            steps=[
                ("scaler", self.scaler),
                ("pca", self.pca),
                ("classifier", self.model),
            ],
        )

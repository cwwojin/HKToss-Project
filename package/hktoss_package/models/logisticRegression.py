from hktoss_package.trainers.mlflow import MLFlowTrainer
from hktoss_package.models.base import BaseSKLearnModel, BaseSKLearnPipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer, make_column_selector


class LogisticRegressionModel(BaseSKLearnModel):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.model = LogisticRegression()


class LogisticRegressionPipeline(BaseSKLearnPipeline):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.model = LogisticRegression()
        # self.pipeline = self.build_pipe()
        self.pipeline = self.bulid_pipe_transformer()

    # pipeline scaler+pca => columntransform change
    def build_pipe(self):
        self.scaler = StandardScaler()
        self.pca = PCA()
        return Pipeline(
            steps=[
                ("scaler", self.scaler),
                ("pca", self.pca),
                ("classifier", self.model),
            ]
        )

    def bulid_pipe_transformer(self):
        self.scaler = StandardScaler()
        self.pca = PCA()

        self.preprocessor = ColumnTransformer(
            transformers=[
                (
                    "scaler_pca",
                    Pipeline([("scaler", self.scaler), ("pca", self.pca)]),
                    # 숫자형 데이터
                ),
            ],
            remainder="passthrough",
        )
        return Pipeline(
            steps=[
                ("preprocessor", self.preprocessor)("classifier", self.model),
            ]
        )

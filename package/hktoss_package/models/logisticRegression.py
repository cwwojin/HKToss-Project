from hktoss_package.models.base import BaseSKLearnModel, BaseSKLearnPipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer


class LogisticRegressionModel(BaseSKLearnModel):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.model = LogisticRegression()


class LogisticRegressionPipeline(BaseSKLearnPipeline):
    def __init__(self, column_type=None, **kwargs):
        super().__init__(**kwargs)
        self.model = LogisticRegression()
        self.pipeline = self.build_pipe_transformer(column_type)

    # pipeline
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

    def build_pipe_transformer(self, column_type):
        self.scaler = StandardScaler()
        self.pca = PCA()

        # Check if numeric columns exist
        if "num" in column_type and column_type["num"]:
            self.preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "scaler_pca",
                        Pipeline([("scaler", self.scaler), ("pca", self.pca)]),
                        column_type["num"],
                    ),
                ],
                remainder="passthrough",
            )
        else:
            # If no numeric columns, no need for ColumnTransformer
            self.preprocessor = "passthrough"

        return Pipeline(
            steps=[
                ("preprocessor", self.preprocessor),
                ("classifier", self.model),
            ]
        )

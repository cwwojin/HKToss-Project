from hktoss_package.models.base import BaseSKLearnModel, BaseSKLearnPipeline
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer, make_column_selector
from xgboost import XGBClassifier


class XGBClassifierModel(BaseSKLearnModel):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.model = XGBClassifier()


class XGBPipeline(BaseSKLearnPipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = XGBClassifier()
        self.pipeline = self.build_pipe_transformer(**kwargs)

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

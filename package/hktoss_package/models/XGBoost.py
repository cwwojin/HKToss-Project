from hktoss_package.models.base import BaseSKLearnPipeline
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier


class XGBPipeline(BaseSKLearnPipeline):
    def __init__(self, column_types=None, **kwargs):
        super().__init__(**kwargs)
        self.model = XGBClassifier()
        self.preprocessor = None
        if column_types:
            self.pipeline = self.build_pipe_transformer(column_types)
        else:
            self.pipeline = self.build_pipe()

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

    def build_pipe_transformer(self, column_types: dict):
        self.scaler = StandardScaler()
        self.pca = PCA()
        self.encoder = OneHotEncoder(handle_unknown="ignore")
        self.preprocessor = None

        # Build column transformer
        transformers = []
        if "num" in column_types and column_types["num"]:
            transformers.append(
                (
                    "numeric",
                    Pipeline([("scaler", self.scaler), ("pca", self.pca)]),
                    column_types["num"],
                )
            )
        if "cat" in column_types and column_types["cat"]:
            transformers.append(
                ("cat", Pipeline([("one_hot", self.encoder)]), column_types["cat"])
            )

        # Build pipe
        if transformers:
            self.preprocessor = ColumnTransformer(
                transformers=transformers, remainder="passthrough"
            )
            return Pipeline(
                steps=[
                    (
                        "preprocessor",
                        self.preprocessor,
                    ),
                    ("classifier", self.model),
                ]
            )
        else:
            return Pipeline(
                steps=[
                    ("classifier", self.model),
                ]
            )

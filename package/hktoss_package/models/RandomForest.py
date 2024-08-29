from hktoss_package.models.base import BaseSKLearnModel, BaseSKLearnPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from yacs.config import CfgNode as CN


class RandomForestClassifierModel(BaseSKLearnModel):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.model = RandomForestClassifier()


class RandomForestPipeline(BaseSKLearnPipeline):
    def __init__(self, column_types=None, **kwargs):
        super().__init__(**kwargs)
        self.model = RandomForestClassifier()
        self.preprocessor = None
        if column_types:
            self.pipeline = self.build_pipe_transformer(column_types)
        else:
            self.pipeline = self.build_pipe()

    def build_pipe(self):
        self.scaler = StandardScaler()
        return Pipeline(
            steps=[
                ("scaler", self.scaler),
                ("classifier", self.model),
            ]
        )

    def build_pipe_transformer(self, column_types: dict):
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder()

        # Build column transformer
        transformers = []
        if "num" in column_types and column_types["num"]:
            transformers.append(
                (
                    "numeric",
                    Pipeline([("scaler", self.scaler)]),
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

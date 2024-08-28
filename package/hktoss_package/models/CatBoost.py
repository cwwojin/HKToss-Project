from catboost import CatBoostClassifier
from hktoss_package.models.base import BaseSKLearnModel, BaseSKLearnPipeline
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer, make_column_selector


class CatBoostClassifierModel(BaseSKLearnModel):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.model = CatBoostClassifier()


class CatBoostPipeline(BaseSKLearnPipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = CatBoostClassifier()
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

    def bulid_pipe_transformer(self, df):
        self.scaler = StandardScaler()
        self.pca = PCA()

        numeric_features = make_column_selector(dtype_include=["int64", "float64"])

        self.preprocessor = ColumnTransformer(
            transformers=[
                (
                    "scaler_pca",
                    Pipeline([("scaler", self.scaler), ("pca", self.pca)]),
                    numeric_features(df),
                ),
            ],
            remainder="passthrough",
        )
        return Pipeline(
            steps=[
                ("preprocessor", self.preprocessor),
                ("classifier", self.model),
            ]
        )

    def fit(self, X, y=None, **kwargs):
        return self.pipeline.fit(X, y, classifier__verbose=False)

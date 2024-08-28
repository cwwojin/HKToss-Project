from hktoss_package.models.base import BaseSKLearnModel, BaseSKLearnPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from yacs.config import CfgNode as CN


class RandomForestClassifierModel(BaseSKLearnModel):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.model = RandomForestClassifier()


class RandomForestPipeline(BaseSKLearnPipeline):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.model = RandomForestClassifier()
        self.pipeline = self.build_pipe()

    def build_pipe(self):
        self.scaler = StandardScaler()
        return Pipeline(
            steps=[
                ("scaler", self.scaler),
                ("classifier", self.model),
            ]
        )

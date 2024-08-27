from hktoss_package.models.base import BaseSKLearnModel, BaseSKLearnPipeline
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class MLPClassifierModel(BaseSKLearnModel):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.model = MLPClassifier()


class MLPPipeline(BaseSKLearnPipeline):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.model = MLPClassifier()
        self.pipeline = self.build_pipe()

    def build_pipe(self):
        self.scaler = StandardScaler()
        return Pipeline(
            steps=[
                ("scaler", self.scaler),
                ("classifier", self.model),
            ]
        )

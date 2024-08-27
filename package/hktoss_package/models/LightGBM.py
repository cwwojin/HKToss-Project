from hktoss_package.models.base import BaseSKLearnModel, BaseSKLearnPipeline
from lightgbm import LGBMClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from yacs.config import CfgNode as CN


class LGBMClassifierModel(BaseSKLearnModel):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.model = LGBMClassifier()


class LGBMPipeline(BaseSKLearnPipeline):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.model = LGBMClassifier()
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

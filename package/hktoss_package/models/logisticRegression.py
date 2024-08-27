from sklearn.linear_model import LogisticRegression
from hktoss_package.models.base import BaseSKLearnModel
from yacs.config import CfgNode as CN


class LogisticRegressionModel(BaseSKLearnModel):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.model = LogisticRegression()

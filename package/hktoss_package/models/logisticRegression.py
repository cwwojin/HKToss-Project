from sklearn.linear_model import LogisticRegression
from hktoss_package.models.base import BaseSKLearnModel
from yacs.config import CfgNode as CN


class LogisticRegressionModel(BaseSKLearnModel):
    def __init__(self, config: CN):
        super().__init__(config)
        self.model = LogisticRegression()

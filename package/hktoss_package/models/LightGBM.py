from lightgbm import LGBMClassifier
from hktoss_package.models.base import BaseSKLearnModel
from yacs.config import CfgNode as CN


class LGBMClassifierModel(BaseSKLearnModel):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.model = LGBMClassifier()

from xgboost import XGBClassifier
from hktoss_package.models.base import BaseSKLearnModel
from yacs.config import CfgNode as CN


class XGBClassifierModel(BaseSKLearnModel):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.model = XGBClassifier()

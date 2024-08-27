from catboost import CatBoostClassifier
from hktoss_package.models.base import BaseSKLearnModel
from yacs.config import CfgNode as CN


class CatBoostClassifierModel(BaseSKLearnModel):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.model = CatBoostClassifier()

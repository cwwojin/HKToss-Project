from yacs.config import CfgNode as CN

_C = CN()

# ======================================================== #
# Data Config                                              #
# ======================================================== #

_C.DATASET = CN()

_C.DATASET.PATH = ".cache/train_data_cache.pkl"  # path to dataset file
_C.DATASET.TEST_SIZE = 0.2
_C.DATASET.RANDOM_STATE = 42
_C.DATASET.ID_COL_NAME = "SK_ID_CURR"
_C.DATASET.TARGET_COL_NAME = "TARGET"

# Sampling
_C.DATASET.SAMPLER = "composite"  # over_random | over_smote | under_random | composite (SMOTENC + RandomUnder)

# ======================================================== #
# Model Config                                             #
# ======================================================== #

# Default : logistic
_C.MODEL_TYPE = (
    "logistic"  # logistic | randomforest | xgboost | catboost | lightgbm | mlp
)

# Logistic Regression
_C.LOGISTIC = CN()
_C.LOGISTIC.C = [1]
_C.LOGISTIC.penalty = ["l2"]
_C.LOGISTIC.solver = [
    # "sag",
    "saga",
]
_C.LOGISTIC.max_iter = [200]
_C.LOGISTIC.class_weight = [
    None,
    # "balanced",
]  # 클래스 불균형 해결을 위한 옵션 추가: 소수 클래스에 가중치

# Random Forest
_C.RANDOMFOREST = CN()
_C.RANDOMFOREST.n_estimators = [200]
_C.RANDOMFOREST.max_depth = [10, 20]
_C.RANDOMFOREST.min_samples_split = [2, 5]
_C.RANDOMFOREST.max_features = ["sqrt", "log2"]
_C.RANDOMFOREST.class_weight = [
    None,
    # "balanced",
]  # 클래스 불균형 해결을 위한 옵션 추가: 소수 클래스에 가중치


# XGBoost
_C.XGBOOST = CN()
_C.XGBOOST.booster = ["gbtree", "gblinear"]
_C.XGBOOST.n_estimators = [100, 200]
_C.XGBOOST.max_depth = [3, 6]
_C.XGBOOST.learning_rate = [0.1]
_C.XGBOOST.subsample = [0.8, 1.0]
_C.XGBOOST.scale_pos_weight = [
    1,
    # 2,
    # 5,
]  # 불균형 데이터 처리를 위한 가중치 옵션 추가: 양성 클래스에 가중치


# LightGBM
_C.LIGHTGBM = CN()
_C.LIGHTGBM.n_estimators = [100]
_C.LIGHTGBM.max_depth = [10, 20]
_C.LIGHTGBM.learning_rate = [0.1]
_C.LIGHTGBM.num_leaves = [30, 60]
_C.LIGHTGBM.is_unbalance = [
    False,
    # True
]  # 불균형 데이터 처리를 위한 옵션

# CatBoost
_C.CATBOOST = CN()
_C.CATBOOST.iterations = [100]
_C.CATBOOST.depth = [4, 6]
_C.CATBOOST.learning_rate = [0.1]
_C.CATBOOST.l2_leaf_reg = [3]

# MLP
_C.MLP = CN()
_C.MLP.hidden_layer_sizes = [(100,)]
_C.MLP.activation = ["relu"]
_C.MLP.solver = ["adam"]
_C.MLP.alpha = [0.001, 0.01]
_C.MLP.learning_rate = ["adaptive"]

# ======================================================== #
# Experiment Config                                        #
# ======================================================== #

# Grid Search
_C.GRID_SEARCH = True
_C.MULTIPROCESSING = False
_C.PCA = CN()
_C.PCA.N_COMPONENTS = ["mle"]

# Threshold Search
_C.TUNE_THRESHOLD = True

# Logger
_C.LOGGER = CN()
_C.LOGGER.EXPERIMENT_NAME = None
_C.LOGGER.RUN_NAME = None

# ======================================================== #
# Airflow Config                                           #
# ======================================================== #

# Airflow
_C.AIRFLOW = True


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()


# 모듈 수준에서 cfg 객체를 정의
cfg = get_cfg_defaults()

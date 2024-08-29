from yacs.config import CfgNode as CN

_C = CN()

# ======================================================== #
# Data Config                                              #
# ======================================================== #

_C.DATASET = CN()

_C.DATASET.PATH = ".data/dataset.csv"  # path to dataset file
_C.DATASET.TEST_SIZE = 0.2
_C.DATASET.RANDOM_STATE = 42
_C.DATASET.ID_COL_NAME = "SK_ID_CURR"
_C.DATASET.TARGET_COL_NAME = "TARGET"

# Sampling
_C.DATASET.SAMPLER = None  # over_random | over_smote | under_random

# ======================================================== #
# Model Config                                             #
# ======================================================== #

# Default : logistic
_C.MODEL_TYPE = (
    "logistic"  # logistic | randomforest | xgboost | catboost | lightgbm | mlp
)

# Logistic Regression
_C.LOGISTIC = CN()
_C.LOGISTIC.C = [0.1, 1, 10]
_C.LOGISTIC.penalty = ["l1", "l2"]
_C.LOGISTIC.solver = [
    "sag",
    "saga",
]
_C.LOGISTIC.max_iter = [50, 100, 200]

# Random Forest
_C.RANDOMFOREST = CN()
_C.RANDOMFOREST.n_estimators = [100, 200, 300]
_C.RANDOMFOREST.max_depth = [10, 20, 30]
_C.RANDOMFOREST.min_samples_split = [2, 5, 10]
_C.RANDOMFOREST.max_features = [None, "sqrt", "log2"]


# XGBoost
_C.XGBOOST = CN()
_C.XGBOOST.n_estimators = [100, 200, 300]
_C.XGBOOST.max_depth = [3, 6, 9]
_C.XGBOOST.learning_rate = [0.01, 0.1, 0.2]
_C.XGBOOST.subsample = [0.6, 0.8, 1.0]


# LightGBM
_C.LIGHTGBM = CN()
_C.LIGHTGBM.n_estimators = [100, 200, 300]
_C.LIGHTGBM.max_depth = [10, 20, -1]
_C.LIGHTGBM.learning_rate = [0.01, 0.1, 0.2]
_C.LIGHTGBM.num_leaves = [30, 60, 120]

# CatBoost
_C.CATBOOST = CN()
_C.CATBOOST.iterations = [100, 200, 300]
_C.CATBOOST.depth = [4, 6, 8]
_C.CATBOOST.learning_rate = [0.01, 0.1, 0.2]
_C.CATBOOST.l2_leaf_reg = [1, 3, 5]

# MLP
_C.MLP = CN()
_C.MLP.hidden_layer_sizes = [(50,), (100,), (100, 50), (100, 100, 50)]
_C.MLP.activation = ["relu", "tanh"]
_C.MLP.solver = ["adam", "lbfgs"]
_C.MLP.alpha = [0.0001, 0.001, 0.01]
_C.MLP.learning_rate = ["adaptive"]

# ======================================================== #
# Experiment Config                                        #
# ======================================================== #

# Grid Search
_C.GRID_SEARCH = True
_C.MULTIPROCESSING = False
_C.PCA = CN()
_C.PCA.N_COMPONENTS = [5, 10, 20, 50, 100]

# Logger
_C.LOGGER = CN()
_C.LOGGER.EXPERIMENT_NAME = None
_C.LOGGER.RUN_NAME = None

# ======================================================== #
# Airflow Config                                           #
# ======================================================== #

# Airflow
_C.AIRFLOW = False


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()

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

_C.PCA_ENABLED = False
_C.PCA = CN()
_C.PCA.N_COMPONENTS = 100  # Temp

# ======================================================== #
# Model Config                                             #
# ======================================================== #

# Default : logistic
_C.MODEL_TYPE = "logistic"

_C.LOGISTIC = CN()

_C.XGBOOST = CN()

# ======================================================== #
# Experiment Config                                        #
# ======================================================== #

# Logger
_C.LOGGER = CN()
_C.LOGGER.EXPERIMENT_NAME = "experiment"
_C.LOGGER.RUN_NAME = None


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()

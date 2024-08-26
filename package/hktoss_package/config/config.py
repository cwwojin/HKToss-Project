from yacs.config import CfgNode as CN

_C = CN()

# -----------------------------------------------------------------------------
# Data Config
# -----------------------------------------------------------------------------

_C.DATASET = CN()

_C.DATASET.PATH = "./.data"  # Dataset cache directory
_C.DATASET.SPLITS = [0.8, 0.2]
_C.DATASET.RANDOM_STATE = 42

# -----------------------------------------------------------------------------
# Model Config
# -----------------------------------------------------------------------------

# Default : logistic
_C.MODEL_TYPE = "logistic"

_C.LOGISTIC = CN()

_C.XGBOOST = CN()

# -----------------------------------------------------------------------------
# Experiment Config
# -----------------------------------------------------------------------------

# Logger
_C.LOGGER = CN()
_C.LOGGER.EXPERIMENT_NAME = "experiment"
_C.LOGGER.RUN_NAME = None


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()

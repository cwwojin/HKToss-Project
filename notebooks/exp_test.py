import os
import os.path as path
from argparse import ArgumentParser
from dotenv import load_dotenv

load_dotenv("../.env")

import pandas as pd
from hktoss_package.config import get_cfg_defaults
from hktoss_package.models import LogisticRegressionModel
from hktoss_package.trainers import MLFlowTrainer
from yacs.config import CfgNode as CN

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def _parse_args():
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to the .yaml config file")

    return parser.parse_args()


def load_model(config: CN):
    if config.MODEL_TYPE == "logistic":
        model = LogisticRegressionModel(config)
    else:
        raise NotImplementedError(f"unrecognized model : {config.MODEL_TYPE}")

    return model


def load_dataset(config: CN):
    dataset_df = pd.read_csv(config.DATASET.PATH, low_memory=False).set_index(
        "SK_ID_CURR"
    )
    X = dataset_df.drop(columns=["TARGET"])
    y = dataset_df["TARGET"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.DATASET.SPLITS[-1],
        random_state=config.DATASET.RANDOM_STATE,
        stratify=y,
    )

    # Scaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    args = _parse_args()
    args = vars(args)
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args["config"])

    # load model
    model = load_model(cfg)

    # get dataset
    X_train, X_test, y_train, y_test = load_dataset(cfg)

    # Trainer
    trainer = MLFlowTrainer(tracking_uri=os.environ.get("MLFLOW_TRACKING_URI"))

    # Run experiment
    trainer.run_experiment(
        model=model, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
    )

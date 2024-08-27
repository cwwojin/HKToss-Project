import os
from argparse import ArgumentParser
from dotenv import load_dotenv

load_dotenv("../.env")

import pandas as pd
from hktoss_package.config import get_cfg_defaults
from hktoss_package.trainers import MLFlowTrainer


def _parse_args():
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to the .yaml config file")

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    args = vars(args)
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args["config"])

    trainer = MLFlowTrainer(
        tracking_uri=os.environ.get("MLFLOW_TRACKING_URI"), config=cfg
    )

    # load dataset
    dataset_df = pd.read_csv(cfg.DATASET.PATH, low_memory=False)

    # run experiment
    trainer.run_experiment(dataframe=dataset_df)

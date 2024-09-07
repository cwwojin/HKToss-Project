import os
from argparse import ArgumentParser
import pandas as pd
from dotenv import load_dotenv
from hktoss_package.config import get_cfg_defaults
from hktoss_package.trainers import MLFlowTrainer


def _parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--mode",
        required=False,
        default="development",
        help="PYTHON_ENV, one of `development`, `production`",
    )
    parser.add_argument("--config", required=True, help="path to the .yaml config file")

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    args = vars(args)

    env_path = ".development.env" if args["mode"] == "development" else ".env"
    load_dotenv("../.development.env", override=True)
    if args["mode"] == "development":
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args["config"])

    trainer = MLFlowTrainer(
        tracking_uri=(
            "http://localhost:5001"
            if args["mode"] == "development"
            else os.environ.get("MLFLOW_TRACKING_URI")
        ),
        config=cfg,
    )

    # load dataset
    dataset_df = pd.read_csv(cfg.DATASET.PATH, low_memory=False)

    # run experiment
    trainer.run_experiment(dataframe=dataset_df)

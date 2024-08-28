from hktoss_package.trainers import MLFlowTrainer
import os


def initialize_trainer(**kwargs):
    cfg = kwargs["ti"].xcom_pull(key="config", task_ids="load_config")
    trainer = MLFlowTrainer(
        tracking_uri=os.environ.get("MLFLOW_TRACKING_URI"), config=cfg
    )
    kwargs['ti'].xcom_push(key="trainer_info", value={"tracking_uri": os.environ.get("MLFLOW_TRACKING_URI"), "config": cfg})

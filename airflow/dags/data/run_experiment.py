from hktoss_package.trainers import MLFlowTrainer
from yacs.config import CfgNode


def run_experiment(**kwargs):
    trainer_info = kwargs["ti"].xcom_pull(
        key="trainer_info", task_ids="initialize_trainer"
    )

    # dict로 변환된 config를 CfgNode로 복원
    config_dict = trainer_info["config"]
    config_class = CfgNode(config_dict)  # dict를 CfgNode 객체로 변환

    # 이 정보로 트레이너 다시 초기화
    trainer = MLFlowTrainer(
        tracking_uri=trainer_info["tracking_uri"], config=config_class
    )

    # 데이터셋 로드
    dataset_df = kwargs["ti"].xcom_pull(key="dataset", task_ids="load_dataset")

    trainer.run_experiment(dataframe=dataset_df)

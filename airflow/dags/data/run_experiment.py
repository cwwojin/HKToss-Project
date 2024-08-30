from hktoss_package.trainers import MLFlowTrainer
from hktoss_package.config import get_cfg_defaults
import pandas as pd
import yaml
import os


def _run_experiment(**kwargs):
    # 설정 파일 경로를 직접 사용합니다.
    config_path = kwargs["config_path"]

    # YAML 파일을 읽어옵니다.
    with open(config_path, "r") as file:
        config_data = yaml.safe_load(file)

    # YAML 파일의 내용을 수정합니다.
    config_data["MODEL_TYPE"] = kwargs["model"]
    config_data["DATASET"]["SAMPLER"] = kwargs["sampler"]

    # 기본 설정을 가져옵니다.
    cfg = get_cfg_defaults()

    cfg.MODEL_TYPE = config_data["MODEL_TYPE"]
    cfg.DATASET.SAMPLER = config_data["DATASET"]["SAMPLER"]

    print(cfg.MODEL_TYPE)
    print(cfg.DATASET.SAMPLER)

    # 설정 파일에서 추가 설정을 병합합니다.
    # cfg.merge_from_file(config_path)

    print(cfg)

    # MLFlowTrainer를 초기화합니다.
    trainer = MLFlowTrainer(
        tracking_uri=os.environ.get("MLFLOW_TRACKING_URI"), config=cfg
    )

    # 데이터셋 로드
    temp_csv_path = ".cache/temp_dataset.csv"  # load_dataset 함수에서 생성한 파일 경로와 일치해야 함

    dataset_df = pd.read_csv(temp_csv_path, low_memory=False)

    # 실험 실행
    trainer.run_experiment(dataframe=dataset_df)

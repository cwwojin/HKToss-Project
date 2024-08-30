from hktoss_package.trainers import MLFlowTrainer
from hktoss_package.config import get_cfg_defaults
import pandas as pd
import yaml
import os


def _run_experiment(**kwargs):
    # 기본 설정을 가져옵니다.
    cfg = get_cfg_defaults()

    cfg.MODEL_TYPE = kwargs["model"]
    cfg.DATASET.SAMPLER = kwargs["sampler"]

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

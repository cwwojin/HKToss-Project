import os
from datetime import datetime

import pandas as pd
from hktoss_package.config import get_cfg_defaults
from hktoss_package.trainers import MLFlowTrainer


def _run_experiment(**kwargs):
    # 기본 설정을 가져옵니다.
    cfg = get_cfg_defaults()

    cfg.MODEL_TYPE = kwargs["model"]
    cfg.DATASET.SAMPLER = kwargs["sampler"]

    if cfg.MODEL_TYPE == "randomforest" or cfg.MODEL_TYPE == "xgboost":
        cfg.DATASET.TEST_SIZE = 0.3

    print(cfg.MODEL_TYPE)
    print(cfg.DATASET.SAMPLER)

    print(cfg)

    # MLFlowTrainer를 초기화합니다.
    trainer = MLFlowTrainer(
        tracking_uri=os.environ.get("MLFLOW_TRACKING_URI"), config=cfg
    )

    current_weekday = datetime.now().strftime("%A")

    # 데이터셋 로드
    temp_csv_path = f".cache/{current_weekday}_add_dataset.csv"  # load_dataset 함수에서 생성한 파일 경로와 일치해야 함

    dataset_df = pd.read_csv(temp_csv_path, low_memory=False)

    # 실험 실행
    trainer.run_experiment(dataframe=dataset_df)

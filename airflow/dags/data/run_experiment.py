from hktoss_package.trainers import MLFlowTrainer
from hktoss_package.config import get_cfg_defaults
import pandas as pd
import os

def _run_experiment(**kwargs):
    # 설정 파일 경로를 직접 사용합니다.
    saved_config_path = "/tmp/config.yaml"
    
    # 설정 파일에서 cfg 객체를 다시 로드합니다.
    cfg = get_cfg_defaults()
    cfg.merge_from_file(saved_config_path)
    
    # MLFlowTrainer를 초기화합니다.
    trainer = MLFlowTrainer(
        tracking_uri=os.environ.get("MLFLOW_TRACKING_URI"), 
        config=cfg
    )

    # 데이터셋 로드
    temp_csv_path = ".cache/temp_dataset.csv"  # load_dataset 함수에서 생성한 파일 경로와 일치해야 함
    dataset_df = pd.read_csv(temp_csv_path, low_memory=False)

    trainer.run_experiment(dataframe=dataset_df)

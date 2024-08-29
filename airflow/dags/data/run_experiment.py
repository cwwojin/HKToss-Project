from hktoss_package.trainers import MLFlowTrainer
import os

def _run_experiment(**kwargs):
    # XCom에서 'config'를 불러옵니다.
    cfg = kwargs["ti"].xcom_pull(key="config", task_ids="load_config")
    
    # Config 객체가 제대로 로드되었는지 확인
    if cfg is None:
        raise ValueError("XCom에서 config를 가져오지 못했습니다.")

    print("Config 객체:", cfg)
    print("Grid Search 설정:", cfg.GRID_SEARCH) # 이 부분에서 오류가 발생할 수 있습니다. <- 이거 확인 중
    
    # MLFlowTrainer를 초기화합니다.
    trainer = MLFlowTrainer(
        tracking_uri=os.environ.get("MLFLOW_TRACKING_URI"), 
        config=cfg
    )

    # 데이터셋 로드
    dataset_df = kwargs["ti"].xcom_pull(key="dataset", task_ids="load_dataset")

    trainer.run_experiment(dataframe=dataset_df)

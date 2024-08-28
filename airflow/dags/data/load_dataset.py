import pandas as pd


def load_dataset(**kwargs):
    cfg = kwargs["ti"].xcom_pull(key="config", task_ids="load_config")
    # dataset_df = pd.read_csv(".cache/train_data_cache.pkl", low_memory=False)
    # dataset_df = pd.read_csv(cfg.DATASET.PATH, low_memory=False)

    # Pickle 파일에서 데이터셋을 로드하여 DataFrame으로 변환
    dataset_df = pd.read_pickle(".cache/train_data_cache.pkl")

    # DataFrame을 임시 CSV 파일로 저장
    temp_csv_path = ".cache/temp_dataset.csv"
    dataset_df.to_csv(temp_csv_path, index=False)

    # CSV 파일을 다시 읽어들임
    dataset_df = pd.read_csv(temp_csv_path, low_memory=False)

    kwargs["ti"].xcom_push(key="dataset", value=dataset_df)

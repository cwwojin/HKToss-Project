import pandas as pd


def load_dataset(**kwargs):
    # 데이터셋 로드
    dataset_df = pd.read_pickle(".cache/train_data_cache.pkl")

    # DataFrame을 임시 CSV 파일로 저장
    temp_csv_path = ".cache/temp_dataset.csv"
    dataset_df.to_csv(temp_csv_path, index=False)

    print(f"데이터셋이 {temp_csv_path} 경로에 저장되었습니다.")

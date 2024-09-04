import pandas as pd
from datetime import datetime


def load_dataset(**kwargs):
    current_weekday = datetime.now().strftime("%A")

    # 데이터셋 로드
    dataset_df = pd.read_pickle(f".cache/train_data_cache_{current_weekday}.pkl")

    # DataFrame을 임시 CSV 파일로 저장
    temp_csv_path = f".cache/{current_weekday}_add_dataset.csv"
    dataset_df.to_csv(temp_csv_path, index=False)

    print(f"데이터셋이 {temp_csv_path} 경로에 저장되었습니다.")

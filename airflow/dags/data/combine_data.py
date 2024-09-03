import os
import pandas as pd
from datetime import datetime


def _combine_batches():
    cache_dir = "/opt/airflow/.cache"
    combined_df = pd.DataFrame()

    # .pkl 파일 리스트를 가져옴 (날짜와 배치 번호가 포함된 파일)
    batch_files = [
        os.path.join(cache_dir, f)
        for f in sorted(os.listdir(cache_dir))
        if f.startswith("train_data_cache_batch_") and f.endswith(".pkl")
    ]

    # 각 배치 파일을 읽어서 데이터프레임에 추가
    for batch_file in batch_files:
        df = pd.read_pickle(batch_file)
        combined_df = pd.concat([combined_df, df], ignore_index=True)

    current_weekday = datetime.now().strftime("%A")

    # 통합된 데이터프레임을 하나의 파일로 저장
    combined_file_path = os.path.join(
        cache_dir, f"train_data_cache_{current_weekday}.pkl"
    )
    combined_df.to_pickle(combined_file_path)

    print(f"All batches combined and saved to {combined_file_path}")

    # 배치 파일 제거
    for batch_file in batch_files:
        os.remove(batch_file)
        print(f"Removed batch file: {batch_file}")

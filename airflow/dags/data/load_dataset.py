import pandas as pd
from hktoss_package.config import get_cfg_defaults


def load_dataset(**kwargs):
    # Airflow 태스크 인스턴스에서 제공된 인자들에서 config_path를 가져옵니다.
    data_path = kwargs["data_path"]
    print(data_path)

    # 데이터셋 로드
    dataset_df = pd.read_pickle(data_path)

    # DataFrame을 임시 CSV 파일로 저장
    temp_csv_path = ".cache/temp_dataset.csv"
    dataset_df.to_csv(temp_csv_path, index=False)

    print(f"데이터셋이 {temp_csv_path} 경로에 저장되었습니다.")

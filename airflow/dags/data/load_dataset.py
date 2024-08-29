import pandas as pd
from hktoss_package.config import get_cfg_defaults


def load_dataset(**kwargs):
    # 설정 파일 경로를 직접 사용합니다.
    saved_config_path = "/tmp/config.yaml"
    
    # 설정 파일에서 cfg 객체를 다시 로드합니다.
    cfg = get_cfg_defaults()
    cfg.merge_from_file(saved_config_path)

    dataset_df = pd.read_pickle(cfg.DATASET.PATH)

    # DataFrame을 임시 CSV 파일로 저장
    temp_csv_path = ".cache/temp_dataset.csv"
    dataset_df.to_csv(temp_csv_path, index=False)

    print(f"데이터셋이 {temp_csv_path} 경로에 저장되었습니다.")

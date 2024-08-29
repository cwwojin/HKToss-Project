from hktoss_package.config import get_cfg_defaults
from yacs.config import CfgNode as CN

def load_config(**kwargs):
    # Airflow 태스크 인스턴스에서 제공된 인자들에서 config_path를 가져옵니다.
    config_path = kwargs["config_path"]
    
    # 기본 설정을 가져옵니다.
    cfg = get_cfg_defaults()
    
    # 설정 파일에서 추가 설정을 병합합니다.
    cfg.merge_from_file(config_path)

    # 여기서 DataSampler와 Model_Type 설정을 변경합니다.
    cfg.DATASET.SAMPLER = kwargs["sampler"]
    cfg.MODEL_TYPE = kwargs["model"]

    # 설정을 XCom에 저장하여 후속 태스크들이 사용할 수 있도록 합니다.
    kwargs["ti"].xcom_push(key="config", value=cfg)
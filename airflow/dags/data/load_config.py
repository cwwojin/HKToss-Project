from hktoss_package.config import get_cfg_defaults

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

    # 변경된 설정을 파일로 저장합니다.
    saved_config_path = "/tmp/config.yaml"
    with open(saved_config_path, "w") as f:
        f.write(cfg.dump())  # yacs CfgNode 객체를 YAML로 저장

    print(f"Config 파일이 {saved_config_path} 경로에 저장되었습니다.")
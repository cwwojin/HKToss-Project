from hktoss_package.config import get_cfg_defaults


def load_config(**kwargs):
    config_path = kwargs["config_path"]
    cfg = get_cfg_defaults()
    cfg.merge_from_file(config_path)
    kwargs["ti"].xcom_push(key="config", value=cfg)

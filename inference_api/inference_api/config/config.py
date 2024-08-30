from pydantic_settings import BaseSettings
import os
from dotenv import load_dotenv

load_dotenv(".env")


class Config(BaseSettings):
    app_name: str = "inference_api"
    api_key: str | None = os.environ.get("INFERENCE_API_KEY")
    mlflow_tracking_uri: str | None = os.environ.get("MLFLOW_TRACKING_URI")
    mlflow_db_uri: str | None = os.environ.get("DB_URI")
    aws_access_key_id: str | None = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str | None = os.environ.get("AWS_SECRET_ACCESS_KEY")
    aws_default_region: str | None = os.environ.get("AWS_DEFAULT_REGION")
    aws_bucket_name: str | None = os.environ.get("AWS_BUCKET_NAME")


config = Config()

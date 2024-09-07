import os

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

env_path = (
    ".development.env" if os.environ.get("PYTHON_ENV") == "development" else ".env"
)
load_dotenv(env_path, override=True)


class Config(BaseSettings):
    python_env: str | None = os.environ.get("PYTHON_ENV")
    app_name: str = "inference_api"
    api_key: str | None = os.environ.get("INFERENCE_API_KEY")
    mlflow_tracking_uri: str | None = os.environ.get("MLFLOW_TRACKING_URI")
    mlflow_db_uri: str | None = os.environ.get("DB_URI")
    aws_access_key_id: str | None = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str | None = os.environ.get("AWS_SECRET_ACCESS_KEY")
    aws_default_region: str | None = os.environ.get("AWS_DEFAULT_REGION")
    aws_bucket_name: str | None = os.environ.get("AWS_BUCKET_NAME")
    s3_endpoint: str | None = os.environ.get("MLFLOW_S3_ENDPOINT_UR")


config = Config()

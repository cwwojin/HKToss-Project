import os
import os.path as path

import streamlit as st
from boto3 import client
from pandas import read_csv

# Init cache directory
DATA_PATH = path.join(path.dirname(__file__), "../.cache")
DEMO_PATH = path.join(DATA_PATH, "dataset_demo.csv")
TOTAL_PATH = path.join(DATA_PATH, "dataset_total.csv")
if not path.isdir(DATA_PATH):
    os.makedirs(DATA_PATH, exist_ok=True)


@st.cache_data(show_spinner=False)
def download_data():
    if not path.isfile(DEMO_PATH):
        s3 = client(
            "s3",
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            region_name=os.environ.get("AWS_REGION"),
            endpoint_url=(
                os.environ.get("MLFLOW_S3_ENDPOINT_URL")
                if os.environ.get("PYTHON_ENV") == "development"
                else None
            ),
        )
        s3.download_file(
            "hktoss-mlops",
            "datasets/dataset_demo.csv",
            DEMO_PATH,
        )
        s3.close()

    if not path.isfile(TOTAL_PATH):
        s3 = client(
            "s3",
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            region_name=os.environ.get("AWS_REGION"),
        )
        s3.download_file(
            "hktoss-mlops",
            "datasets/dataset_total.csv",
            TOTAL_PATH,
        )
        s3.close()
    return None


@st.cache_data(show_spinner=False)
def load_demo_data():
    df = read_csv(DEMO_PATH, low_memory=False)
    return df


@st.cache_data(show_spinner=False)
def load_total_data():
    df = read_csv(TOTAL_PATH, low_memory=False)
    return df

from datetime import datetime
from airflow import DAG

defalut_args = {"start_date": datetime(2024, 1, 1)}

with DAG(
    dag_id="hktoss-project-pipeline",
    schedule_interval="@daily",
    default_args=defalut_args,
    tags=["hktoss", "project", "pipeline"],
    catchup=False,
) as dag:
    pass

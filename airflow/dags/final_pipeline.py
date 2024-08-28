from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.utils.task_group import TaskGroup

from data.fetch_data import _fetch_data
from data.load_config import load_config
from data.load_dataset import load_dataset
from data.load_model import initialize_trainer
from data.run_experiment import run_experiment
from data.combine_data import _combine_batches

defalut_args = {"start_date": datetime(2024, 1, 1)}

with DAG(
    dag_id="hktoss-project-final-pipeline",
    schedule_interval="@daily",
    default_args=defalut_args,
    tags=["hktoss", "project", "pipeline"],
    catchup=False,
) as dag:

    # 1. 데이터를 한 번만 가져오는 작업
    fetch_data = PythonOperator(
        task_id="fetch_data",
        python_callable=_fetch_data,
        execution_timeout=timedelta(minutes=10),
    )

    # 2. 데이터를 결합하는 작업, 이 작업도 한 번만 실행
    combine_data = PythonOperator(
        task_id="combine_data", python_callable=_combine_batches
    )

    # 3. ExternalTaskSensor를 사용하여 combine_data 작업이 완료되었는지 확인
    combine_data_sensor = ExternalTaskSensor(
        task_id="combine_data_sensor",
        external_dag_id="hktoss-project-final-pipeline",
        external_task_id="combine_data",
        mode="reschedule",
        timeout=600,
    )

    load_config_task = PythonOperator(
        task_id="load_config",
        python_callable=load_config,
        op_kwargs={"config_path": "/opt/airflow/config/config.yaml"},
    )

    initialize_trainer_task = PythonOperator(
        task_id="initialize_trainer",
        python_callable=initialize_trainer,
    )

    load_dataset_task = PythonOperator(
        task_id="load_dataset",
        python_callable=load_dataset,
    )

    run_experiment_task = PythonOperator(
        task_id="run_experiment",
        python_callable=run_experiment,
    )

    fetch_data >> combine_data >> combine_data_sensor
    (
        combine_data_sensor
        >> load_config_task
        >> initialize_trainer_task
        >> load_dataset_task
        >> run_experiment_task
    )

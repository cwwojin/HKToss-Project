from datetime import datetime, timedelta
from airflow import DAG
from itertools import product
from airflow.operators.python import PythonOperator
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.utils.task_group import TaskGroup

from data.fetch_data import _fetch_data
from data.load_dataset import load_dataset
from data.run_experiment import _run_experiment
from data.combine_data import _combine_batches


samplers = ["over_random", "over_smote", "under_random", None]
models = ["randomforest", "logistic", "xgboost", "catboost", "lightgbm", "mlp"]

# 모델과 샘플러의 모든 조합 생성
combinations = list(product(models, samplers))

default_args = {"start_date": datetime(2024, 1, 1)}

with DAG(
    dag_id="hktoss-project-final-pipeline",
    schedule_interval="@daily",
    default_args=default_args,
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

    fetch_data >> combine_data >> combine_data_sensor

    previous_group = None  # 이전 그룹을 추적하기 위한 변수

    for i, (model, sampler) in enumerate(combinations):

        sampler_str = sampler if sampler is not None else "none"

        with TaskGroup(group_id=f"group_{i+1}") as model_experiment:
            # 각 조합에 대해 load_dataset_task 생성
            load_dataset_task = PythonOperator(
                task_id=f"load_dataset_{model}_{sampler_str}",
                python_callable=load_dataset,
            )

            # 각 조합에 대해 run_experiment_task 생성
            run_experiment_task = PythonOperator(
                task_id=f"run_experiment_{model}_{sampler}",
                python_callable=_run_experiment,
                op_kwargs={
                    "model": model,
                    "sampler": sampler,
                },
            )

            # 그룹 내에서 Task 의존성 설정
            load_dataset_task >> run_experiment_task

        # 그룹 간 의존성 설정 (순차 실행 보장)
        if previous_group:
            previous_group >> model_experiment
        else:
            combine_data_sensor >> model_experiment

        previous_group = model_experiment  # 현재 그룹을 이전 그룹으로 설정

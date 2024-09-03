from datetime import datetime, timedelta
from airflow import DAG
import os
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup
from airflow.models import Variable

from data.import_to_mongo import _import_data
from data.fetch_data import _fetch_data
from data.load_dataset import load_dataset
from data.run_experiment import _run_experiment
from data.combine_data import _combine_batches

samplers = ["over_smote", "over_random", "under_random", None]
models = ["lightgbm", "catboost", "xgboost", "randomforest", "logistic", "mlp"]


# 모델과 샘플러의 모든 조합 생성
def get_csv_file_path(**kwargs):
    # 현재 날짜를 기반으로 요일을 구함 (일요일=0, ..., 토요일=6)
    day_of_week = datetime.now().weekday()
    file_path = f"/opt/airflow/dags/data/.tmp/dataset_train_sub_split_{(day_of_week + 1) % 7}.csv"

    # 파일 존재 여부 확인
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    return file_path


default_args = {"start_date": datetime(2024, 1, 1)}

with DAG(
    dag_id="hktoss-project-final-pipeline",
    schedule_interval="@daily",
    default_args=default_args,
    tags=["hktoss", "project", "pipeline"],
    catchup=False,
) as dag:

    get_csv_file_path_task = PythonOperator(
        task_id="get_csv_file_path",
        python_callable=get_csv_file_path,
    )

    import_data_task = PythonOperator(
        task_id="import_data",
        python_callable=_import_data,
        op_kwargs={
            "csv_file_path": "{{ task_instance.xcom_pull(task_ids='get_csv_file_path') }}",  # 동적으로 파일 경로를 설정
            "unique_key": "SK_ID_CURR",
        },
        execution_timeout=timedelta(minutes=10),
    )

    fetch_data = PythonOperator(
        task_id="fetch_data",
        python_callable=_fetch_data,
        execution_timeout=timedelta(minutes=10),
    )

    combine_data = PythonOperator(
        task_id="combine_data", python_callable=_combine_batches
    )

    load_dataset_task = PythonOperator(
        task_id="load_dataset",
        python_callable=load_dataset,
    )

    # 작업 순서 설정
    (
        get_csv_file_path_task
        >> import_data_task
        >> fetch_data
        >> combine_data
        >> load_dataset_task
    )

    # 모델과 샘플러 조합에 대한 작업 그룹 생성 및 설정
    previous_task = load_dataset_task

    for model in models:
        with TaskGroup(group_id=f"group_{model}") as model_group:
            for i in range(0, len(samplers), 2):  # 2개씩 묶어 처리
                tasks = []
                with TaskGroup(group_id=f"parallel_{model}_{i}") as parallel_group:
                    for sampler in samplers[i : i + 2]:
                        sampler_str = sampler if sampler is not None else "none"

                        run_experiment_task = PythonOperator(
                            task_id=f"run_experiment_{model}_{sampler_str}",
                            python_callable=_run_experiment,
                            op_kwargs={
                                "model": model,
                                "sampler": sampler,
                            },
                        )

                        tasks.append(run_experiment_task)

                    # 두 개의 작업을 병렬로 실행
                    for task in tasks:
                        previous_task >> task

                previous_task = tasks[
                    -1
                ]  # 마지막 병렬 작업을 다음의 이전 작업으로 설정

        # 모델 그룹이 순차적으로 실행되도록 설정
        load_dataset_task >> model_group

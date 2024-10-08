import os
import os.path as path
from datetime import datetime, timedelta

from airflow.models import Variable
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup
from data.combine_data import _combine_batches
from data.fetch_data import _fetch_data
from data.import_to_mongo import _import_data
from data.load_dataset import load_dataset
from data.poll_collection import poll_mongo_collection
from data.run_experiment import _run_experiment
from dotenv import load_dotenv

from airflow import DAG

os.chdir(path.dirname(__file__))
env_path = (
    ".development.env" if os.environ.get("PYTHON_ENV") == "development" else ".env"
)
load_dotenv(f"../{env_path}", override=True)

samplers = [
    "composite",
    None,
    "over_smote",
    "over_random",
    "under_random",
]
models = ["lightgbm", "catboost", "xgboost", "randomforest", "logistic", "mlp"]


# 모델과 샘플러의 모든 조합 생성
def get_csv_file_path(**kwargs):
    # 현재 날짜를 기반으로 요일을 구함 (일요일=0, ..., 토요일=6)
    day_of_week = datetime.now().weekday()
    file_path = f"data/.tmp/dataset_train_sub_split_{(day_of_week + 1) % 7}.csv"

    # 파일 존재 여부 확인
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    return file_path


def run_poll_mongo_collection(**kwargs):
    # 이전에 저장된 last_checked_id 불러오기 (없으면 None)
    last_checked_id = Variable.get("last_checked_id", default_var=None)

    # MongoDB 컬렉션 폴링
    new_last_checked_id = poll_mongo_collection(last_checked_id)

    # 새로운 last_checked_id가 있으면 저장
    if new_last_checked_id:
        Variable.set("last_checked_id", str(new_last_checked_id))


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

    poll_mongo_collection_task = PythonOperator(
        task_id="poll_mongo_collection_task",
        python_callable=run_poll_mongo_collection,  # Airflow에서 호출할 함수
        execution_timeout=timedelta(minutes=5),
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
        >> poll_mongo_collection_task
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

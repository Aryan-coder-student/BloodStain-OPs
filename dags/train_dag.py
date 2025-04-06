from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2025, 4, 6),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "blood_detection_train_dag",
    default_args=default_args,
    schedule_interval="@daily",
    catchup=False,
) as dag:
    download_task = BashOperator(
        task_id="download_data",
        bash_command="python /opt/airflow/src/download_data.py",
    )

    train_task = BashOperator(
        task_id="train_model",
        bash_command="python /opt/airflow/src/train_model.py",
    )

    dvc_push_task = BashOperator(
        task_id="dvc_push",
        bash_command="dvc push",
    )

    download_task >> train_task >> dvc_push_task
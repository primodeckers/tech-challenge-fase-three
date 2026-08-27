from datetime import datetime
from pathlib import Path
import os
import sys

from airflow import DAG
from airflow.operators.python import PythonOperator

PROJECT = Path(os.environ.get("PROJECT_ROOT", "/opt/airflow/project"))


def load_data():
    import pandas as pd

    path = PROJECT / "data" / "laudos.csv"
    df = pd.read_csv(path)
    if "texto" not in df.columns or "label" not in df.columns:
        raise ValueError("csv sem colunas texto/label")
    if len(df) < 2000:
        raise ValueError(f"dataset pequeno: {len(df)}")
    print(f"linhas={len(df)}")


def train_model():
    sys.path.insert(0, str(PROJECT))
    from model.train import treinar

    treinar()


def save_model():
    path = PROJECT / "artifacts" / "model.joblib"
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    print(f"modelo ok: {path} ({path.stat().st_size} bytes)")


with DAG(
    dag_id="treino_laudos",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
) as dag:
    carrega = PythonOperator(task_id="load_data", python_callable=load_data)
    treina = PythonOperator(task_id="train", python_callable=train_model)
    salva = PythonOperator(task_id="save_model", python_callable=save_model)
    carrega >> treina >> salva

"""
Airflow DAGs:

1. insurance_rag_etl        Daily: regenerate mock data -> build vector index
2. hallucination_harness_eval  Daily, separately: run the labeled eval set
                                through the harness and log metrics
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    "owner": "data-engineering",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="insurance_rag_etl",
    default_args=default_args,
    description="Daily ETL pipeline: ingest -> transform -> embed -> store",
    schedule_interval="0 2 * * *",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["insurance", "rag", "etl"],
) as etl_dag:

    def task_generate_mock_data(**context):
        from ingestion.mock_data import generate_claims, generate_customers, generate_policies, write_csv
        write_csv(generate_policies(), "policies.csv")
        write_csv(generate_claims(), "claims.csv")
        write_csv(generate_customers(), "customers.csv")

    def task_embed_and_store(**context):
        from etl.chunk import all_chunks
        from etl.transform import run_all_transforms
        from embeddings.generate import embed_chunks
        from embeddings.store import upsert_chunks
        from ingestion.loader import load_all

        raw = load_all()
        transformed = run_all_transforms(raw)
        chunks = all_chunks(transformed)
        embedded = embed_chunks(chunks)
        upsert_chunks(embedded)

    generate = PythonOperator(task_id="generate_mock_data", python_callable=task_generate_mock_data)
    embed_store = PythonOperator(task_id="embed_and_store", python_callable=task_embed_and_store)

    generate >> embed_store  # this arrow means "embed_store runs after generate finishes"


with DAG(
    dag_id="hallucination_harness_eval",
    default_args=default_args,
    description="Daily: run the labeled eval set through the harness, log metrics",
    schedule_interval="0 4 * * *",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["insurance", "rag", "eval", "harness"],
) as eval_dag:

    def task_run_eval(**context):
        from harness.eval_set import load_eval_set, run_eval
        summary = run_eval(load_eval_set())
        context["ti"].xcom_push(key="eval_summary", value={
            k: v for k, v in summary.items() if k != "results"
        })

    run_eval_task = PythonOperator(task_id="run_eval", python_callable=task_run_eval)
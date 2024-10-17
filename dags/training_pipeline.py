from __future__ import annotations
import json
from textwrap import dedent
import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator
from src.gem_price_prediction.pipeline.training_pipeline import Training_Pipeline
import numpy as np

# Initialize the training pipeline
training_pipeline = Training_Pipeline()

with DAG(
        "gemstone_training_pipeline",
        default_args={"retries": 2},
        description="It's my training pipeline",
        schedule_interval="@weekly",
        start_date=pendulum.datetime(2024, 10, 15, tz=pendulum.tz.UTC),
        catchup=False,
        tags=["Machine_learning", "Gemstone", "Training_Pipeline"]
) as dag:
    dag.doc_md = __doc__


    def data_ingestion(**kwargs):
        ti = kwargs["ti"]
        train_data_path, test_data_path = training_pipeline.start_data_Ingestion()

        ti.xcom_push("data_ingestion_artifact", {"train_data_path": train_data_path, "test_data_path": test_data_path})


    def data_transformation(**kwargs):
        ti = kwargs["ti"]
        data_ingestion_artifact = ti.xcom_pull(task_ids="data_ingestion", key="data_ingestion_artifact")

        train_arr, test_arr = training_pipeline.start_data_transformation(
            data_ingestion_artifact["train_data_path"],
            data_ingestion_artifact["test_data_path"]
        )

        train_arr = train_arr.to_list()
        test_arr = test_arr.to_list()  # Corrected the hyphen to equals
        ti.xcom_push("data_transformation_artifact", {"train_arr": train_arr, "test_arr": test_arr})


    def model_trainer(**kwargs):
        ti = kwargs["ti"]
        data_transformation_artifact = ti.xcom_pull(task_ids="data_transformation", key="data_transformation_artifact")

        train_arr = np.array(data_transformation_artifact["train_arr"])
        test_arr = np.array(data_transformation_artifact["test_arr"])
        training_pipeline.start_model_training(train_arr, test_arr)


    # Data Ingestion Task
    data_ingestion_task = PythonOperator(
        task_id="data_ingestion",
        python_callable=data_ingestion,
    )
    data_ingestion_task.doc_md = dedent(
        """\
        ### Ingestion Task
        This task creates a train and test file.
        """
    )

    # Data Transformation Task
    data_transformation_task = PythonOperator(
        task_id="data_transformation",
        python_callable=data_transformation,
    )
    data_transformation_task.doc_md = dedent(
        """\
        ### Data Transformation Task
        This task performs data transformation.
        """
    )

    # Model Training Task
    model_trainer_task = PythonOperator(
        task_id="model_trainer",
        python_callable=model_trainer,
    )
    model_trainer_task.doc_md = dedent(
        """\
        ### Model Training Task
        This task performs model training.
        """
    )

    # Task Dependencies
    data_ingestion_task >> data_transformation_task >> model_trainer_task

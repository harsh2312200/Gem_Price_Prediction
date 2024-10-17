Here’s the updated `README.md` file with the **S3** parts removed:

---

# Gem Price Prediction Project

This project aims to predict the prices of gems based on various features using machine learning models. The project incorporates multiple stages, including data ingestion, preprocessing, model training, evaluation, and prediction, all orchestrated using **Airflow** for continuous integration and automation. 

## Table of Contents
- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Tools and Technologies](#tools-and-technologies)
- [Airflow Pipelines](#airflow-pipelines)
  - [Training Pipeline](#training-pipeline)
  - [Prediction Pipeline](#prediction-pipeline)
- [Modeling](#modeling)
- [Setup](#setup)
- [Future Enhancements](#future-enhancements)

## Project Overview

The goal of the project is to build a machine learning model that accurately predicts the price of gems using historical data. We have implemented a modular and scalable workflow for both training and prediction pipelines using **Airflow**, along with model versioning and logging using **MLflow**.

## Architecture

The project is split into two main pipelines:
1. **Training Pipeline**: Responsible for training the model using historical data.
2. **Prediction Pipeline**: Responsible for generating predictions using the trained model on new data.

These pipelines are containerized, and we use shared resources (such as Docker volumes) to store models so that the prediction pipeline can access the most recent model trained by the training pipeline.

## Tools and Technologies

### Workflow Automation:
- **Airflow**: Used for automating, orchestrating, and scheduling the training and prediction pipelines.
  - DAG location: `airflow/dag/training_pipeline.py` (Training)
  - DAG location: `airflow/dag/prediction_pipeline.py` (Prediction)

### Machine Learning:
- **Scikit-learn**: Utilized for implementing multiple machine learning algorithms such as:
  - Linear Regression
  - Elastic Net
  - Lasso
  - Ridge
  - XGBRegressor
- **MLflow**: For logging parameters, metrics, and model artifacts. The best model is selected and registered for future predictions.

### Model Deployment and Sharing:
- **Docker**: The training and prediction pipelines are containerized using Docker, ensuring portability and scalability. A shared Docker volume is used for model sharing between the pipelines.

### Version Control & Tracking:
- **Git**: Source control and versioning of code.
- **DVC (Data Version Control)**: For tracking data, pipelines, and models.

### Task Automation:
- **Airflow**: Each stage of the pipeline (data ingestion, preprocessing, training, and prediction) is automated using Airflow, enabling continuous integration and model retraining.

## Airflow Pipelines

### Training Pipeline

The **training pipeline** is responsible for:
1. Data ingestion
2. Preprocessing
3. Model training
4. Model evaluation
5. Saving the model to a shared resource (local Docker volume)

After the model is trained, it is saved for future use by the prediction pipeline. MLflow is integrated for logging parameters and metrics during the training process.

### Prediction Pipeline

The **prediction pipeline** includes:
1. Data ingestion
2. Preprocessing
3. Loading the latest model from the shared resource (Docker volume)
4. Making predictions

This pipeline ensures that predictions are generated based on the most recently trained model.

## Modeling

We trained various machine learning models using **Scikit-learn** and evaluated them to select the best-performing one. The models we considered include:
- **Linear Regression**
- **Elastic Net**
- **Lasso**
- **Ridge**
- **XGBRegressor**

The **best model** is saved and registered using **MLflow** to ensure that it can be used in the prediction pipeline.

## Setup

### Prerequisites
- **Python 3.7+**
- **Docker** (for containerization)
- **Airflow** (for pipeline orchestration)

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Set Up Airflow
1. Initialize the Airflow database:
   ```bash
   airflow db init
   ```

2. Set up your DAGs by placing them in the `airflow/dags/` directory:
   - `training_pipeline.py`
   - `prediction_pipeline.py`

3. Start the Airflow webserver and scheduler:
   ```bash
   airflow webserver --port 8080
   airflow scheduler
   ```

### Docker Setup
Build and run the containers for your training and prediction pipelines. Example:
```bash
docker-compose up --build
```

### MLflow Setup
1. Set up an MLflow tracking server (optional):
   ```bash
   mlflow ui
   ```

2. Use MLflow within your training pipeline to log parameters, metrics, and models.

## Future Enhancements

- **CI/CD Pipeline**: Integrating a CI/CD pipeline to automate deployment of model updates.
- **Model Monitoring**: Set up model monitoring to track model drift and retrain automatically when performance degrades.
- **Hyperparameter Tuning**: Automating hyperparameter tuning with tools like Hyperopt or Optuna.

---

## Contact

For questions or suggestions, feel free to reach out:
- **Name**: Harsh Kadam
- **Email**: harshkadam997@gmail.com
- **LinkedIn**: [linkedin.com/in/harsh-kadam-198511292/](https://linkedin.com/in/harsh-kadam-198511292/)
- **GitHub**: [github.com/harsh2312200](https://github.com/harsh2312200)

---

This README now reflects a local storage setup with Docker volumes for model sharing. Let me know if you'd like to add anything else!

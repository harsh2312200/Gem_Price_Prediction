import os
import sys
import mlflow
import mlflow.sklearn
import numpy as np
from src.gem_price_prediction.logger.logger import logging
from src.gem_price_prediction.exception.exception import customexception
from dataclasses import dataclass
from src.gem_price_prediction.utils.utils import save_obj, evaluate_model
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


def create_experiment(experiment_name, run_name, run_metrics, model, confusion_matrix_path=None,
                      roc_auc_plot_path=None, run_params=None):
    # mlflow.set_tracking_uri("http://localhost:5000")
    # use above line if you want to use any database like sqlite as backend storage for model else comment this line
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):

        if not run_params == None:
            for param in run_params:
                mlflow.log_param(param, run_params[param])

        for metric in run_metrics:
            mlflow.log_metric(metric, run_metrics[metric])

        if not confusion_matrix_path == None:
            mlflow.log_artifact(confusion_matrix_path, 'confusion_materix')

        if not roc_auc_plot_path == None:
            mlflow.log_artifact(roc_auc_plot_path, "roc_auc_plot")

        mlflow.set_tag("tag1", "Gem Price Prediction")
        mlflow.set_tags({"tag2": "Logistic Regression", "tag3": "Multiclassification using Ovr - One vs rest class"})
        mlflow.sklearn.log_model(model, "model")
    print('Run - %s is logged to Experiment - %s' % (run_name, experiment_name))
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
from src.gem_price_prediction.components.model_evaluation import create_experiment

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info("Splitting dependent and independent variables from data")
            x_train, y_train, x_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Define models to train
            models = {
                "Linear_Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "ElasticNet": ElasticNet(),
                "RandomForest": RandomForestRegressor(),
                "xgboost": XGBRegressor()
            }

            # Evaluate models
            model_report: dict = evaluate_model(x_train, y_train, x_test, y_test, models)

            logging.info(f"Model Report : \n {model_report}")

            # Iterate over models, logging each model's metrics and model separately
            for model_name, metrics in model_report.items():
                experiment_name = "gem_price_prediction_" + str(model_name)  ##basic classifier
                run_name = str(model_name)
                create_experiment(experiment_name, run_name, metrics, models[model_name])

                # Log and register the model in MLflow

                logging.info(f"Logged model {model_name} in MLflow with metrics: {metrics}")

        # Find the best model based on R² score
            best_model_score = max([metrics["r2_score"] for metrics in model_report.values()])
            best_model_name = [name for name, metrics in model_report.items() if metrics["r2_score"] == best_model_score][0]

            best_model = models[best_model_name]


            logging.info(f"Best Model Found: {best_model_name}, R² Score: {best_model_score}")

            # Save the best model locally
            save_obj(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

        except Exception as e:
            logging.info("Exception occurred during Model Training")
            raise customexception(e, sys)


if __name__ == "__main__":
    obj = ModelTrainer()
    obj.initiate_model_training("artifacts/train_arr.csv", "artifacts/test_arr.csv")

import numpy as np
import pandas as pd
import os
import sys
from src.gem_price_prediction.logger.logger import logging
from src.gem_price_prediction.exception.exception import customexception
from dataclasses import dataclass
from pathlib import Path
from src.gem_price_prediction.utils.utils import save_obj, evaluate_model
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

@dataclass
class ModelTrainerConfig :
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info("splitting Dependent and independent variables from data")
            x_train, y_train, x_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Linear_Regression":LinearRegression(),
                "Lasso":Lasso(),
                "Ridge":Ridge(),
                "ElasticNet":ElasticNet(),
                "RandomForest": RandomForestRegressor(),
                "xgboost": XGBRegressor()
            }

            model_report : dict = evaluate_model(x_train,y_train,x_test,y_test,models)

            logging.info(f"Model Report : \n {model_report}")

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')

            save_obj(
            file_path=self.model_trainer_config.trained_model_file_path,
            obj=best_model
            )

        except Exception as e:
            logging.info("Exception occurred in Model Training")
            raise customexception(e,sys)




if __name__=="__main__":
    obj = ModelTrainer()
    obj.initiate_model_training("artifacts/train_arr.csv","artifacts/test_arr.csv")
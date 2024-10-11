import os
import sys
import pickle
import numpy as np
import pandas as pd
from src.gem_price_prediction.logger.logger import logging
from src.gem_price_prediction.exception.exception import customexception
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def save_obj (file_path,obj):
    logging.info(f"Saving object initiated")
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open (file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)

    except Exception as e:
        logging.info(f"Error Occurred while saving {obj}")
        raise customexception(e,sys)

def load_obj(file_path):
    try:
        logging.info("Loading object started")
        with open(file_path,"rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        logging.info("Exception occurred in load_object function utils")
        raise customexception(e,sys)


def evaluate_model(x_train, y_train, x_test, y_test, models):
    try:
        logging.info("Model training started")
        training_model = []
        report = {}
        for model_name, model in models.items():
            model.fit(x_train, y_train)
            training_model.append(model_name)

            # Make predictions on the test data
            y_test_pred = model.predict(x_test)

            # Calculate evaluation metrics
            r2 = r2_score(y_test, y_test_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            mae = mean_absolute_error(y_test, y_test_pred)

            # Log the metrics into the report dictionary for each model
            report[model_name] = {
                "r2_score": r2,
                "rmse": rmse,
                "mae": mae
            }

        # Create a DataFrame for logging purposes (optional)
        scores_df = pd.DataFrame(report).T  # Transpose the dict to a DataFrame for readability
        logging.info(f"Model Scores after training: \n{scores_df}")

        return report

    except Exception as e:
        logging.info("Exception occurred during model evaluation")
        raise customexception(e, sys)




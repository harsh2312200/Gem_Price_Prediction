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


def evaluate_model(x_train,y_train,x_test,y_test,models):
    try :
        logging.info("Model training started")
        training_model = []
        report = {}
        r2= []
        for i in range(len(models)):
            model = list(models.values())[i]
            model.fit(x_train,y_train)
            training_model.append(list(models.keys())[i])
            y_test_pred = model.predict(x_test)

            test_model_score = r2_score(y_test,y_test_pred)
            r2.append(test_model_score)

            report[list(models.keys())[i]] = test_model_score

        training_model = pd.Series(training_model)
        r2 = pd.Series(r2)
        scores = pd.concat([training_model, r2], axis=1, ignore_index=True)
        scores.columns = ['Model', 'r2_score']

        logging.info(f"Model Scores after training : \n {scores}")
        return report

    except Exception as e:
        logging.info("Exception Occurred during the model training")
        raise customexception(e,sys)







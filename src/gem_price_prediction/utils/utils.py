import os
import sys
import pickle
import numpy as np
import pandas as pd
from src.gem_price_prediction.logger.logger import logging
from src.gem_price_prediction.exception.exception import customexception
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def save_obj (file_path,obj):
    logging.info(f"Saving {obj} initiated")
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






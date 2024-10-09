import pandas as pd
import numpy as np
import os
import sys
from src.gem_price_prediction.logger.logger import logging
from src.gem_price_prediction.utils.utils import load_obj
from src.gem_price_prediction.exception.exception import customexception

class PredictPipeline:
    def __init__(self):
        pass

    def initiate_predict(self,features):
        try:
            logging.info("Prediction Pipeline started")
            preprocessor_path = os.path.join("artifacts","preprocessor.pkl")
            model_path = os.path.join("artifacts","model.pkl")

            preprocessor = load_obj(preprocessor_path)
            model = load_obj(model_path)

            scaled_features = preprocessor.transform(features)
            pred = model.predict(scaled_features)

            return pred

        except Exception as e:
            logging.info("Error Occurred in Prediction Pipeline inside initiate_predict")
            raise customexception (e,sys)


class CustomData:
    def __init__(self,
                 carat: float,
                 depth: float,
                 table: float,
                 x: float,
                 y: float,
                 z: float,
                 cut: str,
                 color: str,
                 clarity: str):

        self.carat = carat
        self.depth = depth
        self.table = table
        self.x = x
        self.y = y
        self.z = z
        self.cut = cut
        self.color = color
        self.clarity = clarity

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'carat': [self.carat],
                'depth': [self.depth],
                'table': [self.table],
                'x': [self.x],
                'y': [self.y],
                'z': [self.z],
                'cut': [self.cut],
                'color': [self.color],
                'clarity': [self.clarity]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df

        except Exception as e:
            logging.info('Exception Occurred in prediction pipeline')
            raise customexception(e, sys)



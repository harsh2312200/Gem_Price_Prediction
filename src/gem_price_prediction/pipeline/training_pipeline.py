import numpy as np
import pandas as pd
import os
import sys
from src.gem_price_prediction.components.data_ingestion import DataIngestion
from src.gem_price_prediction.components.data_transformation import DataTransformation
from src.gem_price_prediction.components.model_trainer import ModelTrainer
from src.gem_price_prediction.components.model_evaluation import create_experiment
from src.gem_price_prediction.logger.logger import logging
from src.gem_price_prediction.exception.exception import customexception

class Training_Pipeline:
    def start_data_Ingestion(self):
        try :
            data_ingestion = DataIngestion()
            train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
            return train_data_path,test_data_path

        except Exception as e:
            logging.info("Error Occurred in training_pipeline -> start_data_ingestion")
            raise customexception(e,sys)

    def start_data_transformation(self,train_data_path,test_data_path):
        try :
            data_transformation = DataTransformation()
            train_arr, test_arr = data_transformation.initiate_data_transformation(train_data_path,test_data_path)
            return train_arr, test_arr

        except Exception as e:
            logging.info("Error occurred in training_pipeline -> start_data_transformation")
            raise customexception(e,sys)

    def start_model_training(self,train_arr,test_arr):
        try :
            model_trainer = ModelTrainer()
            model_trainer.initiate_model_training(train_arr,test_arr)

        except Exception as e:
            logging.info("Error Occurred in training_pipeline -> start_model_training")
            raise customexception(e,sys)


    def start_training(self):
        try :
            train_data_path,test_data_path = self.start_data_Ingestion()
            train_arr,test_arr = self.start_data_transformation(train_data_path,test_data_path)
            self.start_model_training(train_arr,test_arr)

        except Exception as e:
            logging.info("Error Occurred in training_pipeline -> start_training")
            raise customexception(e,sys)

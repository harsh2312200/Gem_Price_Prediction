import numpy as np
import pandas as pd
import os
import sys
from src.gem_price_prediction.logger.logger import logging
from src.gem_price_prediction.exception.exception import customexception
from src.gem_price_prediction.components.data_ingestion import DataIngestion
from src.gem_price_prediction.components.data_transformation import DataTransformation
from src.gem_price_prediction.components.model_trainer import ModelTrainer
from src.gem_price_prediction.utils.utils import save_obj,evaluate_model

obj = DataIngestion()
train_data_path,test_data_path = obj.initiate_data_ingestion()

data_transformation = DataTransformation()
train_arr,test_arr = data_transformation.initiate_data_transformation(train_data_path,test_data_path)

model_trainer_obj = ModelTrainer()
model_trainer_obj.initiate_model_training(train_arr,test_arr)


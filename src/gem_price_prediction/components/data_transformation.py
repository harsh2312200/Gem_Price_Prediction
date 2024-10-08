import pandas as pd
import numpy as np
import os
import sys
from src.gem_price_prediction.logger.logger import logging
from src.gem_price_prediction.exception.exception import customexception
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from src.gem_price_prediction.utils.utils import save_obj

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation(self):
        try:
            logging.info("Data Transformation Initiated")

            categorical_cols = ['cut','color','clarity']
            numerical_cols = ['carat','depth','table','x','y','z']
            cut_categories = ['Fair','Good','Very Good','Premium','Ideal']
            color_categories = ['D','E','F','G','H','I','J']
            clarity_categories = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']

            logging.info("Pipeline Initiated")

            num_pipeline = Pipeline(
                steps = [
                    ("Imputer",SimpleImputer(strategy="median")),
                    ("Scaler",StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps = [
                    ("Imputer",SimpleImputer(strategy="most_frequent")),
                    ("OrdinalEncoder",OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                    ("Scaler",StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer([
                ('num_pipeline',num_pipeline,numerical_cols),
                ('cat_pipeline',cat_pipeline,categorical_cols)
            ])

            return preprocessor

        except Exception as e:
            logging.info("An Error Occurred in the get_data_transformation in data_transformation.py")
            raise customexception(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try :
            logging.info("Loading Data for Data Transformation")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("train and test data loaded successfully")

            logging.info(f"Train DataFrame Head : \n {train_df.head().to_string()}")
            logging.info(f"Test DataFrame Head : \n {test_df.head().to_string()}")

            preprocessing_obj = self.get_data_transformation()

            target_column = 'price'
            drop_column = [target_column,'id']
            input_features_train_df = train_df.drop(columns=drop_column,axis=1)
            input_features_test_df = test_df.drop(columns=drop_column, axis=1)
            target_feature_train_df = train_df[target_column]
            target_feature_test_df = test_df[target_column]

            logging.info("Applying preprocessor_object on training and testing dataset")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_features_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_features_test_df)
            logging.info("Successfully transformed training and testing data")

            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            save_obj(file_path=self.data_transformation_config.preprocessor_obj_file_path,obj=preprocessing_obj)
            logging.info("Preprocessing Object Pickle file is saved")

            return (train_arr,test_arr)

        except Exception as e:
            logging.info("Error occurred in initiated_data_transformation")
            raise customexception (e,sys)


if __name__ == "__main__":
    obj = DataTransformation()
    obj.initiate_data_transformation("artifacts/train.csv","artifacts/test.csv")


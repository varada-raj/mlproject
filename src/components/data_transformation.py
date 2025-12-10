import os
from unicodedata import category
import pandas as pd
import numpy as np
import sys
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from src.utils import save_object

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer #used to handle null/missing values
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path:str=os.path.join("artifacts","preprocessor.pkl")#path fopr storing pkl file

class DataTransformation:
    def __init__(self):
        self.dataTransformation_config=DataTransformationConfig()
    
    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation preprocessing
        '''
        try:
            # here in numerical columns we won't include math_score, because , we use this function, 
            # only for in-dependent cols, data, and we use math_score as pred/dependent data
            numerical_columns = ["writing_score","reading_score"]
            categorical_columns = [
                'gender', 
                'race_ethnicity', 
                'parental_level_of_education', 
                'lunch',
                'test_preparation_course'
            ]

            #used to handle null/missing values
            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )
            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("Standard Scaling",StandardScaler(with_mean=False))
                ]
            )

            logging.info("Numerical data scaling completed")

            logging.info("Categorical data encoding completed")

            logging.info(f"Categorical Columns:{categorical_columns}")
            logging.info(f"Numerical Columns:{numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("Num_pipeline",num_pipeline,numerical_columns),
                    ("Cat_pipeline",cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read Train and Test data complete")

            logging.info("obtaning preprocessed object")

            preprocessor_obj = self.get_data_transformer_object()

            target_column_name="math_score"
            numerical_columns = ["writing_score","reading_score"]

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(f"Applying Preprocessing object on training dataframe and testing dataframe.")

            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)

            #used to concatenate the input features and target features 
            #np.c_ is used to concatenate the arrays along the second axis (columns)
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved preprocessing objects")
            save_object(
                file_path=self.dataTransformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
            return (
                train_arr,
                test_arr,
                self.dataTransformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)
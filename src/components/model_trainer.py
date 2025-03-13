# Modelling
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso, ElasticNet
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

import sys
import os
from dataclasses import dataclass

from src.utils import save_pkl,evaluate_model,save_report,select_best_model
from src.logger import logging
from src.exception import customException

@dataclass
class ModelTrainerConfig:
    trainer_model_file_path = os.path.join("artifacts","model.pkl")

@dataclass
class ModelReportConfig:
    report_path = os.path.join("artifacts","reports.txt")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.model_report_config = ModelReportConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            x_train = train_arr[:,:-1]
            y_train = train_arr[:,-1]
            x_test = test_arr[:,:-1]
            y_test = test_arr[:,-1]
            logging.info("split training and test data")

            models = {
                "LinearRegression":LinearRegression(),
                "Ridge":Ridge(),
                "Lasso":Lasso(),
                "ElasticNet":ElasticNet(),
                "SVR":SVR(),
                "KNeighborsRegressor":KNeighborsRegressor(),
                "DecisionTreeRegressor":DecisionTreeRegressor() ,
                "RandomForestRegressor":RandomForestRegressor() ,
                "AdaBoostRegressor":AdaBoostRegressor() ,
                "GradientBoostingRegressor":GradientBoostingRegressor(),
                "CatBoostRegressor":CatBoostRegressor(),
                "XGBRegressor":XGBRegressor(),
            }
            logging.info("models initialized")

            model_report = evaluate_model(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models)
            save_report(self.model_report_config.report_path,model_report)
            logging.info("report saved successfully")

            best_model, best_r2_score = select_best_model(model_report)
            best_model_object = models[best_model]
            save_pkl(self.model_trainer_config.trainer_model_file_path,best_model_object)
            return best_r2_score

        except Exception as e:
            logging.info("something went wrong while training the model")
            raise customException(e,sys)
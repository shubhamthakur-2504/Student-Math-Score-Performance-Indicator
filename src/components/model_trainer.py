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

            param_grids = {
                "LinearRegression": {},
                
                "Ridge": {
                    "alpha": [0.01, 0.1, 1, 10, 100]
                },
                
                "Lasso": {
                    "alpha": [0.0001, 0.001, 0.01, 0.1, 1]
                },
                
                "ElasticNet": {
                    "alpha": [0.0001, 0.001, 0.01, 0.1, 1],
                    "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]
                },
                
                "SVR": {
                    "kernel": ["linear", "poly", "rbf", "sigmoid"],
                    "C": [0.1, 1, 10, 100],
                    "epsilon": [0.01, 0.1, 0.2, 0.5]
                },
                
                "KNeighborsRegressor": {
                    "n_neighbors": [3, 5, 7, 9],
                    "weights": ["uniform", "distance"],
                    "metric": ["euclidean", "manhattan", "minkowski"]
                },
                
                "DecisionTreeRegressor": {
                    "max_depth": [3, 5, 10, None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": ["sqrt", "log2", None] 
                },
                
                "RandomForestRegressor": {
                    "n_estimators": [50, 100, 200, 300],
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": [None, "sqrt", "log2"]
                },
                
                "AdaBoostRegressor": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 0.5, 1]
                },
                
                "GradientBoostingRegressor": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 0.2, 0.3],
                    "max_depth": [3, 5, 7, 10],
                    "min_samples_split": [2, 5, 10]
                },
                
                "CatBoostRegressor": {
                    "iterations": [100, 500, 1000],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "depth": [4, 6, 8, 10]
                },
                
                "XGBRegressor": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 0.2, 0.3],
                    "max_depth": [3, 5, 7, 10],
                    "subsample": [0.6, 0.8, 1.0],
                    "colsample_bytree": [0.6, 0.8, 1.0]
                }
            }


            model_report, trained_models = evaluate_model(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models, params=param_grids)
            save_report(self.model_report_config.report_path,model_report)
            logging.info("report saved successfully")

            best_model, best_r2_score = select_best_model(model_report)
            best_model_object = trained_models[best_model]
            save_pkl(self.model_trainer_config.trainer_model_file_path,best_model_object)
            return best_r2_score

        except Exception as e:
            logging.info("something went wrong while training the model")
            raise customException(e,sys)
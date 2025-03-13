import sys
import dill
import json
import os
from src.exception import customException
from src.logger import logging
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def save_report(file_path,report):
    try:
        file_dir = os.path.dirname(file_path)
        os.makedirs(file_dir,exist_ok=True)
        report_str = json.dumps(report, indent=4)

        with open(file_path,'w') as file_obj:
            file_obj.write(report_str)
        
        logging.info("report file saved successfully")
    except Exception as e:
        logging.error("something went wrong while saving report")
        raise customException(e,sys)
    
def save_pkl(file_path,file):
    try:
        file_dir = os.path.dirname(file_path)
        os.makedirs(file_dir,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            dill.dump(file,file_obj)
        
        logging.info("pickle file saved successfully")
    except Exception as e:
        logging.error("something went wrong while saving pickle file")
        raise customException(e,sys)
    

def evaluate_model(x_train,y_train,x_test,y_test,models):
    report={}
    try:
        for i in range(len(list(models))):

            model = list(models.values())[i]
            name = list(models.keys())[i]

            try:
                
                model.fit(x_train,y_train)

                y_pred_train = model.predict(x_train)
                y_pred_test = model.predict(x_test)

                r2_train = r2_score(y_train,y_pred_train)
                r2_test = r2_score(y_test,y_pred_test)

                mae_train = mean_absolute_error(y_train,y_pred_train)
                mae_test = mean_absolute_error(y_test,y_pred_test)

                mse_train = mean_squared_error(y_train,y_pred_train)
                mse_test = mean_squared_error(y_test,y_pred_test)

                result = {
                    "train":{
                        "r2":r2_train,
                        "mae":mae_train,
                        "mse":mse_train
                    },
                    "test":{
                        "r2":r2_test,
                        "mae":mae_test,
                        "mse":mse_test
                    }
                }
                report[name]= result
                if not report:
                    logging.warning("no model trained")
                logging.info(f"model {name} trained successfully")
            except Exception as e:
                logging.error(f"something went wrong while training {name} model")
        return report
    except Exception as e:
        logging.error("something went wrong while evaluating models")
        raise customException(e,sys)
    

def select_best_model(report):
    best_model = None
    best_r2_score = -float('inf') 
    
    for model_name, metrics in report.items():
        r2_test = metrics['test']['r2']
        if r2_test > best_r2_score:
            best_r2_score = r2_test
            best_model = model_name
            
    logging.info(f"The best model is {best_model} with an R2 score of {best_r2_score}")
    
    return best_model, best_r2_score

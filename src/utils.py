import sys
import dill
import os
from src.exception import customException
from src.logger import logging

def save_pkl(file_path,file):
    try:
        file_dir = os.path.dirname(file_path)
        os.makedirs(file_dir,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            dill.dump(file,file_obj)
        
        logging.info("pickle file saved successfully")
    except Exception as e:
        logging.warning("something went wrong while saving pickle file")
        raise customException(e,sys)
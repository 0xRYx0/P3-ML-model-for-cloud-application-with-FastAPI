''' 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Udacity :  Machine Learning DevOps Engineer (MLOps) Nano-degree
  Project :  3 - Deploying a ML Model on Heroku with FastAPI
  Step    :  Main pipeline script
  Author  :  Rakan Yamani
  Date    :  19 June 2023
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''

import os
import logging
import joblib
from pipeline.data import clean_data
from pipeline.model import train_model
from pipeline.evaluate import evaluate_model, evaluate_sliced_data

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

logging.basicConfig(filename='logging.log', filemode='a', level=logging.INFO,
                    format='%(asctime)s %(name)s %(levelname)s - %(message)s', datefmt="%m/%d/%y %I:%M:%S %p")

def run():
    '''
    This is the main script for executing the pipeline 
    '''
    
    # prepare freatures and labels by calling clean_data method from 'pipeline.data' 
    X,y = clean_data(os.getcwd()+'/../data/census.csv')
    logging.info("SUCCESS: Preparing required features (X) and leabels (y)") 
    logging.info(f"Features (X): {X.shape}") 
    logging.info(f"Labels (y): {y.shape}") 
    
    # split data for training:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    logging.info("SUCCESS: Preparing required features (X) and leabels (y)") 
    logging.info(f"X_train data shape: {X_train.shape}")
    logging.info(f"y_train data shape: {y_train.shape}")
    logging.info(f"X_test data shape: {X_test.shape}")
    logging.info(f"y_test data shape: {y_test.shape}")
    
    # create Random Forest model: 
    rf_model = RandomForestClassifier(class_weight='balanced', random_state=42)
    logging.info("SUCCESS: Creating Random Forest model") 
    
    # train and optimise RF model using 'train_model' method from 'pipeline.model'
    rf_model = train_model(rf_model, X_train, y_train)
    logging.info("SUCCESS: Training and optimising Random Forest model using Grid-search approach") 
    
    # validate model using 'evaluate_model' method from 'pipeline.evaluate'
    with open(os.getcwd()+'/../model/model_evaluation_scores.txt', 'w') as file:
        evaluate_model(file, rf_model, X_train, y_train, "Train metrics")
        logging.info("SUCCESS: Validating model performance uing train data") 
        
        evaluate_model(file, rf_model, X_test, y_test, "Test metrics")
        logging.info("SUCCESS: Validating model performance uing test data") 
        
    
    # validate model using 'evaluate_model' method from 'pipeline.evaluate'
    with open(os.getcwd()+'/../model/slices_evaluation_scores.txt', 'w') as file:
        evaluate_sliced_data(file, rf_model, X_train, y_train, "Train metrics")
        logging.info("SUCCESS: Validating sliced data performance uing train data") 
        
        evaluate_sliced_data(file, rf_model, X_test, y_test, "Test metrics")   
        logging.info("SUCCESS: Validating sliced data performance uing test data")  
    
    # save model 
    joblib.dump(rf_model, os.getcwd()+'/../model/model.pkl')
    logging.info("SUCCESS: Exporing model")  
  
if __name__ == "__main__":
    run()
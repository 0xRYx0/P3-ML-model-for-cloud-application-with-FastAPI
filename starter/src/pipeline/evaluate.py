''' 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Udacity :  Machine Learning DevOps Engineer (MLOps) Nano-degree
  Project :  3 - Deploying a ML Model on Heroku with FastAPI
  Step    :  Model evaluation script
  Author  :  Rakan Yamani
  Date    :  19 June 2023
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
import os
import logging 
import pandas as pd
from pipeline.model import inference
from sklearn.metrics import fbeta_score, precision_score, recall_score

logging.basicConfig(filename=os.getcwd()+'/../logging.log', filemode='a', level=logging.INFO,
                    format='%(asctime)s %(name)s %(levelname)s - %(message)s', datefmt="%m/%d/%y %I:%M:%S %p")

def compute_model_metrics(y_train, y_preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
        y_train (np.array): Known labels
        y_preds (np.array): Predicted labels
        
    Returns
    -------
        precision : float
        recall    : float
        fbeta     : float
    """
    f1 = fbeta_score(y_train, y_preds, beta=1, zero_division=1)
    precision = precision_score(y_train, y_preds, zero_division=1)
    recall = recall_score(y_train, y_preds, zero_division=1)
    logging.info("SUCCESS: Validating the trained  model using precision, recall, and F1")
    logging.info(f"Precision=[{precision}] Recall=[{recall}] F1=[{f1}]")    
    
    return precision, recall, f1


def evaluate_model(file, model_pipe, X, y, mode):
    """
    This method is to validate the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
        file (object): file object to capture the results 
        model_pipe: 
        X (np.array): Training data
        y (np.array): Known labels
        mode: message to clarify the operation
        
    Returns: None 

    """
    # create an inference model using 'inference' method from 'pipeline.model':
    y_preds = inference(model_pipe, X)
    
    # validating model:
    precision, recall, f1 = compute_model_metrics(y, y_preds)
    
    # write validation score to file:
    print(f'{mode}:\t Precision=[{precision:.4f}]  Recall=[{recall:.4f}]  F1 score=[{f1:.4f}]', file=file)
    
    
def evaluate_sliced_data(file, model_pipe, X, y, mode): 
    """
        This method is to validate the a slice of data using the trained machine learning model for precision, recall, and F1.
    
    Inputs
    ------
        file (object): file object to capture the results 
        model_pipe: 
        X (np.array): Training data
        y (np.array): Known labels
        mode: message to clarify the operation
        
    Returns: None 
    """
    # create an inference model using 'inference' method from 'pipeline.model':
    y_preds = inference(model_pipe, X)
    
    slicing_columns = ['sex', 'race','marital_status', 'education']
    df = pd.concat([X[slicing_columns], y], axis=1)
    df['predicted_salary'] = y

    # validating model per slice:
    for col in slicing_columns:
        print(f'\n### Metrics for [{col}] ###')
        for value in sorted(df[col].unique()):
            precision, recall, f1 = compute_model_metrics(df[df[col] == value]['predicted_salary'],
                                                          df[df[col] == value]['salary'])
            print(f'{value}:\t Precision=[{precision:.4f}]  Recall=[{recall:.4f}]  F1 score=[{f1:.4f}]', file=file)
    
    
    






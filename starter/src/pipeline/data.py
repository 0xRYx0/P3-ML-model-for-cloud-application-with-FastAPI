''' 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Udacity :  Machine Learning DevOps Engineer (MLOps) Nano-degree
  Project :  3 - Deploying a ML Model on Heroku with FastAPI
  Step    :  Data Cleansing Script
  Author  :  Rakan Yamani
  Date    :  19 June 2023
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''

import pandas as pd
import os
import logging

logging.basicConfig(filename=os.getcwd()+'/../logging.log', filemode='a', level=logging.INFO,
                    format='%(asctime)s %(name)s %(levelname)s - %(message)s', datefmt="%m/%d/%y %I:%M:%S %p")


def clean_data(path):
    """
    This models is to load a csv dataset from its original path location and perform required EDA operations
    
    Input:
        path (str): The path to the csv file

    Returns:
        X (pandas dataframe): Features dataframe 
        y (pandas dataframe): Labels dataframe 
    """
    
    df = pd.read_csv(path)
    
    # drop duplicates values
    df = df[~df.duplicated()] 
    logging.info("SUCCESS: Removed duplicates values")        
    
    # delete all white spaces in each row of categorical columns   
    for col in df.columns:
        df[col] = df[col].replace(' ', '', regex=True)          
    logging.info("SUCCESS: Removed all white spaces in each row of categorical columns")  
    
    # delete all white spaces in column names 
    df.columns = [col.replace(' ', '') for col in df.columns]
    logging.info("SUCCESS: Replaced all white spaces in column names")    
    
    # replace hypone char (-) with underscor char (_) in column names 
    df.columns = [col.replace('-', '_') for col in df.columns]
    logging.info("SUCCESS: Replaced hypone char (-) with underscor char (_) in column names")  
      
    # map salaries to either 1 for ">50K" or 0 for "<=50K"
    df['salary'] = df['salary'].map({'>50K': 1, '<=50K': 0})
    logging.info("SUCCESS: Mapped salaries to either 1 for '>50K' or 0 for '<=50K'")  

    # prepare dataset (X) and labels (y):
    X = df.drop(['salary'], axis=1)
    y = df.pop('salary')
    logging.info("SUCCESS: Prepared Training data and labels")  

    return X, y
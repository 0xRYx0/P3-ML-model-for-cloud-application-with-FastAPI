''' 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Udacity :  Machine Learning DevOps Engineer (MLOps) Nano-degree
  Project :  3 - Deploying a ML Model on Heroku with FastAPI
  Step    :  Testing script
  Author  :  Rakan Yamani
  Date    :  19 June 2023
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''

import logging
import os
import pandas as pd

from data import clean_data
from model import create_pipeline

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OrdinalEncoder


logging.basicConfig(filename=os.getcwd()+'/../logging.log', filemode='a', level=logging.INFO,
                    format='%(asctime)s %(name)s %(levelname)s - %(message)s', datefmt="%m/%d/%y %I:%M:%S %p")


def test_model_pipeline_creation():
    """
    Tests if model pipeline is created with correct configuration
    """
    # define main variables:
    drop_columns = ['education']
    numeric_columns = ['age','capital_gain','capital_loss','fnlgt', 'hours_per_week']
    categorical_columns = ['marital_status', 'native_country', 'occupation',
                          'relationship', 'race', 'sex', 'workclass']

    # define grid prams
    param_grid = {
        'model__n_estimators': list(range(50, 251, 50)),
        'model__max_depth': list(range(2, 15, 2)),
        'model__min_samples_leaf': list(range(1, 51, 10))
    }
    
    rf_model = RandomForestClassifier(class_weight='balanced', random_state=42)
    model_pipe = create_pipeline(rf_model, categorical_columns, numeric_columns, drop_columns)
    
    
    # check classifer 
    assert isinstance(model_pipe[1], RandomForestClassifier), "Model is not Random Forest Classifier"
        
    # check features columns     
    assert model_pipe[0].transformers[0][2] == drop_columns, "Dropped cloumn is mismatched"
    assert model_pipe[0].transformers[1][2] == numeric_columns, "Numeric cloumns are mismatched" 
    assert model_pipe[0].transformers[2][2] == categorical_columns, "Categorical cloumns are mismatched"  
    
    # check numerical and categorical encoders        
    assert isinstance(model_pipe[0].transformers[1][1], StandardScaler), "Numeric encoder is not StandardScaler"  
    assert isinstance(model_pipe[0].transformers[2][1][1], OrdinalEncoder), "Categorical encoder is not OrdinalEncoder"    
    print("SECCESS: Model pipeline was formed correctly")
        
    

def test_data_shape_match():
    """
    Tests if dataset shape are matched
    """
    
    X,y = clean_data(os.getcwd()+'/../data/census.csv')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    assert X_train.shape[0] == y_train.shape[0], "Shape mismatch between X_train and y_train"
    assert X_test.shape[0] == y_test.shape[0], "Shape mismatch between X_test and y_test"
    assert X_train.shape[1] == X_test.shape[1], "Number of features mismatch between X_train and X_test"
    print("SECCESS: Shapes of sliced data are matching")



def assert_dataset_requirements():
    """
    Tests if dataset requirements were satisfied 
    """
    
    X,y = clean_data(os.getcwd()+'/../data/census.csv')
    df = pd.concat([X,y], axis=1)

    # expected columns and data types after processing  
    required_columns = {
        'age': 'int64',
        'capital_gain': 'int64',
        'capital_loss': 'int64',
        'hours_per_week': 'int64',
        'education_num': 'int64',
        'fnlgt': 'int64',
        'education': 'object',
        'marital_status': 'object',
        'occupation': 'object',
        'relationship': 'object',
        'race': 'object',
        'sex': 'object',
        'native_country': 'object',
        'salary': 'int64',
        'workclass': 'object'
    }
    
    # Check if all required columns are present in the dataset
    assert set(required_columns.keys()).issubset(df.columns), "Missing required columns in the dataset"
    print("SECCESS: Required column were set")
    
    # Check if the data types of required columns match the expected types
    for column, expected_dtype in required_columns.items():
        assert df[column].dtype == expected_dtype, f"Column '{column}' has incorrect data type: expected {expected_dtype} got {df[column].dtype}"
    print("SECCESS: Required column and data types were set")
    
    
    
# test_model_pipeline_creation()
# test_data_shape_match()
# assert_dataset_requirements()
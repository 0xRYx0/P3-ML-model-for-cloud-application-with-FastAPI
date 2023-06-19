''' 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Udacity :  Machine Learning DevOps Engineer (MLOps) Nano-degree
  Project :  3 - Deploying a ML Model on Heroku with FastAPI
  Step    :  Model training script
  Author  :  Rakan Yamani
  Date    :  19 June 2023
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
import os
import logging 

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

logging.basicConfig(filename=os.getcwd()+'/../logging.log', filemode='a', level=logging.INFO,
                    format='%(asctime)s %(name)s %(levelname)s - %(message)s', datefmt="%m/%d/%y %I:%M:%S %p")

def create_pipeline(model, categorical_columns, numeric_columns, drop_columns):
    """
        This method is to return a pipeline of the provided Random Forest model
    Inputs
    ------
        model : Trained machine learning model with optimal prams
        categorical_columns (dict): range of features
        numeric_columns (np.array): Training data
        drop_columns (np.array): Known labels
        
    Returns
    -------
        rf_pipeline: pipelined machine learning model

    """   
    # Preprocessor of numerical features:
    numeric_encoder = StandardScaler()
    
    # Preprocessor of categorical features:
    categorical_encoder = make_pipeline(SimpleImputer(strategy='most_frequent'), 
                          OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=1111))
    
    # Preprocessor of all features:
    features_preprocessor = ColumnTransformer([
        ('drop', 'drop', drop_columns),
        ('numerical', numeric_encoder, numeric_columns),
        ('categorical', categorical_encoder, categorical_columns)        
        ],
        # handling columns not specified in the transformer, but present in the data passed to 'fit'
        remainder='passthrough' 
    )

    # model pipeline
    rf_pipeline = Pipeline([
        ('features_preprocessor', features_preprocessor),
        ('model', model)
    ])
    
    logging.info("SUCCESS: Creating model pipeline utlizing provided model & features") 
    
    return rf_pipeline

def get_best_model(pipeline, param_grid, X_train, y_train):
    """
    Hyper-tuning supplied model using provided selections of features  

    Inputs
    ------
        model : Trained machine learning model with optimal prams
        param_grid (dict): range of features
        X_train (np.array): Training data
        y_train (np.array): Known labels
        
    Returns
    -------
        precision : float
        recall : float
        fbeta : float
    """
    # prepare search grid component
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        n_jobs=5,
        cv=StratifiedKFold(),
        scoring='f1',
        error_score='raise'
    )
    logging.info("SUCCESS: Preparing a grid search for prameters optimzation") 
    
    # fit grid search with train data and labels
    _fit = grid_search.fit(X_train, y_train)
    logging.info("SUCCESS: Fitting the grid search with provided prams selections") 
    
    #return best model
    return grid_search.best_estimator_    

def train_model(model, X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
        X_train (np.array): Training data
        y_train (np.array): Known labels
        
    Returns
    -------
        model: Trained machine learning model with optimal prams
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

    # create model pipeline with associated prams and features
    pipeline = create_pipeline(model, categorical_columns, numeric_columns, drop_columns)
    logging.info("SUCCESS: Creating model pipeline with associated prams and features")
    
    # evalute best model
    best_model = get_best_model(pipeline, param_grid, X_train, y_train)
    logging.info("SUCCESS: Optaining best model")
    
    return best_model

def inference(model, X):
    """ 
    Run model inferences and return the predictions.

    Inputs
    ------
        model : Trained machine learning model.
        X (np.array): Data used for prediction.
        
    Returns
    -------
        preds (np.array): Predictions from the model.
    """
    
    preds = model.predict(X)
    logging.info("SUCCESS: Predecting model data")
    
    return preds


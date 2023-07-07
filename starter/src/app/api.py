''' 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Udacity :  Machine Learning DevOps Engineer (MLOps) Nano-degree
  Project :  3 - Deploying a ML Model on Heroku with FastAPI
  Step    :  API Calls script
  Author  :  Rakan Yamani
  Date    :  19 June 2023
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
import joblib
import os
import yaml
import sys
import logging 
import numpy as np
import pandas as pd

from fastapi import FastAPI, Body
from pydantic import BaseModel

sys.path.append(os.getcwd()+'/../..')
from directories import _API_APP_CONFIGURATION, _MODEL_CONFIGURATION

# Create FastAPI instance
app = FastAPI()

# Loading configurations
with open(_API_APP_CONFIGURATION) as fp:
    configurations = yaml.safe_load(fp)
        
# # Loading required model
model = joblib.load(_MODEL_CONFIGURATION)
        
# Model input data schema
class InputData(BaseModel):
    age: int
    capital_gain: int
    capital_loss: int
    fnlgt: int
    education_num: int
    hours_per_week: int
    workclass: str = None
    education: str = None
    marital_status: str = None
    occupation: str = None
    relationship: str = None
    race: str = None
    sex: str = None
    native_country: str = None


# GET endpoint for root
@app.get("/")
async def index():
    logging.info("SUCCESS: API route for - Index Endpoint") 
    return {"message": "Here we go! Welcome"}

# GET endpoint for features
@app.get("/features_details/{feature}")
async def feature_info(feature):
    info = configurations['features_details'][feature]
    return info


# POST endpoint for model inference
@app.post("/predictions/")
async def inference(input_data: InputData = Body(...,examples=configurations['post_examples'])):
    # print('1. start function')    
    # features = np.array([input_data.__dict__[f] for f in configurations['features_details']].keys())
    # print(features)
    # print('2. start features')  
    # df_features = pd.DataFrame(features.reshape(1, -1), columns=configurations['features_details'].keys())
    # print(features)
    # print('3. Reshape features')  
    # predicted_label = int(model.predict(df_features))
    # print('4. predicted_label')  
    # prediction_probability = float(model.predict_proba(df_features)[:, 1])
    # print('5. prediction_probability')  
    # pred = '>50k' if predicted_label == 1 else '<=50k'
    # print('. prediction_probability') 
    
    
    data = input_data.dict()
    features = np.array([data[f] for f in configurations['features_details'].keys()]).reshape(1, -1)
    df = pd.DataFrame(features, columns=configurations['features_info'].keys())
    pred_label = int(model.predict(df))
    pred_probs = float(model.predict_proba(df)[:, 1])
    pred = '>50k' if pred_label == 1 else '<=50k'

    return {'Prediction': pred_label, 'Probability': pred_probs, 'Salary Range': pred}
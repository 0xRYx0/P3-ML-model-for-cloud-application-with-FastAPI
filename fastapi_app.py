''' 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Udacity :  Machine Learning DevOps Engineer (MLOps) Nano-degree
  Project :  3 - Deploying a ML Model on Heroku with FastAPI
  Step    :  API Calls script
  Author  :  Rakan Yamani
  Date    :  19 June 2023
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
from fastapi import FastAPI, Body
import joblib
import os
import yaml
import logging 
import numpy as np
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel
from typing_extensions import Annotated

# Create FastAPI instance
app = FastAPI()

# Loading configurations
with open("fastapi_config.yaml") as fp:
    configurations = yaml.safe_load(fp)
        
# Loading required model
model = joblib.load("fastapi_model.pkl")

        
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
    
    class Config:
        schema_extra = {
            "examples": [
                {
                    "age": 45,
                    "workclass": "State-gov",
                    "fnlgt": 448512,
                    "education": "bachelors",
                    "education_num": 14,
                    "marital_status": "Divorced",
                    "occupation": "prof-specialty",
                    "relationship": "wife",
                    "race": "Black",
                    "sex": "female",
                    "capital_gain": 0,
                    "capital_loss": 0,
                    "hours_per_week": 60,
                    "native_country": "taiwan"
                }
            ]
        }
    

# GET endpoint for root
@app.get("/")
async def index():
    logging.info("SUCCESS: API route for - Index Endpoint") 
    return {"message": "Here we go! Welcome"}

# GET endpoint for features
@app.get("/features_details/{feature}")
async def feature_info(feature):
    info = configurations['fastapi_features_details'][feature]
    return info


# POST endpoint for model inference
@app.post("/prediction/")
async def prediction(input_data: Annotated[
        InputData,
        Body(...,
            examples=[
                {
                    "age": 45,
                    "workclass": "State-gov",
                    "fnlgt": 448512,
                    "education": "bachelors",
                    "education_num": 14,
                    "marital_status": "Divorced",
                    "occupation": "prof-specialty",
                    "relationship": "wife",
                    "race": "Black",
                    "sex": "female",
                    "capital_gain": 0,
                    "capital_loss": 0,
                    "hours_per_week": 60,
                    "native_country": "taiwan"
                },
                {
                    "age": 45,
                    "workclass": "State-gov",
                    "fnlgt": 448512,
                    "capital_loss": 0,
                    "hours_per_week": 60,
                    "native_country": "taiwan"
                },
                {
                    "age": 45,
                    "workclass": "State-gov"
                    
                },
            ],
        ),
    ]):
        
    print('1.1 start function')    
    features = np.array([input_data.__dict__[f] for f in configurations['fastapi_features_details']])
    print('2.1 start features')  
    features = pd.DataFrame(features.reshape(1, -1), columns=configurations['fastapi_features_details'])
    print('3.1 Reshape features')  
    predicted_label = int(model.predict(features))
    print('4.1 predicted_label')  
    prediction_probability = float(model.predict_proba(features)[:, 1])
    print('5.1 prediction_probability')  
    pred = '>50k' if predicted_label == 1 else '<=50k'
    print('6.1 prediction_probability') 

    return {'Prediction': predicted_label, 'Probability': prediction_probability, 'Salary Range': pred}
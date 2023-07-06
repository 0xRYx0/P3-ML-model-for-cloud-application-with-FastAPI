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
            "examples": {
                "class >50k [label 1]": {
                    "summary": "An example of class >50k",
                    "description": "A example that should predict a class of >50k",
                    "value": {
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
                        "native_country": "taiwan",
                    },
                },
                "class <=50k [label 0]": {
                    "summary": "An example of an individual with an income of <=50k",
                    "description": "This example represents an individual with various characteristics that are used to predict their income level",
                    "value": {
                        "age": 37,
                        "workclass": "Self-emp-inc",
                        "fnlgt": 32165,
                        "education": "masters",
                        "education_num": 14,
                        "marital_status": "Married",
                        "occupation": "adm-clerical",
                        "relationship": "Husband",
                        "race": "Asian-Pac-Islander",
                        "sex": "male",
                        "capital_gain": 2174,
                        "capital_loss": 0,
                        "hours_per_week": 40,
                        "native_country": "united-states",
                    },
                },
                "error_sample": {
                    "summary": "An example of a sample that will cause an error",
                    "description": "This example represents a sample that will cause an error with the model due to missing age and fnlgt variables",
                    "value": {
                        "workclass": "local-gov",
                        "education": "assoc-voc",
                        "education_num": 11,
                        "marital_status": "divorced",
                        "occupation": "prof-specialty",
                        "relationship": "unmarried",
                        "race": "white",
                        "sex": "female",
                        "capital_gain": 0,
                        "capital_loss": 0,
                        "hours_per_week": 1,
                        "native_country": "taiwan"
                    },
                },
                "missing_sample": {
                    "summary": "An example of a sample with missing values",
                    "description": "This example showcases the model's ability to handle missing values for certain features",
                    "value": {
                        "age": 81,
                        "fnlgt": 120478,
                        "education_num": 11,
                        "capital_gain": 0,
                        "capital_loss": 0,
                        "hours_per_week": 1
                    },
                },
            }
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
async def inference(input_data: Annotated [InputData,Body(None,examples=InputData.Config.schema_extra["examples"])]):
        
    print('1.1 start function')    
    features = np.array([input_data.__dict__[f] for f in configurations['features_details']])
    print('2.1 start features')  
    features = pd.DataFrame(features.reshape(1, -1), columns=configurations['features_details'])
    print('3.1 Reshape features')  
    predicted_label = int(model.predict(features))
    print('4.1 predicted_label')  
    prediction_probability = float(model.predict_proba(features)[:, 1])
    print('5.1 prediction_probability')  
    pred = '>50k' if predicted_label == 1 else '<=50k'
    print('6.1 prediction_probability') 

    return {'Prediction': predicted_label, 'Probability': prediction_probability, 'Salary Range': pred}
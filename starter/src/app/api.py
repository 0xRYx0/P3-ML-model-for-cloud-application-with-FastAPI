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
import typing
import requests

from pipeline.data import clean_data
from pipeline.evaluate import compute_model_metrics

from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.model_selection import train_test_split

# Create FastAPI instance
app = FastAPI()

# Model input data schema
class InputData(BaseModel):
    age: int
    capital_gain: int
    capital_loss: int
    fnlgt: int
    education_num: int
    hours_per_week: int
    workclass: Optional[str] = None
    education: Optional[str] = None
    marital_status: Optional[str] = None
    occupation: Optional[str] = None
    relationship: Optional[str] = None
    race: Optional[str] = None
    sex: Optional[str] = None
    native_country: Optional[str] = None
    
    
with open("config.yaml") as fp:
    config = yaml.safe_load(fp)

# model = joblib.load(os.getcwd()+'/../model/model.pkl')


# GET endpoint for root
@app.get("/")
async def index():
    logging.info("SUCCESS: API route for - Index Endpoint") 
    return {"message": "Here we go! Welcome"} 


@app.get("/feature_info/{feature}")
async def feature_info(feature):
    info = config['features_details'][feature]
    return info


# POST endpoint for model inference
@app.post("/prediction/{input_data}")
def inference(input_data: InputData):
    
    X,y = clean_data(os.getcwd()+f'/../data/{input_data}')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    model = joblib.load(os.getcwd()+'/../../model/model.pkl')
    y_preds = model.predict(X_train)
    precision, recall, f1 = compute_model_metrics(y_train, y_preds)
    result = {"Model Matrics": f"Precision=[{precision}] Recall=[{recall}] F1=[{f1}]"}
    return result

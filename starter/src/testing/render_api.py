''' 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Udacity :  Machine Learning DevOps Engineer (MLOps) Nano-degree
  Project :  3 - Deploying a ML Model on Heroku with FastAPI
  Step    :  API Live Query Testing Script
  Author  :  Rakan Yamani
  Date    :  19 June 2023
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''

import requests


data = {
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

# GET request
response = requests.get('https://ml-model-fastapi.onrender.com/')
print(response.status_code)
print(response.json())

# GET request
response = requests.get('https://ml-model-fastapi.onrender.com/features_details/age')
print(response.status_code)
print(response.json())

# POST request
response = requests.post('https://ml-model-fastapi.onrender.com/prediction', json=data)
print(response.status_code)
print(response.json())
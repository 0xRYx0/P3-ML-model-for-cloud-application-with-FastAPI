''' 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Udacity :  Machine Learning DevOps Engineer (MLOps) Nano-degree
  Project :  3 - Deploying a ML Model on Heroku with FastAPI
  Step    :  API Calls Testing Script
  Author  :  Rakan Yamani
  Date    :  19 June 2023
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''

import pytest
import os 
import sys
from http import HTTPStatus
from fastapi.testclient import TestClient

sys.path.append(os.getcwd()+'/../..')
from directories import _SOURCE_API_DIRECTORY

sys.path.insert(1, _SOURCE_API_DIRECTORY)
from api import app

# Initialize the test client
client = TestClient(app)


data_label_1 = {
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

data_label_0 = {
    "age": 37,
    "workclass": "Self-emp-inc",
    "fnlgt": 32165,
    "education": "masters",
    "education_num": 13,
    "marital_status": "Married",
    "occupation": "adm-clerical",
    "relationship": "Husband",
    "race": "Asian-Pac-Islander",
    "sex": "male",
    "capital_gain": 2174,
    "capital_loss": 0,
    "hours_per_week": 40,
    "native_country": "united-states"
}


def test_greetings():
    """
    This function tests the GET request to the greetings endpoint and asserts the response status code,
    request method, and response message.

    Returns:
        None
    """
    response = client.get('/')
    assert response.status_code == HTTPStatus.OK , "Unreachable endpoint: Greetings" 
    assert response.request.method == "GET" , "Request method is not GET" 
    assert response.json() == {"message": "Here we go! Welcome"}
    print("SECCESS: Testing API call for test_greetings endpoint")


@pytest.mark.parametrize('test_input, expected', [
    ('age', "The age of the person in years. It is represented as a numerical value. (Numerical - Integer)"),
    ('sex', "The gender of the person, either Male or Female. It is represented as a categorical variable. (Nominal Categorical - String)"),
    ('hours_per_week', "The number of hours worked per week by the person. It is represented as a numerical value. (Numerical - Integer)")
])
def test_feature_details_status_and_response(test_input: str, expected: str):
    """
    This function tests the GET request to the features_details endpoint with different test inputs
    and asserts the response status code, request method, and the expected feature details.

    Args:
        test_input (str): Example input.
        expected (str): Expected output.

    Returns:
        None
    """
    response = client.get(f'/features_details/{test_input}')
    assert response.status_code == HTTPStatus.OK , "Unreachable endpoint: Greetings" 
    assert response.request.method == "GET" , "Request method is not GET" 
    assert response.json() == expected , f"Mismatched feature details for {test_input}"  
    print("SECCESS: Testing API call for test_feature_details_status_and_response endpoint")


def test_predict_above_50k():
    """
    This function tests the POST request to the predict endpoint and asserts the response status code
    """
    response = client.post("/predictions/", json=data_label_1)
    assert response.status_code == HTTPStatus.OK , "Unreachable endpoint: Prediction" 
    assert response.json()['Salary Range'] == '>50k', "Inaccurate salary range for prediction" 
    print("SECCESS: Testing API call for test_predict_above_50k endpoint")
    
def test_predict_below_50k():
    """
    This function tests the POST request to the predict endpoint and asserts the response status code
    """
    response = client.post("/predictions/", json=data_label_0)
    assert response.status_code == HTTPStatus.OK , "Unreachable endpoint: Prediction" 
    assert response.json()['Salary Range'] == '<=50k', "Inaccurate salary range for prediction" 
    print("SECCESS: Testing API call for test_predict_below_50k endpoint")

def test_missing_feature_predict():
    """
    This function tests the POST request to the predict endpoint with missing features and asserts
    the response status code, request method, and the expected error detail.

    Returns:
        None
    """
    data = {
        "age": 0
    }
    response = client.post("/predictions/", json=data)
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY, "Unreachable endpoint: Prediction" 
    assert response.request.method == "POST", "Request method is not POST"
    assert response.json()["detail"][0]["type"] == "value_error.missing", "Error: unknow issue"
    print("SECCESS: Testing API call for test_missing_feature_predict endpoint")
    
def test_predict_status():
    """
    This function tests the POST request to the predict endpoint and asserts the response status code
    """
    response = client.post("/predictions/", json=data_label_1)
    assert response.status_code == HTTPStatus.OK , "Unreachable endpoint: Prediction" 
    assert response.request.method == "POST" , "Request method is not POST"
    print("SECCESS: Testing API call for test_predict_status endpoint")

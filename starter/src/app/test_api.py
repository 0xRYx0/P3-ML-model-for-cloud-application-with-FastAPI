import pytest
from http import HTTPStatus
from fastapi.testclient import TestClient

from pipeline.api import app


# Initialize the test client
client = TestClient(app)

# model = joblib.load(os.path.join(os.getcwd(), "..","..", "model", "model.pkl"))

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
    print("SECCESS: Testing API call for greetigns endpoint")


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
    print("SECCESS: Testing API call for predicting feature details")


def test_predict_status():
    """
    This function tests the POST request to the predict endpoint and asserts the response status code,
    request method, predicted label, predicted probability, and salary range.

    Returns:
        None
    """
    data = {
        'age': 38,
        'fnlgt': 15,
        'education_num': 1,
        'capital_gain': 0,
        'capital_loss': 0,
        'hours_per_week': 5
    }
    response = client.post("/prediction/", json=data)
    assert response.status_code == HTTPStatus.OK , "Unreachable endpoint: Prediction" 
    assert response.request.method == "POST" , "Request method is not POST)"
    assert response.json()['Prediction'] == 0 or response.json()['Prediction'] == 1 , "Inaccurate labels for prediction"  
    assert response.json()['Probability'] >= 0 and response.json()['Probability'] <= 1 , "Inaccurate probability for prediction" 
    assert response.json()['Salary Range'] == '>50k' or response.json()['Salary Range'] == '<=50k' , "Inaccurate salary range for prediction" 
    print("SECCESS: Testing API call for predicting input with proper features")


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
    response = client.post("/prediction/", json=data)
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY, "Unreachable endpoint: Prediction" 
    assert response.request.method == "POST", "Request method is not POST"
    assert response.json()["detail"][0]["type"] == "value_error.missing", "Error: unknow issue"
    print("SECCESS: Testing API call for predicting input with missing features")

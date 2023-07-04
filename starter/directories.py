import os
from pathlib import Path

_MAIN_DIRECTORY = Path(__file__).parent.absolute()                          

# Main directories:
_DATA_DIRECTORY = os.path.join(_MAIN_DIRECTORY, 'data')
_MODEL_DIRECTORY = os.path.join(_MAIN_DIRECTORY, 'model')
_SOURCE_DIRECTORY = os.path.join(_MAIN_DIRECTORY, 'src')

# Data sub-files: 
_DATA_ORIGINAL_SET = os.path.join(_DATA_DIRECTORY, 'census.csv')
_DATA_CLEANED_SET = os.path.join(_DATA_DIRECTORY, 'clean_census_data.csv')

# Model sub-files: 
_MODEL_CONFIGURATION = os.path.join(_MODEL_DIRECTORY, 'model.pkl')

# Source sub-directories & files
# Application:
_SOURCE_API_DIRECTORY = os.path.join(_SOURCE_DIRECTORY, 'app')
_API_APP= os.path.join(_SOURCE_API_DIRECTORY, 'api.py')
_API_APP_CONFIGURATION = os.path.join(_SOURCE_API_DIRECTORY, 'config.yaml')

# Pipeline:
_SOURCE_PIPELINE_DIRECTORY = os.path.join(_SOURCE_DIRECTORY, 'pipeline')
_PIPELINE_DATA= os.path.join(_SOURCE_PIPELINE_DIRECTORY, 'data.py')
_PIPELINE_EVALUATE = os.path.join(_SOURCE_PIPELINE_DIRECTORY, 'evaluate.py')
_PIPELINE_MODEL = os.path.join(_SOURCE_PIPELINE_DIRECTORY, 'model.py')

# Testing:
_SOURCE_TESTING_DIRECTORY = os.path.join(_SOURCE_DIRECTORY, 'testing')
_TESTING_API= os.path.join(_SOURCE_TESTING_DIRECTORY, 'test_api.py')
_TESTING_MODEL = os.path.join(_SOURCE_TESTING_DIRECTORY, 'test_model.py')
_TESTING_RENDER_API= os.path.join(_SOURCE_TESTING_DIRECTORY, 'test_model.py')



# print(_MAIN_DIRECTORY)
# print(_API_APP_CONFIGURATION)
# print(_MODEL_CONFIGURATION)
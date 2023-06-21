# Project 3: Deploying ML Model for Cloud Application with FastAPI
In this project, we will develop a classification model on publicly available `Census Bureau` data, by doing the following:
1. Create unit tests to monitor the model performance on various data slices. 
2. Deploy your model using the FastAPI package and create API tests. 
3. Incorporate the slice validation and the API tests into a CI/CD framework using GitHub Actions.

* Note:
    - Two datasets was provided in the starter code to experience updating the dataset and model in `git`.
    - Working in a command line environment is recommended for ease of use with `git` and `dvc`. If on Windows, `WSL1` or `2` is recommended.

# Environment & Repositories Set up
* Download and install conda
* Run ```conda create -n [envname] "python=3.8" scikit-learn pandas numpy pytest jupyter jupyterlab fastapi uvicorn -c conda-forge```
* Install `git` either through conda ```conda install git``` or (through  CLI, e.g. ```sudo apt-get git```)
* Create a directory for the project and initialize git
* Connect your local git repo to GitHub, continually commit changes
* Setup GitHub Actions on your repo (GitHub pre-made Actions could be used if, at a minimum,runs `pytest` and `flake8` on push and requires both to pass without error.

# Data
* Download `census.csv` and commit it to `dvc`.  [more [info](https://archive.ics.uci.edu/dataset/20/census+income) about the data]
* The data is messy, to clean it, remove all spaces.


# Model
* Using the starter code, write a machine learning model that trains on the clean data and saves the model. Complete any function that has been started.
* Write unit tests for at least 3 functions in the model code.
* Write a function that outputs the performance of the model on slices of the data.
    * Suggestion: for simplicity, the function can just output the performance on slices of just the categorical features.
* Write a model card using the provided template.

# API Creation
*  Create a RESTful API using FastAPI this must implement:
    * GET on the root giving a welcome message.
    * POST that does model inference.
    * Type hinting must be used.
    * Use a Pydantic model to ingest the body from POST. This model should contain an example.
   	 * Hint: the data has names with hyphens and Python does not allow those as variable names. Do not modify the column names in the csv and instead use the functionality of FastAPI/Pydantic/etc to deal with this.
* Write 3 unit tests to test the API (one for the GET and two for POST, one that tests each prediction).

# API Deployment
* Create a free Heroku account (for the next steps you can either use the web GUI or download the Heroku CLI).
* Create a new app and have it deployed from your GitHub repository.
    * Enable automatic deployments that only deploy if your continuous integration passes.
    * Hint: think about how paths will differ in your local environment vs. on Heroku.
    * Hint: development in Python is fast! But how fast you can iterate slows down if you rely on your CI/CD to fail before fixing an issue. I like to run flake8 locally before I commit changes.
* Write a script that uses the requests module to do one POST on your live API.

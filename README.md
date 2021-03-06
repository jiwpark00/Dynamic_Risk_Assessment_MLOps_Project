# Dynamic Risk Assessment MLOps Project
## This is the fourth and last project in Machine Learning DevOps Nanodegree.

The goal here is to take a model and mimic the end-to-end process, including scenarios for re-training and checking for potential drifts as expected during model build and deployment, and release.

# Data Ingestion
The file ingestion.py combines the two data by reading in the directory, combines them, removes duplicates, and then saves it.

# Training, Scoring, and Deployment
Training is done via training.py file. The output model file is in the pickle format. This model is not hyperparameter tuned.

Scoring is done to write the F1 score to a .txt file for the latestscore measured.

Deployment takes the saved model, ingested file, and F1 score into the production env directory.

# Diagnostics
There is a single diagnostics.py file that does a quite bit of work here.

- This file checks to ensure that the deployed model can score on the test dataset.

- This file also measures the performance of data ingestion and training and report this.

- Summary statistics for the numerical columns are calculated for mean, median, and standard deviation.

- Dependency statuses are carefully evaluated for currently being used by the project (from Udacity env), installed version, and the most recent version as called by on Apr 2022.

# Reporting
Model reporting section carries out metric reporting as well as allowing API setup using Flask framework and curl (for POST)/requests (for GET) calls embedded in apicalls python script.

- Reporting file generates and saves confusion matrix as well as predicted values.

- App file contains 1 endpoint for prediction (the parameter to pass is the file location) and 3 endpoints for scoring, summary statistics, and diagnostics, respectively.

- API Call file uses subprocess and requests to call API endpoints, retrieve the outputs (after changing from bytes to more manageable output), and saves the result.

# Process Automation
Existing codes have been edited to 1) prevent overwrites (as this is critical but not adequately addressed in the original project) and 2) ensure tracking and logging for key fields such as date.

- Noteworthy changes:

- After changing the config file - ingestion file was edited to include date for re-deployment purpose runs. This allows creation of new ingestedfiles list and finaldata csv that is used for model re-training.

- This type of checking for re-deployment happens for training, scoring, deployment, diagnostics, app, and apicalls to ensure that existing data is not overwritten and the new runs are kept tracked. For the purpose of testing, I also set the low new score to ensure that I can "force" the fullprocess code (this is the end-to-end re-deployment code) can execute.

- Confusionmatrix2.png and apireturns2.txt include results of this re-training.

- Cronjob.txt includes the setup for running the fullprocess.py every 10 minutes (tested from Udacity env) 
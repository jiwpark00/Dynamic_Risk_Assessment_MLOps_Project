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
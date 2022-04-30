from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
from datetime import datetime


#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 

# read the model and test_data
model = pickle.load(open('practicemodels/trainedmodel.pkl', 'rb'))
datalocation = os.getcwd() + '/' + test_data_path + '/testdata.csv'  
test_data = pd.read_csv(datalocation)

#################Function for model scoring
def score_model(model, test_data):
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file
    y_true = test_data['exited']
    y_test_cols = test_data.iloc[:,1:-1].values 
    
    f1_result = metrics.f1_score(y_true, model.predict(y_test_cols))
    
    return f1_result

f1_output = score_model(model,test_data)

def write_score(score=None):
    if config['output_model_path'] == "practicemodels":
        f1_output = f1_output # e.g., we don't pass in a score
        output_path = os.path.join(config['output_model_path']) 
        f1_file = open(os.getcwd() + '/' + output_path + '/' + 'latestscore.txt',"w")
        f1_file.write(str(f1_output))
        f1_file.close()
    else:
        f1_output = score
        output_path = os.path.join(config['output_model_path'])
        todays_date = datetime.today().strftime('%Y-%m-%d') 
        f1_file = open(os.getcwd() + '/' + output_path + '/' + 'latestscore_' + todays_date + '.txt',"w")
        f1_file.write(str(f1_output))
        f1_file.close()

write_score()
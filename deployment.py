from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json



##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path']) 

# read the model
model_path = os.path.join(config['output_model_path']) 
lr = pickle.load(open(model_path+'/trainedmodel.pkl', 'rb'))

# read score and ingestedfiles
latestscore_path = os.getcwd() + '/' + model_path + '/latestscore.txt'
latestscore_ = pd.read_csv(latestscore_path)
ingestfiles_path = os.getcwd() + '/' + dataset_csv_path + '/ingestedfiles.txt'
ingestfiles_ = pd.read_csv(ingestfiles_path,header=None)

####################function for deployment
def store_model_into_pickle(model,latestscore_,ingestfiles_):
    #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    
    pickle.dump(lr, open(prod_deployment_path+'/trainedmodel.pkl', 'wb'))
    
    latestscore = list(latestscore_)
    ingestfiles = list(ingestfiles_[0].values)
    
    latestscore_to_prod = open(os.getcwd() + '/' + prod_deployment_path + '/' + 'latestscore.txt',"w")
    for score in latestscore:
        latestscore_to_prod.write(score + "\n")
    latestscore_to_prod.close()
    
    ingestfiles_to_prod = open(os.getcwd() + '/' + prod_deployment_path + '/' + 'ingestedfiles.txt',"w")
    for file in ingestfiles:
        ingestfiles_to_prod.write(file + "\n")
    ingestfiles_to_prod.close()

store_model_into_pickle(model=lr,latestscore_=latestscore_,ingestfiles_=ingestfiles_)


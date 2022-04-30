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
# todays_date = '2022-04-29'
todays_date = datetime.today().strftime('%Y-%m-%d')

# read score and ingestedfiles
def import_files():
    # if scenarios were for training
    if config['output_model_path'] == 'practicemodels':

        # read the model
        model_path = os.path.join(config['output_model_path']) 
        lr = pickle.load(open(model_path+'/trainedmodel.pkl', 'rb'))

        latestscore_path = os.getcwd() + '/' + model_path + '/latestscore.txt'
        latestscore_ = pd.read_csv(latestscore_path)
        ingestfiles_path = os.getcwd() + '/' + dataset_csv_path + '/ingestedfiles.txt'
        ingestfiles_ = pd.read_csv(ingestfiles_path,header=None)
    else:
        # read the model
        model_path = os.path.join(config['output_model_path']) 
        lr = pickle.load(open(model_path+'/trainedmodel_' + todays_date + '.pkl', 'rb'))

        latestscore_path = os.getcwd() + '/' + model_path + '/latestscore_' + todays_date + '.txt'
        latestscore_ = pd.read_csv(latestscore_path)
        ingestfiles_path = os.getcwd() + '/' + dataset_csv_path + '/ingestedfiles_' + todays_date + '.txt'
        ingestfiles_ = pd.read_csv(ingestfiles_path,header=None)

    return lr, latestscore_, ingestfiles_

####################function for deployment
def store_model_into_pickle(model,latestscore,ingestfiles):
    #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    
    if config['output_model_path'] == 'practicemodels':
        pickle.dump(model, open(prod_deployment_path+'/trainedmodel.pkl', 'wb'))
    
        latestscore = list(latestscore)
        ingestfiles = list(ingestfiles[0].values)
        
        latestscore_to_prod = open(os.getcwd() + '/' + prod_deployment_path + '/' + 'latestscore.txt',"w")
        for score in latestscore:
            latestscore_to_prod.write(score + "\n")
        latestscore_to_prod.close()
        
        ingestfiles_to_prod = open(os.getcwd() + '/' + prod_deployment_path + '/' + 'ingestedfiles.txt',"w")
        for file in ingestfiles:
            ingestfiles_to_prod.write(file + "\n")
        ingestfiles_to_prod.close()

    else:
        pickle.dump(model, open(prod_deployment_path+'/trainedmodel_' + todays_date + '.pkl', 'wb'))
    
        latestscore = list(latestscore)
        ingestfiles = list(ingestfiles[0].values)
        
        latestscore_to_prod = open(os.getcwd() + '/' + prod_deployment_path + '/' + 'latestscore_' + todays_date + '.txt',"w")
        for score in latestscore:
            latestscore_to_prod.write(score + "\n")
        latestscore_to_prod.close()
        
        ingestfiles_to_prod = open(os.getcwd() + '/' + prod_deployment_path + '/' + 'ingestedfiles_' + todays_date + '.txt',"w")
        for file in ingestfiles:
            ingestfiles_to_prod.write(file + "\n")
        ingestfiles_to_prod.close()

lr_val, latestscore_val, ingestfiles_val = import_files()
store_model_into_pickle(model=lr_val,latestscore=latestscore_val,ingestfiles=ingestfiles_val)
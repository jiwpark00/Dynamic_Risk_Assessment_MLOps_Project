
import pandas as pd
import numpy as np
import timeit
import os
import pickle
import json
import requests
import subprocess
from datetime import datetime

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 

full_test_data_path = os.getcwd() + '/' + test_data_path + '/testdata.csv'
test_df = pd.read_csv(full_test_data_path)

# todays_date = '2022-04-29'
todays_date = datetime.today().strftime('%Y-%m-%d')

model_path = os.path.join(config['prod_deployment_path']) 
if config['output_model_path'] == 'practicemodels':
    lr = pickle.load(open(model_path+'/trainedmodel.pkl', 'rb'))
else: # this for re-deployment
    lr = pickle.load(open(model_path+'/trainedmodel_' + todays_date + '.pkl', 'rb'))

dataset_path = os.getcwd() + '/' + dataset_csv_path + '/finaldata.csv'
dataset_for_stats = pd.read_csv(dataset_path)
numeric_cols = dataset_for_stats.columns[1:-1] # excluding first column and last column that's actually a target

##################Function to get model predictions
def model_predictions(df, model):
    #read the deployed model and a test dataset, calculate predictions
    preds = model.predict(df.iloc[:,1:-1].values)

    return preds #return value should be a list containing all predictions

##################Function to get summary statistics
def dataframe_summary():
    #calculate summary statistics here
    metric_list = []
    
    for col in numeric_cols:
        col_vals = {}
        col_vals['column_name'] = col
        col_vals['mean'] = np.mean(dataset_for_stats[col].values)
        col_vals['median'] = np.median(dataset_for_stats[col].values)
        col_vals['std'] = np.std(dataset_for_stats[col].values)
        
        metric_list.append(col_vals)

    return metric_list #return value should be a list containing all summary statistics

def check_NA():
    # measure the percentage of NA values in each numeric dataset columns
    for col in numeric_cols:
        nas=list(dataset_for_stats.isna().sum())
        napercents=[nas[i]/len(dataset_for_stats.index) for i in range(len(nas))]
    
    return napercents

##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py
    starttime = timeit.default_timer()
    os.system('python ingestion.py')
    ingestion_time = timeit.default_timer() - starttime
    
    starttime = timeit.default_timer()
    os.system('python training.py')
    training_time = timeit.default_timer() - starttime    
    
    #return a list of 2 timing values in seconds
    return ingestion_time, training_time

##################Function to check dependencies
def outdated_packages_list():
    #get a list of details for each package
    # shows the name of Python module being used - from requirements
    requirements = pd.read_csv('requirements.txt',sep="==",header=None)
    requirements.columns = ['package_name', 'current_version']
    
    # shows the current installed Python module - from pip list
    installed = subprocess.check_output(['pip', 'list', '--format', 'json'])
    parsed_results = json.loads(installed)
    installed_df = pd.DataFrame([(element["name"], element["version"]) for element in parsed_results])
    
    installed_df.columns = ['package_name','installed_version']
    requirements_installed = pd.merge(requirements, installed_df, how='inner', left_on=['package_name'],right_on=['package_name'])
    
    # the most recent python module available - from pip

    most_recent = []

    for package in list(requirements_installed['package_name']):
        response = requests.get(f'https://pypi.org/pypi/{package}/json')
        version = response.json()['info']['version']
        most_recent.append((package, version))
    
    most_recent_df = pd.DataFrame(most_recent)
    most_recent_df.columns = ['package_name','most_recent_version']
    requirements_installed_recent = pd.merge(requirements_installed, most_recent_df, how='inner', left_on=['package_name'],right_on=['package_name'])    
    
    # we will return this to the production deployment
    
    if config['output_model_path'] == 'practicemodels':
        output_path_dependencies = os.getcwd() + '/' + os.path.join(config['prod_deployment_path']) + '/dependencies_status.csv'
        requirements_installed_recent.to_csv(output_path_dependencies,index=False)
    else:
        output_path_dependencies = os.getcwd() + '/' + os.path.join(config['prod_deployment_path']) + '/dependencies_status_' + todays_date + '.csv'
        requirements_installed_recent.to_csv(output_path_dependencies,index=False)

    return requirements_installed_recent

model_predictions(test_df,lr)
dataframe_summary()
check_NA()
execution_time()
outdated_packages_list()





    

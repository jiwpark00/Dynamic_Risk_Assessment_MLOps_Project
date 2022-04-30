from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import scoring
from diagnostics import *
import json
import os
from datetime import datetime

######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path']) 

# todays_date = '2022-04-29'
todays_date = datetime.today().strftime('%Y-%m-%d')

# read the model and test_data
model_path = os.path.join(config['output_model_path']) 
if config['output_model_path'] == 'practicemodels':
    prediction_model = pickle.load(open(model_path+'/trainedmodel.pkl', 'rb'))
else:
    prediction_model = pickle.load(open(model_path+'/trainedmodel_' + todays_date + '.pkl', 'rb'))
datalocation = os.getcwd() + '/' + test_data_path + '/testdata.csv'  
test_data = pd.read_csv(datalocation)

# Adding this to test the app successfully loads
@app.route("/", methods=["GET"])
def homepage():
    return {"Welcome: ": "Here is home page"}

######################Prediction Endpoint
@app.route("/prediction", methods=["POST"])
def predict():        
    #call the prediction function you created in Step 3
    # if dataset_location is not passed, this API call doesn't error - it still returns a result
    dataset_location_received = request.args.get('dataset_location')
    default_data = pd.read_csv(dataset_location_received)
    preds = model_predictions(default_data,prediction_model)
    preds = np.array2string(preds, precision=2, separator=',',
                      suppress_small=True)
    return preds #add return value for prediction outputs

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def scoring_stats():        
    #check the score of the deployed model
    f1_output = scoring.score_model(prediction_model, test_data)
    return {"f1 output is:" : str(f1_output)} #add return value (a single F1 score number)

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def summary_stats():        
    #check means, medians, and modes for each column
    summarystats = dataframe_summary()
    return {"Here is the summary stats": summarystats}#return a list of all calculated summary statistics

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostics_stats():        
    #check timing and percent NA values
    na_p = check_NA()
    
    execute_times = execution_time()
    
    dependencies = outdated_packages_list()
    dependencies = dependencies.to_dict() # this allows writing output as json
    
    return {'NA percent is (from each numeric col in the order)': na_p, 'execution time is (0 = training, 1 = ingestion): ': execute_times, \
           'List of dependencies and details is shown here: ': dependencies}#add return value for all diagnostics

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)

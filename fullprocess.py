import apicalls
import reporting
from diagnostics import model_predictions, test_df, dataframe_summary, check_NA, execution_time, outdated_packages_list
import deployment
import json
import os
import pandas as pd
import ingestion
import pickle
import scoring
import training

# Load config.json and correct path variable
with open('config.json', 'r') as f:
    config = json.load(f)

# Check and read new data
# first, read ingestedfiles.txt
prod_deployment_path = os.path.join(config['prod_deployment_path'])
source_folder_path = config['input_folder_path']

ingested_files_list = pd.read_csv(
    os.getcwd() +
    '/' +
    prod_deployment_path +
    '/' +
    'ingestedfiles.txt',
    header=None)[0].values

# second, determine whether the source data folder has files that aren't
# listed in ingestedfiles.txt
source_directory = os.getcwd() + '/' + source_folder_path
source_data_files = os.listdir(source_directory)

# this is much quick with set than for loop
new_files = list(set(source_data_files) - set(ingested_files_list))

# Deciding whether to proceed, part 1
# if you found new data, you should proceed. otherwise, do end the process here
if len(new_files) == 0:
    print("No new files so exiting")
    quit()  # this will exit the program
else:
    print("We found new files")
    print(new_files)

# If there was/were new data then below will run
new_df = ingestion.merge_multiple_dataframe()
new_df = new_df.reset_index(drop=True)
new_df["exited"] = new_df["exited"].astype(
    float)  # object type affects the data later

ingestion.data_write()  # writes the finaldata that could be used for training as needed

# Checking for model drift
# check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
# Load the latest model score
latest_model_score = pd.read_csv(
    os.getcwd() +
    '/' +
    prod_deployment_path +
    '/' +
    'latestscore.txt',
    header=None)
# change from list to float
latest_model_score = float(latest_model_score[0].values)


latest_model = pickle.load(
    open(
        prod_deployment_path +
        '/trainedmodel.pkl',
        'rb'))
new_result = scoring.score_model(latest_model, new_df)
scoring.write_score(score=new_result)

# just for testing
# new_result = 0.2

if new_result < latest_model_score:
    print("There is a drift so we will have to re-train")

else:
    print("There is no drift so exiting")
    quit()

# ##################Deciding whether to proceed, part 2
# #if you found model drift, you should proceed. otherwise, do end the process here
# # This allows us to overwrite the constant for trainingdata
# There are dependencies required so loading this here

training.trainingdata = new_df

training.train_model()

# ##################Re-deployment
# #if you found evidence for model drift, re-run the deployment.py script

# #we want to avoid overwriting existing in case we want to revert back
# #as a result, we need to note these re-deployment with specific indicators
model, new_scores, new_files_read = deployment.import_files()
# new_scores.columns = ['0.2'] # only run for testing

deployment.store_model_into_pickle(model, new_scores, new_files_read)
# ##################Diagnostics and reporting
# #run diagnostics.py and reporting.py for the re-deployed model

model_predictions(test_df, model)
dataframe_summary()
check_NA()
execution_time()
outdated_packages_list()
print("Diagnostics run is completed")

reporting.score_model()
print("Reporting confusion matrix is completed")

apicalls.run_api()  # this executes API runs
print("API call is completed")

import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from diagnostics import *
from datetime import datetime

# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])

full_test_data_path = os.getcwd() + '/' + test_data_path + '/testdata.csv'
test_df = pd.read_csv(full_test_data_path)

# todays_date = '2022-04-29'
todays_date = datetime.today().strftime('%Y-%m-%d')

model_path = os.path.join(config['prod_deployment_path'])
if config['output_model_path'] == 'practicemodels':
    lr = pickle.load(open(model_path + '/trainedmodel.pkl', 'rb'))

    matrix_path = os.path.join(config['output_model_path'])
    full_matrix_path = matrix_path + '/confusionmatrix.png'

else:  # this for re-deployment
    lr = pickle.load(
        open(
            model_path +
            '/trainedmodel_' +
            todays_date +
            '.pkl',
            'rb'))

    matrix_path = os.path.join(config['output_model_path'])
    full_matrix_path = matrix_path + '/confusionmatrix2.png'


# Function for reporting
def score_model():
    # calculate a confusion matrix using the test data and the deployed model
    # write the confusion matrix to the workspace
    preds = model_predictions(test_df, lr)
    actuals = list(test_df['exited'])

    c_matrix = metrics.confusion_matrix(y_true=actuals, y_pred=preds)
    classes = ["0", "1"]

    df_c_matrix = pd.DataFrame(c_matrix, index=classes, columns=classes)
    c_matrix_plot = sns.heatmap(df_c_matrix, annot=True)
    c_matrix_plot.figure.savefig(full_matrix_path)  # save the confusion matrix


if __name__ == '__main__':
    score_model()

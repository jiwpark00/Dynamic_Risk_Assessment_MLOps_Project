import pandas as pd
import numpy as np
import os
import json
from datetime import datetime




#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']
                      
#############Function for data ingestion
def merge_multiple_dataframe():
    
    end_df = pd.DataFrame(columns=['corporation', 'lastmonth_activity', 'lastyear_activity',
       'number_of_employees', 'exited'])
    #check for datasets, compile them together, and write to an output file
    input_directory = os.getcwd() + '/' + input_folder_path
    filenames = os.listdir(input_directory)
    for each_file in filenames:
        if each_file == '.DS_Store': # resolving this Mac specific issue
            continue
        filepath = input_folder_path + '/' + each_file
        df1 = pd.read_csv(filepath)
        end_df = end_df.append(df1)
    if config['input_folder_path'] == "practicedata":
        ingestedfiles = open(os.getcwd() + '/' + output_folder_path + '/' + 'ingestedfiles.txt',"w")
        for file in filenames:
            ingestedfiles.write(file + "\n")
        ingestedfiles.close()
    else:
        todays_date = datetime.today().strftime('%Y-%m-%d')
        ingestedfiles = open(os.getcwd() + '/' + output_folder_path + '/' + 'ingestedfiles_' + todays_date + '.txt',"w")
        for file in filenames:
            ingestedfiles.write(file + "\n")
        ingestedfiles.close()
    
    end_df.drop_duplicates(inplace=True)

    return end_df

def data_write():

    result = merge_multiple_dataframe()

    # This was added due to re-testing on Mac causing some errors for my env
    clean_cols = []
    for col in result.columns:
        if col != 'Unnamed: 0':
            clean_cols.append(col)

    result = result[clean_cols] # handling error during different env

    # This block is added to ensure process automation does not overwrite the existing training data
    if config['input_folder_path'] == "practicedata":
        output_name = os.getcwd() + '/' + output_folder_path + '/finaldata.csv'
        result.to_csv(output_name, index=False)
    else:
        # todays_date = datetime.today().strftime('%Y-%m-%d')
        todays_date = '2022-04-29'
        output_name = os.getcwd() + '/' + output_folder_path + '/finaldata_' + todays_date + '.csv'
        result.to_csv(output_name, index=False)

    print("New finaldata has been written")

if __name__ == '__main__':
    merge_multiple_dataframe()
    data_write()

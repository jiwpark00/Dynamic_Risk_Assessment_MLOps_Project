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
        filepath = input_folder_path + '/' + each_file
        df1 = pd.read_csv(filepath)
        end_df = end_df.append(df1)
    
    end_df.drop_duplicates(inplace=True)

    return end_df

result = merge_multiple_dataframe()

# This was added due to re-testing on Mac causing some errors for my env
clean_cols = []
for col in result.columns:
	if col != 'Unnamed: 0':
		clean_cols.append(col)

result = result[clean_cols] # handling error during different env

output_name = os.getcwd() + '/' + output_folder_path + '/finaldata.csv'
result.to_csv(output_name, index=False)

if __name__ == '__main__':
    merge_multiple_dataframe()

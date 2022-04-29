import requests
import subprocess
from subprocess import DEVNULL, STDOUT, check_call
import os
import json

#Specify a URL that resolves to your workspace

# This is for Udacity workspace
# URL = "http://172.17.0.2"

# This is for normal env
URL = "http://127.0.0.1"

#Call each API endpoint and store the responses
#Decode utf-8 is run to change from bytes
response1 = subprocess.run(['curl',"-X", "POST", URL+':8000/prediction?dataset_location=testdata/testdata.csv'],capture_output=True).stdout #put an API call here
response1 = response1.decode("utf-8")
response2 = requests.get(URL+':8000/scoring').content #put an API call here
response2 = response2.decode("utf-8")
response3 = requests.get(URL+':8000/summarystats').content#put an API call here
response3 = response3.decode("utf-8")
response4 = requests.get(URL+':8000/diagnostics').content#put an API call here
response4 = response4.decode("utf-8")

#combine all API responses
responses = response1 + '\n' + response2 + response3 + response4 #combine reponses here
#write the responses to your workspace

#saving output
# loading config
with open('config.json','r') as f:
    config = json.load(f) 
dataset_csv_path = os.path.join(config['output_model_path']) 
output_path = os.getcwd() + '/' + dataset_csv_path + '/apireturns.txt'

text_file = open(output_path, "w")
n = text_file.write(responses)
text_file.close()

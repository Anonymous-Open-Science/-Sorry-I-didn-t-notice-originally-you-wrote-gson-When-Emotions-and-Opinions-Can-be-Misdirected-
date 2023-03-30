import pandas as pd
import numpy as np
import os
#from sklearn.metrics import classification_report
from sklearn.metrics import *
import datetime

BATCH = '270800'
MODEL = 'openai-api'
report_path = 'classification_report-42.v2.csv'
WRITE_REPORT = False

data_files_dict = [
    
    {'file': '7.1.BERT4SentiSE.SO-ML-classified-'+MODEL+'.csv', 'dataset': '7.1.BERT4SentiSE-StackOverflow'}
    ]


def write_report_internal(dataset_name, df, report_path, file_name):
    target_names = ['positive', 'negative', 'neutral']
  
    # add model column and default value to df at the beginning of the dataframe
    df.insert(0, 'model', MODEL)
    df.insert(1, 'dataset', dataset_name)
    df.insert(2, 'test_dataset_type', 'ML')
    
    # get current date and time yyyy-mm-dd hh:mm
    now = datetime.datetime.now()
    dt_string = now.strftime("%Y-%m-%d %H:%M")
    df['time'] = dt_string
    df['data_file'] = file_name
    df['train_noise'] = 'none'
    df['batch'] = BATCH
    
    # read the existing report file
    df_report = pd.read_csv(report_path)

    # append the new report to the existing report
    df_report = df_report.append(df, ignore_index=True)

    # write the report to the file
    df_report.to_csv(report_path, index=False)


def write_report(dataset_name, y_true, y_pred, report_path, file_name):
    target_names = ['positive', 'negative', 'neutral']

    # # print the report in readable format
    print(classification_report(y_true, y_pred, target_names=target_names))
    
    # report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    report = classification_report(y_true, y_pred, output_dict=True)
    df = pd.DataFrame(report).transpose()
    # convert the index to a column 'class'
    df.reset_index(level=0, inplace=True)
    df.rename(columns={'index': 'class'}, inplace=True)
    write_report_internal(dataset_name, df, report_path, file_name)

for data_file_dict in data_files_dict:
    data_file = data_file_dict['file']
    dataset_name = data_file_dict['dataset']

    # load the csv files into data frames
    df_data = pd.read_csv(data_file)

    # remove rows with empty label
    df_data = df_data[df_data['label'].notna()]

    y_test = list(df_data["label"])
    y_pred = list(df_data["label_"+MODEL])

    # convert label values from float to int
    y_test = [int(i) for i in y_test]
    y_pred = [int(i) for i in y_pred]



    #print(model)
    #print("*"*50)
    print(classification_report(y_test, y_pred))

    # print f1 score
    print("OpenAI ", dataset_name, f1_score(y_test, y_pred, average='weighted'))
    #print("-"*50)

    file_name = os.path.basename(data_file)

    if WRITE_REPORT:
        write_report(dataset_name, y_test, y_pred, report_path, file_name)



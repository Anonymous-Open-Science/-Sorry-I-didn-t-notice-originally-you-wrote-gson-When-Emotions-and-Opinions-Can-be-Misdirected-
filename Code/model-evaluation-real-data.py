import pandas as pd
import numpy as np
import os
from sklearn.metrics import *
import datetime

# suppress warnings
import warnings
warnings.filterwarnings("ignore")

report_path = 'classification_report-42.v2.csv'

data_file = 'Online-review-dataset.xlsx'


models = [
        {'model': 'simcse','dataset': '4.3.BinLin-SO', 'train_noise': 'none', 'column': 'BL-SO-None'},
          {'model': 'simcse','dataset': '4.3.BinLin-SO', 'train_noise': 'Auto-Emotion-D-T', 'column': 'BL-SO-Noise2'},
          {'model': 'simcse','dataset': '5.Senti4SD-GoldStandard-EmotionPolarity', 'train_noise': 'none', 'column': 'Senti4SD-None'},
          {'model': 'simcse','dataset': '5.Senti4SD-GoldStandard-EmotionPolarity', 'train_noise': 'Auto-Emotion-D-T', 'column': 'Senti4SD-Noise2'},
          {'model': 'simcse','dataset': '6.Opiner-StackOverflow', 'train_noise': 'none', 'column': 'Opiner-None'},
          {'model': 'simcse','dataset': '6.Opiner-StackOverflow', 'train_noise': 'Auto-Emotion-D-T', 'column': 'Opiner-Noise2'},
          {'model': 'simcse','dataset': '7.1.BERT4SentiSE-StackOverflow', 'train_noise': 'none', 'column': 'BERT-None'},
          {'model': 'simcse','dataset': '7.1.BERT4SentiSE-StackOverflow', 'train_noise': 'Auto-Emotion-D-T', 'column': 'BERT-Noise2'},
          {'model': 'openai-chat','dataset': 'openai', 'train_noise': 'none', 'column': 'label_openai-chat'},
          {'model': 'openai-api','dataset': 'openai', 'train_noise': 'none', 'column': 'label_openai-api'},
          ]
TEST_DATA_SET_TYPE = 'SO-API'
BATCH = '291920'
SETTING = BATCH #'261350' # SO-LIB file labeling setting


# # load the csv files into data frames
# df_data = pd.read_csv(data_file)

# load the excel files into data frames
df_data = pd.read_excel(data_file)

# remove rows with empty label
df_data = df_data[df_data['label'].notna()]

y_test = list(df_data["label"])

# keep rows with MLSet = Test
# df_data = df_data[df_data['MLSet'] == 'Test']

# convert y_test values to int
y_test = [int(i) for i in y_test]


def write_report_internal(model_dict, df, report_path, file_name):
    target_names = ['positive', 'negative', 'neutral']
  
    # add model column and default value to df at the beginning of the dataframe
    df.insert(0, 'model', model_dict['model'])
    df.insert(1, 'dataset', model_dict['dataset'])
    df.insert(2, 'test_dataset_type', TEST_DATA_SET_TYPE)
    
    # get current date and time yyyy-mm-dd hh:mm
    now = datetime.datetime.now()
    dt_string = now.strftime("%Y-%m-%d %H:%M")
    df['time'] = dt_string
    df['data_file'] = file_name
    df['train_noise'] = model_dict['train_noise']
    df['batch'] = BATCH
    df['setting'] = SETTING
    
    # read the existing report file
    df_report = pd.read_csv(report_path)

    # append the new report to the existing report
    df_report = df_report.append(df, ignore_index=True)

    # write the report to the file
    df_report.to_csv(report_path, index=False)


def write_report(model_dict, y_true, y_pred, report_path, file_name):
    target_names = ['positive', 'negative', 'neutral']

    # # print the report in readable format
    print(classification_report(y_true, y_pred, target_names=target_names))
    
    # report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    report = classification_report(y_true, y_pred, output_dict=True)
    df = pd.DataFrame(report).transpose()
    # convert the index to a column 'class'
    df.reset_index(level=0, inplace=True)
    df.rename(columns={'index': 'class'}, inplace=True)
    write_report_internal(model_dict, df, report_path, file_name)

for model_dict in models:
    column = model_dict['column']
    y_pred = list(df_data[column])
    
    # convert y_pred values to int
    y_pred = [int(i) for i in y_pred]

    #print(model)
    #print("*"*50)
    #print(classification_report(y_test, y_pred))
    
    # print f1 score in 2 decimal places and in tabular format with dataset name
    print(model_dict['dataset'], round(f1_score(y_test, y_pred, average='weighted'), 2))

    #print("-"*50)


    file_name = os.path.basename(data_file)
    write_report(model_dict, y_test, y_pred, report_path, file_name)




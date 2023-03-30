import pandas as pd

# suppress warnings
import warnings
warnings.filterwarnings("ignore")

def check_matached():
    expected_label = [0, 1, 2]
    label = [0, 1, 1]

    df = pd.DataFrame({'expected_label': expected_label, 'label': label})

    # add a new column to the data frame with the name "matched". 
    # set the value of the column to 1 if the expected label and the label match, 0 otherwise.
    df['matched'] = df.apply(lambda row: 1 if row['expected_label'] == row['label'] else 0, axis=1)


def keep_matched():
    # load the csv file into a data frame
    filename = 'emotional-sentences_2023_03_10_2_relevant.csv'
    df = pd.read_csv(filename)

    # only rows where macthed is 1
    df = df[df['matched'] == 1]

    # remove duplicate rows based on the text column
    df = df.drop_duplicates(subset=['text'])

    # rename the file, replace relevant with matched
    filename = filename.replace('relevant', 'matched')

    # save the file
    df.to_csv(filename, index=False)

def add_emotion_noise_to_data(df_data, data_label, df_emotion, emotion_label, MLSet, add_noise):
    # #make it lower case
    # MLSet = MLSet.lower()
    # # change MLSet column values to lower case
    # df_emotion['MLSet'] = df_emotion['MLSet'].str.lower()
    # df_data['MLSet'] = df_data['MLSet'].str.lower()

    # pick rows from df_emotion where the label is emotion_label and the MLSet is the same as the MLSet of df_data
    df_noise = df_emotion[(df_emotion['label'] == emotion_label) & (df_emotion['MLSet'] == MLSet)]

   
    # pick rows from df_data where the label is data_label
    df_data_emotion = df_data[(df_data['label'] == data_label) & (df_data['MLSet'] == MLSet)]

    # if df_noise length is less than df_data_emotion length, then repeat the df_noise rows until it is equal to df_data_emotion length
    if len(df_noise) < len(df_data_emotion):
        df_noise = df_noise.append([df_noise] * (len(df_data_emotion) // len(df_noise)), ignore_index=True)

    # add two new columns to df_data_emotion with name 'noise' and 'noise-label'
    # noise column should come from df_emotion_pos. noise-label should be 0. 
    df_data_emotion['noise'] = df_noise.sample(n=len(df_data_emotion))['text'].values
    df_data_emotion['noise-label'] = emotion_label

    # concatenate noise and text-orig values into column called 'text'
    if add_noise:
        df_data_emotion['text'] = df_data_emotion['noise'] + ' ' + df_data_emotion['text-orig']
    else:
        df_data_emotion['text'] = df_data_emotion['text-orig']

    return df_data_emotion

def add_emotion_noise(data_file, emotion_file, MLSet, add_noise, double_set):
    # load the csv files into data frames
    df_data = pd.read_csv(data_file)
    df_emotion = pd.read_csv(emotion_file)

    # rename the column 'text' to 'text-orig' of df_data
    df_data.rename(columns={'text': 'text-orig'}, inplace=True)


    df_data_pos = add_emotion_noise_to_data(df_data, 0, df_emotion, 1, MLSet, add_noise)
    df_data_neg = add_emotion_noise_to_data(df_data, 1, df_emotion, 0, MLSet, add_noise)
    df_data_neu = add_emotion_noise_to_data(df_data, 2, df_emotion, 0, MLSet, add_noise)

    # combine all the data frames df_data_pos, df_data_neg, df_data_neu
    df_data_1 = df_data_pos.append(df_data_neg, ignore_index=True)
    df_data_1 = df_data_1.append(df_data_neu, ignore_index=True)


    
    if double_set:
        df_data_pos = add_emotion_noise_to_data(df_data, 0, df_emotion, 0, MLSet, add_noise)
        df_data_neg = add_emotion_noise_to_data(df_data, 1, df_emotion, 1, MLSet, add_noise)
        df_data_neu = add_emotion_noise_to_data(df_data, 2, df_emotion, 1, MLSet, add_noise)

        # combine all the data frames
        df_data2 = df_data_pos.append(df_data_neg, ignore_index=True)
        df_data2 = df_data2.append(df_data_neu, ignore_index=True)

        df_data = df_data_1.append(df_data2, ignore_index=True)
    else:
        df_data = df_data_1

    return df_data

# keep_matched()

emotion_file = ''

data_file = '7.1.BERT4SentiSE.SO-ML.csv'

data_files = [  '4.3.StackOverflow-data-ML.csv',
                ]

# create a dictionary of MLSet and its noise
MLSet_noise = {'Train': True, 'Validate': False, 'Test': False}
# join values of MLSets with '-' for which noise is True.
MLSets = [MLSet for MLSet, noise in MLSet_noise.items() if noise]
MLSet_str = '-'.join(MLSets)
file_suffix = '-auto-emotion-double-'+MLSet_str+'.csv'

for data_file in data_files:
    df_master = pd.DataFrame()
    
    for MLSet in MLSet_noise:
        df = add_emotion_noise(data_file, emotion_file, MLSet, MLSet_noise[MLSet], True)
        df_master = df_master.append(df, ignore_index=True)

    # replace the file name with the new file name
    data_file = data_file.replace('.csv', file_suffix)

    # save the file
    df_master.to_csv(data_file, index=False)







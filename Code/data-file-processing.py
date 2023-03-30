import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

csv_file_path = ''

# get folder location from file path
csv_folder_path = os.path.dirname(csv_file_path)

# load the csv file in a dataframe
df = pd.read_csv(csv_file_path)

# get the list of all the columns
def get_columns():
    return df.columns

def get_ml_sets(split_files=False):

    global df

    # shuffle the dataframe
    df = df.sample(frac=1).reset_index(drop=True)

    # sometimes, df may have a column 'label_orig' whose values may need to be mapped to the values in the 'sentiment' column
    # label_orig values can be -1, 0, or 1
    # here is the map for label_orig values to sentiment values:
    # -1 -> negative
    # 0 -> neutral
    # 1 -> positive

    # if there is a column named 'code', then we shall have to add a new column named 'sentiment' and populate it with the values from the 'label_orig' column
    # there should be mapping between the values in the 'code' column and the values in the 'sentiment' column
    # for example, if the 'code' column has values -1, 0, and 1, then the 'sentiment' column should have values 'negative', 'neutral', and 'positive' respectively
    # the mapping should come from a dictionary. The dictionary should be defined here.
    label_to_sentiment_map = {-1: 'negative', 0: 'neutral', 1: 'positive'}
    if 'code' in get_columns() and 'sentiment' not in get_columns():
        df['sentiment'] = df['code'].replace(label_to_sentiment_map)


    # sometimes, df may not have a label column. We shall have to see if there is a column named 'sentiment'.
    # if there is, we shall have to add a new column named 'label' and populate it with the values from the 'sentiment' column
    # there should be mapping between the values in the 'sentiment' column and the values in the 'label' column
    # for example, if the 'sentiment' column has values 'positive', 'negative', and 'neutral', then the 'label' column should have values 0, 1, and 2 respectively
    # sentiment values should not be case sensitive. Here is the code:
    if 'sentiment' in get_columns() and 'label' not in get_columns():
        df['label'] = df['sentiment'].str.lower().replace({'positive': 0, 'negative': 1, 'neutral': 2})

    # remove any surrounding " or ' from the text
    df['text'] = df['text'].str.strip('"').str.strip("'")

    print(df)

    # apply downscaling to the data
    # get the counts of each label
    label_counts = df['label'].value_counts()
    # get the minimum count
    min_count = label_counts.min()
    max_count = label_counts.max()
   

    # #### UNDER SAMPLING #### Works well with Opiner dataset
    # grouped = df.groupby('label')
    # n_samples = min_count

    # # Use the 'sample' method to randomly select n_samples from each group
    # downsampled = grouped.apply(lambda x: x.sample(n=n_samples))

    # # Reset the index of the downsampled DataFrame
    # df = downsampled.reset_index(drop=True)

    # ### OVER SAMPLING ### Does not work well.
    # grouped = df.groupby('label')
    # n_samples = max_count
    # print('n_samples: ', n_samples)

    # oversampled = grouped.apply(lambda x: x.sample(n=n_samples, replace=True))

    # # Reset the index of the oversampled DataFrame
    # df = oversampled.reset_index(drop=True)

    # split the data into train (70%), validate (20%), and test (10%) sets
    train_ratio = 0.7
    val_ratio = 0.2
    test_ratio = 0.1

    # use the stratify parameter to ensure proportionate labels in each set
    train, test, train_labels, test_labels = train_test_split(df['text'], df['label'], test_size=test_ratio, stratify=df['label'])
    train, val, train_labels, val_labels = train_test_split(train, train_labels, test_size=val_ratio/(train_ratio+val_ratio), stratify=train_labels)

    # create a new column 'ML-set' and populate it with the corresponding values
    df.loc[train.index, 'MLSet'] = 'Train'
    df.loc[val.index, 'MLSet'] = 'Validate'
    df.loc[test.index, 'MLSet'] = 'Test'

    # display the counts of each set and each label
    # print(df['MLSet'].value_counts())
    # print(df['label'].value_counts())


    # add ML suffix to file name of csv_file_path
    csv_file_path_output = csv_file_path.replace('.csv', '-ML.csv')

    # save the dataframe to a csv file
    df.to_csv(csv_file_path_output, index=False)




#get_ml_sets()

def get_cross_validation_sets():
    
    # We have a DataFrame named 'df' containing the data
    # and a column named 'label' containing the labels (0, 1, or 2)

    # create three arrays, one for each label
    label0_data = df[df['label'] == 0].values
    label1_data = df[df['label'] == 1].values
    label2_data = df[df['label'] == 2].values

    # shuffle each array
    np.random.shuffle(label0_data)
    np.random.shuffle(label1_data)
    np.random.shuffle(label2_data)

    # split each array into 10 equal parts
    label0_splits = np.array_split(label0_data, 10)
    label1_splits = np.array_split(label1_data, 10)
    label2_splits = np.array_split(label2_data, 10)

    # combine the splits for each label into a list of 10 splits
    splits = []
    for i in range(10):
        split = np.concatenate([label0_splits[i], label1_splits[i], label2_splits[i]])
        # add a new column with i as the value
        split = np.insert(split, 0, i+1, axis=1)
        splits.append(split)


    # perform 10-fold cross validation
    kf = KFold(n_splits=10, shuffle=True)
    for i, (train_indices, val_indices) in enumerate(kf.split(splits)):
        train_data = np.concatenate([splits[j] for j in train_indices])
        val_data = np.concatenate([splits[j] for j in val_indices])
        
        # print top 10 texts of validation set
        print(val_data[:10,1])
       
        # use the train_data and val_data for training and validation in your model
        # print(f'Fold {i+1}: train data shape = {train_data.shape}, val data shape = {val_data.shape}')

        # train_data contains the Fold, text, sentiment, and label columns
        # text column is at index 1. label column is at index 3
        X_train, X_val, y_train, y_val = list(train_data[:,1]), list(val_data[:,1]), list(train_data[:,3]), list(val_data[:,3])

        # print top 10 texts of validation set
        print(X_val[:10])

        

        # create a new data frame with X_val and y_val
        #df_val = pd.DataFrame({'text': X_val, 'label': y_val})



    # convert splits to dataframe and save to csv file
    columns = ['Fold', 'text', 'sentiment', 'label']
    df_cv = pd.DataFrame(np.concatenate(splits), columns=columns)
    #df_cv.to_csv(os.path.join(csv_folder_path, '07.SO.BERT4SentiSE.Combined-CrossValidation.csv'), index=False)
 
    
# get_cross_validation_sets()
get_ml_sets()

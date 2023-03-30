
import pandas as pd
import numpy as np

import sys

import os
import logging



file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
from config import *
from opinion_extraction import * # for invoking process_raw_text_body

# keywords =  ["slf4j", "log4j"]
# keywords =  ["memcpy", "memmove"]
# keywords = ['pandas', 'numpy']
#keywords = ['lxml', 'beautifulsoup']
#keywords = ['gson', 'jackson']
# keywords = ['gson', '2.8']
keywords = ['gson']

# query = ("SELECT id, body FROM Posts where body like '%mysql%connector%' LIMIT 2")
# query = "SELECT id, body FROM stackoverflow.Posts where (Tags like '%<slf4j>%' and Tags like '%<log4j2>%') order by ViewCount Desc LIMIT 0, 5;"
# query = "SELECT id, body FROM stackoverflow.Posts where (Tags like '%<slf4j>%' and Tags like '%<log4j2>%')  LIMIT 0, 50;"
# query = "SELECT id, body FROM stackoverflow.Posts where (body like '%memcpy%' and body like '%memmove%')  LIMIT 0, 50;"
# query = "SELECT id, text as body FROM stackoverflow.Comments where (text like '%memcpy%' OR text like '%memmove%')  LIMIT 0, 50;"
# query = "SELECT id, text as body FROM stackoverflow.Comments where (text like '%pandas%' and text like '%numpy%')  LIMIT 0, 50;"
#query = "SELECT id, body, creationdate FROM stackoverflow.Posts where (body like '%lxml%' and body like '%beautifulsoup%')  LIMIT 0, 50;"

dual_library_comparison = len(keywords)>1
if dual_library_comparison:
    query = "SELECT id, body, creationdate, lasteditdate FROM stackoverflow.Posts where (body like '%" +keywords[0]+ "%' and body like '%" +keywords[1]+ "%')  LIMIT 0, 50;"
else:
    query = "SELECT id, body, creationdate, lasteditdate FROM stackoverflow.Posts where (body like '% " +keywords[0]+ " %')  LIMIT 0, 50000;"

# check if the keyword[1] is in the form of x.y and denotes version number
if dual_library_comparison:
    if len(keywords[1].split('.')) == 2:
        # check if the keyword[1] contains only digits and dots
        if keywords[1].replace('.', '', 1).isdigit():
            dual_library_comparison = False
            print("dual_library_comparison: ", dual_library_comparison)



def get_file_name():
    file_name = "stackoverflow_"
    for keyword in keywords:
        file_name += keyword + "_"
    file_name = file_name[:-1] + ".csv"
    return file_name

def store_data(myresult, column_names):
    # create a dataframe from the result set after processing body with process_raw_text method and merge the returned sentences array from process_raw_text
    df = pd.DataFrame(myresult, columns=column_names)
    df['body'] = df['body'].apply(process_raw_text_body)

    # add a new date column to the dataframe which will store latest date of creation and last edit
    df['date'] = df[['creationdate', 'lasteditdate']].max(axis=1)


    # write the dataframe to a csv file with name of the keywords
    df.to_csv(data_dir+get_file_name(), index=False)
    #print(df.head())
    logging.info('Data stored in file: '+get_file_name())
    return df

def collect_data():
    if CACHED_DATA == False:
        # Connect to mysql database and select all rows from table
        import pwd
        import mysql.connector
        from mysql.connector import errorcode

        print("Connecting to database...")
        cnx = mysql.connector.connect(user=DB_USERID, password=DB_PWD, database=DB_NAME)
        print("Connected to database")


        cursor = cnx.cursor()

        cursor.execute(query)

        # print the result
        myresult = cursor.fetchall()

        #print column names
        print(cursor.column_names)

        myresult = store_data(myresult, cursor.column_names)
        cursor.close()
        cnx.close()
    else:
        # load myresult dataframe from stored data file. 
        # This is to avoid connecting to mysql database everytime
        myresult = pd.read_csv(data_dir+get_file_name())
        logging.log(LOG_LEVEL_APPLICATION_DEBUG, "Cached opinion data from stack overflow:\n"+str(myresult))

    return myresult

def clearnup_datafile(filepath, keyword):
    # load file as dataframe
    df = pd.read_csv(filepath)
    df['body'] = df['body'].apply(breakdown_sentences)

    # explode the body column to create a new row for each sentence
    df = df.explode('body')



    # move rows with nan in body
    df = df[df['body'].notna()]

    # remove rows with length less than 10
    df = df[df['body'].str.len() > 10]


    # remove rows that does not contain the keyword
    # df = df[df['body'].str.contains(keyword, case=False)]

    #remove rows that does not contain the keyword. some rows can have non string values
    df = df[df['body'].apply(lambda x: keyword in str(x))]

    # save the dataframe to a new cleaned file, add _cleaned to the file name
    df.to_csv(filepath[:-4]+'_cleaned.csv', index=False)

    # split the df so that each part contains 1000 rows maximum
    # and save each part to a new file
    df_split = np.array_split(df, 1000)
    for i, df in enumerate(df_split):
        df.to_csv(filepath[:-4]+'_cleaned_'+str(i)+'.csv', index=False)


def split_file(filepath):
    # load file as dataframe
    df = pd.read_csv(filepath)

    # split the df so that each part contains 1000 rows maximum
    # and save each part to a new file
    df_split = np.array_split(df, 10)
    for i, df in enumerate(df_split):
        df.to_csv(filepath[:-4]+'_cleaned_'+str(i)+'.csv', index=False)

    # convert df to list
    df_list = df.values.tolist()

    # split the df_list into 10 parts and print the first 10 rows of each part
    df_list_split = np.array_split(df_list, 10)
    for i, df in enumerate(df_list_split):
        # append each part to same file
        start = 'head'

        # insert start at the beginning of each part
        df = np.insert(df, 0, start, axis=0)

        # convert df to list
        df_list = df.tolist()
        # print the size of df_list
        print(len(df_list))


        print(df[:10])

    


if __name__ == "__main__":
    # collect_data()
    # clearnup_datafile(data_dir+get_file_name(), keywords[0])

    # load the cleaned file
    df = pd.read_csv(data_dir+get_file_name()[:-4]+'_cleaned.csv')

    split_file(data_dir+get_file_name()[:-4]+'_cleaned.csv')

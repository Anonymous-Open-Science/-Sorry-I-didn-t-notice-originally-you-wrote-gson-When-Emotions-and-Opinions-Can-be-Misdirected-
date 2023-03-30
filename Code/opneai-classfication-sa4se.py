
import pandas as pd

import os
import openai

openai.api_key_path = '.env'
MAX_SENTENCES_FOR_CLASSIFICATION = 1000
PROMPT_SIZE = 10
MAX_TOKEN_SIZE = 512
MODEL = 'api' #  chat or api

TROUBLESHOOTING = False
#TROUBLESHOOTING = True

if TROUBLESHOOTING:
  MAX_SENTENCES_FOR_CLASSIFICATION = 10
  PROMPT_SIZE = 5



data_file = ''

# suppress warnings
import warnings
warnings.filterwarnings("ignore")

def get_model_output_chat(prompt, role = 'user'):
  messages=[
      # # adding the system message deterioates the model performance. SO-BL f1score becomes 0.72 from 0.77 with the system message
      # {"role": 'system', "content": 'You will respond as sentiment detection model where your detected sentiments will strictly be either positive, negative or neutral.'},
      {"role": role, "content": prompt},
  ]


  completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages
  )
  # print(completion)

  model_output = completion.choices[0].message
  return model_output["content"]

def get_model_output_api(prompt):
  response = openai.Completion.create(
    model="text-davinci-003",
    prompt=prompt,
    temperature=0.7,
    max_tokens=MAX_TOKEN_SIZE*4,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
  )

  model_output = response["choices"][0]["text"]
  return model_output

output_suffix = '-classified-openai-'+MODEL+'.csv'
output_file = data_file.replace('.csv', output_suffix)


# read csv data file and load into list
df = pd.read_csv(data_file)
df = df.reset_index(drop=True)

df = df[df['MLSet'] == 'Test']
print('df length: ', len(df))

# load already classified data
df_classified = pd.DataFrame()

try:
  df_classified = pd.read_csv(output_file, encoding='utf-8')

  # drop already classified data based on 'id'. id should match fully.
  df = df[~df['id'].isin(df_classified['id'])]
  df = df.reset_index(drop=True)  

except:
  print('No classified data found')

# keep only MAX_SENTENCES_FOR_CLASSIFICATION rows from df
df = df.head(MAX_SENTENCES_FOR_CLASSIFICATION)



def get_sentiment(model_output):
   # model_output = '\n\n1. Negative\n2. Positive\n3. Negative'

    # split model output into lines
    model_output_lines = model_output.splitlines()
    print('Model output: \n', model_output_lines)

    # skip a line if it does not start with a number
    sentiments = []
    model_labels = []
    for sentiment_line in model_output_lines:
      if len(sentiment_line) > 1:

        try:
          sentiment = sentiment_line.split('.')[1]
        except:
          sentiment = sentiment_line

        sentiment = sentiment.strip()
        print('sentiment: ', sentiment)
        sentiments.append(sentiment)

        # if sentiment contains 'positive' then label = 0
        # if sentiment contains 'negative' then label = 1
        # if sentiment contains 'neutral' then label = 2
        if 'positive' in sentiment.lower():
          model_labels.append(0)
        elif 'negative' in sentiment.lower():
          model_labels.append(1)
        elif 'neutral' in sentiment.lower():
          model_labels.append(2)
        else: # if nothing is found, consider neutral
          model_labels.append(2)

    return sentiments, model_labels

# convert the data frame to a dictionary
df_dict = df.to_dict('records')
data_len = len(df_dict)
print('df_dict length: ', data_len)

if data_len == 0:
  print('No data to classify')
  exit()


df_master = pd.DataFrame()
# loop over the dictionary for PROMPT_SIZE rows.
# create a prompt for each group of PROMPT_SIZE rows
# call the model and get the classification
# append the classification to the dictionary
# write the dictionary to a csv file
try:
  for i in range(0, len(df_dict), PROMPT_SIZE):
      # lines left to process
      lines_left = min(len(df_dict) - i, PROMPT_SIZE)
      
      # create prompt
      prompt = "Classify the sentiment in these technical comments and give results in separate lines:\n"
      prompt = "Classify the sentiment in these technical comments:\n"
      prompt = "Classify the sentiment in these technical comments and give results strictly in "+str(lines_left)+" separate lines (either positive, negative, or neutral):\n"
      for j in range(0, lines_left):
        text = str(df_dict[i+j]['text'])

        # truncate text to MAX_TOKEN_SIZE characters multiple of 4
        # https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
        text = text[:MAX_TOKEN_SIZE*4]

        # last group may not have PROMPT_SIZE sentences
        prompt += str(j+1) + ". \"" + text + "\"\n"

      print('*'*80)
      print(prompt)
      print("Prompt Tokens: "+str(len(prompt)/4))
      print('Data Processed: ', i, ' of ', data_len)

      # call model
      if MODEL == 'api':
        model_output = get_model_output_api(prompt)
      else:
        model_output = get_model_output_chat(prompt)
        
      try:
        sentiments, model_labels = get_sentiment(model_output)
      
        # append the classification to the dictionary
        for j in range(0, lines_left):
          df_dict[i+j]['label_openai-'+MODEL] = model_labels[j]
          df_dict[i+j]['sentiment_openai-'+MODEL] = sentiments[j]
      except Exception as e:
        print(e)
        print('SKIPPING this batch. Error in getting sentiment')
        continue
        
      # get a dataframe with the current rows
      df = pd.DataFrame(df_dict[i:i+lines_left])
      df_master = df_master.append(df)

except Exception as e:
  # print the error
  print(e)
  print('Error occurred. Saving to file with current records: ', len(df_master))    

total_rows = len(df_master)
print('Total rows: ', total_rows)
# append the df_master to the end of df_classified
df_master = df_classified.append(df_master)

df_master.to_csv(output_file, index=False)
print('Saved to file: ', output_file)


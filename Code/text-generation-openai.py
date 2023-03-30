
import pandas as pd
import datetime

import os
import openai

# suppress all warnings
import warnings
warnings.filterwarnings("ignore")


openai.api_key_path = '.env'
SAMPLE_COUNT = 20 # number of sentences to generate per OpenAI API call
file_name = 'emotional-sentences.csv'


# create new data frame
df_master = pd.DataFrame()


# define an array of positive and negative samples
positive_samples = ['appeciate your time', 'appreciate your opinion.', 'excellent question.', 'good luck.', 'Hopefully it will work.', 'loved your solution.', 'thank you.', 'that is great.', 'your idea seems interesting', 'your suggestion is helpful', ]
negative_samples = ['cannot agree', 'hate to disagree.', 'I am afraid, cannot agree.', 'sorry to disagree.', 'very sorry to disagree', 'you are a sad person', 'You are incorrect', 'you have given an absurd logic.', 'you have got a bad idea', 'your opinion is baseless.', ]



def generate_sentences(sentiment, sample, prompt):

  messages=[
      {"role": "user", "content": prompt},
  ]

  completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages
  )
  print(completion)

  message = completion.choices[0].message

  # message = {
  #   "content": "\n\n1. Much appreciated. \n2. Many thanks. \n3. I am grateful. \n4. You're amazing. \n5. Thanks a bunch. \n6. I owe you one. \n7. You're too kind. \n8. Thank you kindly. \n9. My sincere gratitude. \n10. You rock!",
  #   "role": "assistant"
  # }

  # parse the message content to get each sentence separated by a new line
  # and then split the message content by new line to get each sentence
  # in a list
  sentences = message["content"].split("\n")
  # remove blank lines
  sentences = [sentence for sentence in sentences if sentence != ""]

  # remove the preceding number from each sentence
  # and then remove the preceding space from each sentence
  sentences = [sentence.split(".")[1][1:] for sentence in sentences]

  # trim any leading or trailing quotes from each sentence
  sentences = [sentence.strip('"') for sentence in sentences]


  # print each sentence in a newline
  for sentence in sentences:
      print(sentence)

  # add the sentences to a new data frame under the column name "sentence"
  df = pd.DataFrame(sentences, columns=["text"])
  df['sentiment'] = sentiment
  df['sample'] = sample
  df['prompt'] = prompt
  df['expected-label'] = 0 if sentiment == 'positive' else 1
  df['generation_time'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

  return df



def generate_sentences_from_samples(sentiment, samples):
  global df_master

  # loop over samples and call generate_sentences. 
  for sample in samples:
    prompt = "Generate "+str(SAMPLE_COUNT)+" simple sentences similar to "+ sample +" with "+sentiment+" sentiment."
    df = generate_sentences(sentiment, sample, prompt)
    
    # append the new data frame to the master data frame
    df_master = df_master.append(df, ignore_index=True)

# run a loop for 10 times
for i in range(1):
  #continue
  generate_sentences_from_samples('positive', positive_samples)
  generate_sentences_from_samples('negative', negative_samples)


# append the report to the existing csv file. Append the header only once
with open(file_name, 'a') as f:
    # add a new line if the file already exists
    if os.stat(file_name).st_size != 0:
        f.write('\n')
    
    # add period to the end of each sentence which does not have a period or a question mark or an exclamation mark.
    df_master['text'] = df_master['text'].apply(lambda x: x if x.endswith('.') or x.endswith('?') or x.endswith('!') else x + '.')

    df_master.to_csv(f, header=f.tell() == 0, index=False)


# read the file using df and remove empty lines
df = pd.read_csv(file_name)
df = df[df['text'].notna()]

# save the file again.
df.to_csv(file_name, index=False)



# ## Test code
# sentiment = 'negative'
# sample = 'sorry to disagree.'
# prompt = "Generate 5 short sentences similar to "+ sample +" with "+sentiment+" sentiment."
# df = generate_sentences(sentiment, sample, prompt)
# df_master = df_master.append(df, ignore_index=True)

# save the master data frame to a csv file, append if the file already exists, otherwise create a new file with headers
# append the report to the existing csv file. Append the header only once. 
# ensure that the new data is written with a new line
















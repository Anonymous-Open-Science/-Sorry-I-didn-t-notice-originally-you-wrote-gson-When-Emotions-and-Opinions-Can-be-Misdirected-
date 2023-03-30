
import pandas as pd

import nltk
import os
import sys
import spacy

# file_dir = os.path.dirname(__file__)
# sys.path.append(file_dir)
#from config import *
#from opinion_extraction import extract_sentence_with_adjectives


file_path = ''
nlp = spacy.load("en_core_web_lg")

import warnings
warnings.filterwarnings("ignore")

import logging

LOG_LEVEL_APPLICATION_DEBUG = 15

# LOG_LEVEL = LOG_LEVEL_APPLICATION_DEBUG
LOG_LEVEL = logging.DEBUG


CACHED_DATA = True

DB_NAME = 'stackoverflow'
DB_USERID = ''
DB_PWD = ''

logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=LOG_LEVEL)


# read the file
df = pd.read_csv(file_path)
print(df.head(10))

# get the unique values in source column of df
df_source = df['source'].unique()


def extract_sentence_with_adjectives(sentence_list, check_word_list=True):
    # remove sentences that do not contain ajdective
    sentences = []
    adjectives = []
    for sent in sentence_list:
        doc = nlp(sent)
        adj_sentence = False
        for token in doc:
            if token.pos_ == 'ADJ':
                adjectives.append(token.text)
                adj_sentence = True

        if adj_sentence:
            sentences.append(sent)
    
    return sentences, adjectives

def get_top_words(words_list):
    df_words_list = pd.DataFrame(words_list, columns=['word'])
    df_words_list['count'] = 1
    df_words_list = df_words_list.groupby(['word']).count().reset_index()
    df_words_list = df_words_list.sort_values(by=['count'], ascending=False)
    return df_words_list

def get_top_words_from_sentence(sentence_list):
    # get the sentences. Stem and lemmatize the words.
    # then calculate the word frequency

    # remove non-alhpa characters
    sentence_list = [sentence.replace('[^a-zA-Z]', ' ') for sentence in sentence_list]

    # lower case
    sentence_list = [sentence.lower() for sentence in sentence_list]

    # # use TF-IDF to get the top words
    # from sklearn.feature_extraction.text import TfidfVectorizer
    # tfidf = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    # tfidf_matrix = tfidf.fit_transform(sentence_list)
    # feature_names = tfidf.get_feature_names()

    # # get the top words
    # from sklearn.cluster import KMeans
    # num_clusters = 5
    # km = KMeans(n_clusters=num_clusters)
    # km.fit(tfidf_matrix)
    # clusters = km.labels_.tolist()

    
    

    words_list = []
    for sentence in sentence_list:
        words = sentence.split()
        words_list.extend(words)

    # # lower case
    # words_list = [word.lower() for word in words_list]

    # remove all stop words using spacy
    stop_words = spacy.lang.en.stop_words.STOP_WORDS

    from nltk.corpus import stopwords
    stop_words_nltk = set(stopwords.words('english'))

    # add the stop words from nltk to spacy
    for word in stop_words_nltk:
        stop_words.add(word)

    custom_stop_words = ['the', 'this', 'there', 'you', 'use', 'try', 'code', 'java']

    for word in custom_stop_words:
        stop_words.add(word)



    # remove words less than 3 characters
    words_list = [word for word in words_list if len(word) > 2]

    # # remove words with numbers
    # words_list = [word for word in words_list if not any(c.isdigit() for c in word)]

    # remove words with numbers
    words_list = [word for word in words_list if not any(not c.isalpha() for c in word)]



    # remove words with special characters
    words_list = [word for word in words_list if not any(c in word for c in ['@', '#', '$', '%', '^', '&', '*', '(', ')', '-', '_', '=', '+', '[', ']', '{', '}', '|', '\\', ':', ';', '"', "'", '<', '>', ',', '.', '?', '/', '~', '`'])]


    # stem the words using spacy
    global nlp
    doc = nlp(' '.join(words_list))
    words_list = [token.lemma_ for token in doc]

    # remove words with only one character
    words_list = [word for word in words_list if len(word) > 2]
    words_list = [word for word in words_list if word not in stop_words]


    return get_top_words(words_list)

def get_top_adjectives(adjectives, source, type):
    df_adjectives = get_top_words(adjectives)
    df_adjectives['source'] = source
    df_adjectives['sentiment'] = type
    df_adjectives.to_csv(source + '.'+type+'.adjectives.csv', index=False)
    logging.info('Top 10 '+type+' adjectives:')
    logging.info("\n"+str(df_adjectives.head(10)))

    return df_adjectives

# for each of the unique values in source column of df
# get the text with positive and negative sentiment
# and save it to a file

# have a master dataframe for all the frequent words for each source
df_master = pd.DataFrame(columns=['source', 'sentiment', 'word', 'count'])
df_master_adj = pd.DataFrame(columns=['source', 'sentiment', 'word', 'count'])

for source in df_source:
    logging.info('Processing source: ' + source)
    df_source_file = df[df['source'] == source]

    # sentiment can be case insensitive
    df_source_file['sentiment'] = df_source_file['sentiment'].str.lower()
    df_source_file_positive = df_source_file[df_source_file['sentiment'] == 'positive']
    df_source_file_negative = df_source_file[df_source_file['sentiment'] == 'negative']

    sentences_pos_unused, adjectives_pos = extract_sentence_with_adjectives(df_source_file_positive['text'], False)
    sentences_neg_unused, adjectives_neg = extract_sentence_with_adjectives(df_source_file_negative['text'], False)


    # count the frequency of each adjective
    df_adjective_pos = get_top_adjectives(adjectives_pos, source, 'positive')
    df_adjective_neg = get_top_adjectives(adjectives_neg, source, 'negative')

    # get the top words from the sentences
    df_source_file_positive = get_top_words_from_sentence(df_source_file_positive['text'])
    df_source_file_negative = get_top_words_from_sentence(df_source_file_negative['text'])

    logging.info('Top 10 words from positive sentences:')
    print(df_source_file_positive.head(10))
    logging.info('Top 10 words from negative sentences:')
    print(df_source_file_negative.head(10))

    # add source column to the dataframes
    df_source_file_positive['source'] = source
    df_source_file_negative['source'] = source

    # add sentiment column to the dataframes
    df_source_file_positive['sentiment'] = 'positive'
    df_source_file_negative['sentiment'] = 'negative'

    # append the dataframes to the master dataframe
    df_master = df_master.append(df_source_file_positive)
    df_master = df_master.append(df_source_file_negative)

    df_master_adj = df_master_adj.append(df_adjective_pos)
    df_master_adj = df_master_adj.append(df_adjective_neg)

    
# save the dataframes to a file
# df_master.to_csv('word_frequency.csv', index=False)
df_master_adj.to_csv('word_frequency_adj.csv', index=False)

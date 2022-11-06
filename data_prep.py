#%%
import pickle
from transformers import pipeline
import pandas as pd
import re
import numpy as np
from plotnine import ggplot, aes, geom_line
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
# %%
# %% download and save pipeline and model
# nlp_model = pipeline(model='nlptown/bert-base-multilingual-uncased-sentiment')

#  save the specific model
# with open("models/sentiment_bert.pkl", "wb") as output_file:
#     pickle.dump(nlp_model, output_file)

#%% load model
with open("models/sentiment_bert.pkl", "rb") as input_file:
    nlp_model = pickle.load(input_file)


# %% Import Speeches
biden_speech_file = open('data/Biden_2021.txt', 'r')
trump_speech_file = open('data/Trump_2017.txt', 'r')
biden_speech_raw = biden_speech_file.read()
trump_speech_raw = trump_speech_file.read()

# %%
biden_speech_processed = biden_speech_raw.replace('\n', ' ').split('.')
biden_speech_processed = [i for i in biden_speech_processed if i != '']
# delete empty strings
trump_speech_processed = trump_speech_raw.replace('\n', ' ').split('.')
trump_speech_processed = [i for i in trump_speech_processed if i != '']
# %% run the NLP model
biden_sentiment = nlp_model(biden_speech_processed)
trump_sentiment = nlp_model(trump_speech_processed)

# %% convert results into dataframe
df_biden = pd.DataFrame(biden_sentiment)
df_biden['text'] = biden_speech_processed
df_biden['stars'] = [float(re.findall(r'\b\d+\b', i)[0]) for i in df_biden['label']]
df_trump = pd.DataFrame(trump_sentiment)
df_trump['text'] = trump_speech_processed
df_trump['stars'] = [float(re.findall(r'\b\d+\b', i)[0]) for i in df_trump['label']]

# add source
df_biden['source'] = 'Biden'
df_trump['source'] = 'Trump'


# %%
n_rows_biden = len(df_biden)
n_sectors = 360
df_biden['direction'] = np.round(df_biden.index / n_rows_biden * n_sectors, 0)
#%%
n_rows_trump = len(df_trump)
df_trump['direction'] = np.round(df_trump.index / n_rows_trump * n_sectors, 0)


df_combined = pd.concat([df_biden, df_trump])


#%% create function for data prep
def prepSpeech(file_path = 'data/Biden_2021.txt', president='Biden'):
    speech_file = open(file_path, 'r')
    speech_raw = speech_file.read()
    speech_processed = speech_raw.replace('\n', ' ').split('.')
    speech_processed = [i for i in speech_processed if i != '']
    list_sentiment = nlp_model(speech_processed)
    df_sentiment = pd.DataFrame(list_sentiment)
    df_sentiment['text'] = speech_processed
    df_sentiment['stars'] = [float(re.findall(r'\b\d+\b', i)[0]) for i in df_sentiment['label']]
    df_sentiment['source'] = president
    n_rows = len(df_sentiment)
    n_sectors = 360
    df_sentiment['direction'] = np.round(df_sentiment.index / n_rows * n_sectors, 0)
    return df_sentiment


# %% sentiment over speech
df_biden = prepSpeech(file_path='data/Biden_2021.txt', president='Biden')
df_trump = prepSpeech(file_path='data/Trump_2017.txt', president='Trump')
#%%
df_combined = pd.concat([df_biden, df_trump])
df_combined.to_csv("data/df_concat.csv", index=False)

# %%

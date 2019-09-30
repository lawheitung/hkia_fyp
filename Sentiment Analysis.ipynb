# #Sentiment Analysis
# In [1]:
import pandas as pd


# ### Load and Merge data
# In [2]:
hkia_df = pd.read_csv('./data/topic.csv')
hkia_replies_df = pd.read_csv('./data/reply.csv')
    # Re-Order columns * new changes*
cols = ['topic_id', 'heading', 'username', 'location', 'datetime', 'topic_desc']
hkia_df = hkia_df[cols]
hkia_df.set_index('topic_id', inplace=True)
hkia_df
hkia_replies_df.set_index('topic_id', inplace=True)
def filter_fn(row):
    if "Message from TripAdvisor staff" in row['reply_content']:
        return False
    else:
        return True
m = hkia_replies_df.apply(filter_fn, axis=1)
hkia_replies_df = hkia_replies_df[m]
topic_df = pd.concat(
    [hkia_df['topic_desc'], hkia_replies_df['reply_content']])
topic_df = topic_df.to_frame(name='topic_content')
topic_df['topic_content'] = topic_df['topic_content'].apply(
    lambda x: x.replace('\n', ' '))
def merge_text(x):
    return pd.Series(dict(topic_content='\n\n'.join(x['topic_content'])))
topic_df = topic_df.groupby('topic_id').apply(merge_text)


# In [3]:
# corpus = topic_df['topic_content'].str.cat(sep='\n\n')
corpus = topic_df['topic_content'].tolist()


# ### with StanfordCoreNLP
# In [4]:
from pycorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP('http://localhost:9000')
def strip_non_ascii(text):
    return ''.join([i if ord(i) < 128 else ' ' for i in text])
corpus = [strip_non_ascii(c) for c in corpus]

import json


# ### Results
# In [5]:
for c in range(len(corpus)):
    with open('results/result_{}.json'.format(c), 'w') as fp:
        result = nlp.annotate(corpus[c],
                              properties={
                                  'annotators': 'sentiment',
                                  'outputFormat': 'json',
        })
        json.dump(result, fp)
        print 'Done with topic: {}'.format(c)


# ### Read the results into a DataFrame
# In[6]:
sentiment_df = pd.DataFrame(
    columns=['sentence', 'sentimentValue', 'sentiment'])
for c in range(len(corpus)):
    with open('results/result_{}.json'.format(c), 'r') as json_file:
        result = json.load(json_file)
        for s in result["sentences"]:
            #             print("{}: '{}': {} (Sentiment Value) {} (Sentiment)".format(
            #                 s["index"],
            #                 " ".join([strip_non_ascii(t["word"]) for t in s["tokens"]]),
            #                 s["sentimentValue"], s["sentiment"]))
            sentiment_df = sentiment_df.append({
                'sentence': " ".join(
                    [strip_non_ascii(t["word"]) for t in s["tokens"]]),
                'sentimentValue': s["sentimentValue"],
                'sentiment': s["sentiment"]}, ignore_index=True)

# In[7]:
sentiment_df[m]






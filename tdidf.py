#!/usr/bin/env python
# coding: utf-8

# # HKIA Access Mode Choice Text Mining

# In[1]:


import nltk 
from nltk import FreqDist
nltk.download('stopwords') # run this one time
import pandas as pd 
pd.set_option("display.max_colwidth", 200) 
import numpy as np 
import re 
import spacy
import gensim 
from gensim import corpora 
# libraries for visualization 
import pyLDAvis 
import pyLDAvis.gensim 
import matplotlib.pyplot as plt 
import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')

from nltk.corpus import stopwords 
stop_words = stopwords.words('english')


# ## Load data

# In[2]:


# Trip Advisor topics
hkia_df = pd.read_csv('./data/topic.csv')
hkia_replies_df = pd.read_csv('./data/reply.csv')


# In[3]:


# Re-Order columns
cols = ['topic_id', 'heading', 'username', 'location', 'datetime', 'topic_desc']
hkia_df = hkia_df[cols]
hkia_df.set_index('topic_id', inplace=True)
hkia_df


# In[4]:


# Set topic id as index for replies too
hkia_replies_df.set_index('topic_id', inplace=True)


# In[5]:


def filter_fn(row):
    if "Message from TripAdvisor staff" in row['reply_content']:
        return False
    else:
        return True

m = hkia_replies_df.apply(filter_fn, axis=1)
hkia_replies_df = hkia_replies_df[m]


# In[6]:


# Concat both the replies and topics data
topic_df = pd.concat(
    [hkia_df['topic_desc'], hkia_replies_df['reply_content']])
topic_df = topic_df.to_frame(name='topic_content')


# In[7]:


# Group the data by topic_id and then merge text \n
def merge_text(x):
    return pd.Series(dict(topic_content='\n'.join(x['topic_content'])))


topic_df = topic_df.groupby('topic_id').apply(merge_text)
topic_df


# In[8]:


# remove unwanted characters, numbers and symbols 
topic_df['topic_content'] = topic_df['topic_content'].str.replace("[^a-zA-Z#]", " ")


# In[9]:


# function to plot most frequent terms
def freq_words(x, terms=30):
    all_words = ' '.join([text for text in x])
    all_words = all_words.split()

    fdist = FreqDist(all_words)
    words_df = pd.DataFrame({'word': list(fdist.keys()),
                             'count': list(fdist.values())})
    # selecting top 20 most frequent words
    d = words_df.nlargest(columns="count", n=terms)
    plt.figure(figsize=(20, 5))
    ax = sns.barplot(data=d, x="word", y="count")
    ax.set(ylabel='Count')
    plt.show()


# In[10]:


with pd.option_context('display.max_colwidth', -1):
    print topic_df['topic_content']


# In[11]:


freq_words(topic_df['topic_content'])


# ### Removing stop words

# In[12]:


stop_words = stopwords.words('english')
# function to remove stopwords


def remove_stopwords(p):
    p_new = " ".join([i for i in p if i not in stop_words])
    return p_new


# remove short words (length < 3)
topic_df['topic_content'] = topic_df['topic_content'].apply(lambda x: ' '.join([w for
                                                                                w in x.split() if len(w) > 2]))

# remove stopwords from the text
t_cont = [remove_stopwords(td.split()) for td in topic_df['topic_content']]

# make entire text lowercase
t_cont = [td.lower() for td in t_cont]


# In[13]:


freq_words(t_cont)


# ### Lemmatization

# In[14]:


get_ipython().system('python -m spacy download en #one time run')


# In[15]:


nlp = spacy.load('en', disable=['parser', 'ner'])


# In[16]:


def lemmatization(texts, tags=['NOUN', 'ADJ']):
    output = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        output.append([token.lemma_ for token in doc if
                       token.pos_ in tags])
    return output


# In[17]:


tokenized_tcont = pd.Series(t_cont).apply(lambda x: x.split())
tokenized_tcont = tokenized_tcont.apply(lambda x: [unicode(i) for i in x] )
print(tokenized_tcont[1])


# In[18]:


t_cont_lemmatized = lemmatization(tokenized_tcont)
print(t_cont_lemmatized[1]) # print lemmatized topic descs


# In[19]:


t_cont_lemmatized_2 = []
for i in range(len(t_cont_lemmatized)):
    t_cont_lemmatized_2.append(' '.join(t_cont_lemmatized[i]))

topic_df['topic_content'] = t_cont_lemmatized_2

freq_words(topic_df['topic_content'], 35)


# ## TF-IDF Analysis

# In[20]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[21]:


tfidf = TfidfVectorizer()


# In[22]:


topic_cont_list = topic_df['topic_content'].tolist()
response = tfidf.fit_transform(topic_cont_list)


# In[ ]:


topics_tfidf = pd.DataFrame(response.toarray())
topics_tfidf.columns = tfidf.get_feature_names()


# In[45]:


tfidf_across_all_samples_mean = topics_tfidf.mean(axis=0)


# In[48]:


with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print tfidf_across_all_samples_mean.sort_values(ascending=False)


# In[ ]:





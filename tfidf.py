# HKIA Access Mode Choice Text Mining

# In[1]:


import nltk 
from nltk import FreqDist
nltk.download('stopwords')
import pandas as pd 
pd.set_option("display.max_colwidth", 200) 
import numpy as np 
import re 
import spacy
import gensim 
from gensim import corpora 
# libraries for visualization (not working very well)
# import pyLDAvis 
# import pyLDAvis.gensim 
# import matplotlib.pyplot as plt 
# import seaborn as sns 
# get_ipython().run_line_magic('matplotlib', 'inline')

from nltk.corpus import stopwords 
stop_words = stopwords.words('english')


#Load data

# [2]:
# Trip Advisor topics
hkia_df = pd.read_csv('./data/topic.csv')
hkia_replies_df = pd.read_csv('./data/reply.csv')


# [3]:
# Re-Order columns
cols = ['topic_id', 'heading', 'username', 'location', 'datetime', 'topic_desc']
hkia_df = hkia_df[cols]
hkia_df.set_index('topic_id', inplace=True)
hkia_df


# [4]:
# Set topic id as index for replies
hkia_replies_df.set_index('topic_id', inplace=True)


# [5]:
def filter_fn(row):
    if "Message from TripAdvisor staff" in row['reply_content']:
        return False
    else:
        return True

m = hkia_replies_df.apply(filter_fn, axis=1)
hkia_replies_df[m]


# [6]:
# Concat both the replies and topics data * new changes*
topic_df = pd.concat(
    [hkia_df['topic_desc'], hkia_replies_df['reply_content']])
topic_df = topic_df.to_frame(name='topic_content')


# [7]:
# Group the data by topic_id and then merge text * new changes *
def merge_text(x):
    return pd.Series(dict(topic_content='\n'.join(x['topic_content'])))

topic_df = topic_df.groupby('topic_id').apply(merge_text)
topic_df


# [8]:
topic_df['topic_content'] = topic_df['topic_content'].str.replace("[^a-zA-Z#]", " ")


# [9]:
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


# In[12]:


stop_words = stopwords.words('english')
# function to remove stopwords
def remove_stopwords(p):
    p_new = " ".join([i for i in p if i not in stop_words])
    return p_new


# make entire text lowercase
t_cont = [td.lower() for td in t_cont]

# remove short words (length < 3)
topic_df['topic_content'] = topic_df['topic_content'].apply(lambda x: ' '.join([w for
                                                                                w in x.split() if len(w) > 2]))
t_cont = [remove_stopwords(td.split()) for td in topic_df['topic_content']]



# In[13]:


freq_words(t_cont)


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


# In[20]:


t_cont_lemmatized_2 = []
for i in range(len(t_cont_lemmatized)):
    t_cont_lemmatized_2.append(' '.join(t_cont_lemmatized[i]))

topic_df['topic_content'] = t_cont_lemmatized_2

freq_words(topic_df['topic_content'], 35)







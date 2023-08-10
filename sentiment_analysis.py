#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm


# In[3]:


# Read in data
df = pd.read_csv('Reviews.csv')


# In[4]:


# Plot count of reviews by stars
ax = df['Score'].value_counts().sort_index()     .plot(kind='bar',
          title='Count of Reviews by Stars',
          figsize=(10, 5))
ax.set_xlabel('Review Stars')
plt.show()


# In[5]:


# Perform sentiment analysis on a single example review
example = df['Text'][50]
print(example)


# In[6]:


tokens = nltk.word_tokenize(example)
tokens[:10]


# In[7]:


tagged = nltk.pos_tag(tokens)
tagged[:10]


# In[8]:


entities = nltk.chunk.ne_chunk(tagged)
entities.pprint()


# In[9]:


sia = SentimentIntensityAnalyzer()


# In[10]:


sia.polarity_scores('I am so happy!')


# In[11]:


sia.polarity_scores('This is the worst thing ever.')


# In[12]:


sia.polarity_scores(example)


# In[13]:


# Run the polarity score on the entire dataset
res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row['Text']
    myid = row['Id']
    res[myid] = sia.polarity_scores(text)
    
vaders = pd.DataFrame(res).T
vaders = vaders.reset_index().rename(columns={'index': 'Id'})
vaders = vaders.merge(df, how='left')


# In[14]:


# Plot compound score by Amazon star review
ax = sns.barplot(data=vaders, x='Score', y='compound', color='blue')
ax.set_title('Compound Score by Amazon Star Review')
plt.show()

# Plot positive, neutral, and negative scores by Amazon star review
fig, axs = plt.subplots(1, 3, figsize=(12, 3))
sns.barplot(data=vaders, x='Score', y='pos', ax=axs[0], color='green')
sns.barplot(data=vaders, x='Score', y='neu', ax=axs[1], color='gray')
sns.barplot(data=vaders, x='Score', y='neg', ax=axs[2], color='red')
axs[0].set_title('Positive')
axs[1].set_title('Neutral')
axs[2].set_title('Negative')
plt.tight_layout()
plt.show()


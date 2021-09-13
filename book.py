#!/usr/bin/env python
# coding: utf-8

# # Reading the data

# In[1]:


from nltk.tokenize import word_tokenize
raw_text = open('report.txt').read()


# In[2]:


print(raw_text[0:1000])


# # Cleaning Data

# In[5]:


import re
text = re.sub("[^a-zA-Z]"," ",str(raw_text)) #removing puncutuations
text1 = text.lower()                         #lower case letters 
text2 = " ".join(text1.split())              #removing white-space 
print(text2[0:1000])


# # TOKENIZING

# In[7]:


text3 = re.sub("[^\w]", " ",  text2).split()
print(text3[0:100])


# # REMOVING STPOWORDS

# In[9]:


import nltk
import string
from nltk.corpus import stopwords
stopwords = nltk.corpus.stopwords.words('english')
words_new = [i for i in text3 if i not in stopwords]
print(words_new[0:100])


# # Lemmatization & Stemming

# In[11]:


from nltk.stem import PorterStemmer
steam = nltk.PorterStemmer()
text_words = [steam.stem(word) for word in words_new]
print(text_words[0:500])


# # POSITIVE AND NEGATIVE WORDS

# In[12]:


from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
def sentiment_analyzer(sentiment_text):
    score = SentimentIntensityAnalyzer().polarity_scores(sentiment_text)
    print(score)
    negative = score['neg']
    positive = score['pos']
    
    if negative > positive: 
        print('Neagtive Sentiment')
    else:
        print('Positive sentiment')
    get_ipython().run_line_magic('matplotlib', 'inline')
    data = {'neg': 0.034,'neu': 0.803, 'pos': 0.164, 'compound': 1.0}
    names = list(data.keys())
    values = list(data.values())
    labels = ['negative', 'neutral', 'positive', 'compound']
    plt.bar(range(len(data)), values, tick_label = labels)
    plt.show()
sentiment_analysis = sentiment_analyzer(text2)


# # Average Sentence Length

# In[13]:


def Avg_length(text):
    sentence = text.split(".")
    words = text.split(" ")
    
    if(sentence[len(sentence) - 1] == ""):
        avg_sentence_length = len(words) / len(sentence) - 1
    else:
        avg_sentence_length = len(words) / len(sentence)
    return avg_sentence_length
average = Avg_length(text2)
print(average)


# # Total Word Count

# In[ ]:


import re
def word_count(text):
    frequency = {}
    pattern = re.findall(r'\b[a-z]{2,15}\b', text)
    for word in pattern:
        count = frequency.get(word, 0)
        frequency[word] = count + 1
        frequency_list = frequency.keys()
        for words in frequency_list:
            print(words, frequency[words])
            
    return text
total_count = word_count(text2)
print(total_count)


# In[ ]:





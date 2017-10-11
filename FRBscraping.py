
# coding: utf-8

# In[39]:

import urllib.request
from bs4 import BeautifulSoup as bs 
import numpy as np
import nltk
from nltk import word_tokenize
from nltk.collections import *


# In[50]:

def get_paragraphs(url):
    res = urllib.request.urlopen(url)
    res = res.read()
    soup = bs(res, "html.parser")
    statement = soup.find('div', attrs={'id': 'article'})
    paragraphs =statement.findAll('p')[1:-1]
    return paragraphs    


# In[51]:

url = "https://www.federalreserve.gov/newsevents/pressreleases/monetary20110126a.htm"


# In[53]:

paragraphs = get_paragraphs(url)
print (paragraphs[1].text.strip())


# In[59]:

speech=[]
for paragraph in paragraphs:
    tokens = word_tokenize(paragraph.text.strip())
    speech.append(tokens)


# In[67]:

speech = np.array(speech)


# In[68]:

speech = np.concatenate(speech, axis = 0)
text = nltk.Text(speech)
text.collocations()


# In[ ]:




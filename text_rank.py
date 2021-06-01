#-*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import nltk
nltk.download('punkt') # one time execution
import re

print("\n\nIf you don't have nltk, install it with pip")
print("type in 'pip install nltk'")


df = pd.read_csv("tennis_articles_v1.csv")


print("\n\nThe columns of tennis_articles_v1.csv")
print(df.columns)

print("\n\nThe data structure of tennis_article_v1.csv")
print(df.head())

print("\n\nArticle Sample from tennis_articles_v1.csv")
print(df['article_text'][0])

from nltk.tokenize import sent_tokenize

sentences = []
for s in df['article_text']:
  sentences.append(sent_tokenize(s))

sentences = [y for x in sentences for y in x] # flatten list


print("\n\nTop 5 sample sentences split from article of tennis_articles_v1.csv")
print(sentences[:5])

# remove punctuations, numbers and special characters
clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")

# make alphabets lowercase
clean_sentences = [s.lower() for s in clean_sentences]


print("\n\nAfter clearing Sample sentences split from article of tennis_articles_v1.csv")
print(clean_sentences[:5])



print("\n\nDonwloading stopwords using ntlk...")
nltk.download('stopwords')

from nltk.corpus import stopwords
stop_words = stopwords.words('english')

print("\n\nEnglish stopwords..")
print(stop_words)


# function to remove stopwords
def remove_stopwords(sen):
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new


# remove stopwords from the sentences
print("\n\nStarting removal of stopwords on clean sentences")
clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]

print("\n\nAfter removing stopwords, top 5 sample sentences")
print(clean_sentences[:5])

print("\n\nIf you don't have vector files such as glove, type in as follows:")
print("wget http://nlp.stanford.edu/data/glove.6B.zip")
print("unzip glove*.zip")

## Extract word vectors
word_embeddings = {}
f = open('glove.6B.100d.txt', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    word_embeddings[word] = coefs
f.close()


print("The number of word embedding of glove 100-d: {}".format(len(word_embeddings)))


print("\n\nmaking sentence vector using glove-100d")

sentence_vectors = []
for i in clean_sentences:
  if len(i) != 0:
    v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
  else:
    v = np.zeros((100,))
  sentence_vectors.append(v)



print("\n\nMake similarity matrix with sentence vector")
#similarity matrix
sim_mat = np.zeros([len(sentences), len(sentences)])

from sklearn.metrics.pairwise import cosine_similarity

for i in range(len(sentences)):
  for j in range(len(sentences)):
    if i != j:
      sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]


print("\n\ncoverting sentence's similarity matrix to a graph..")
import networkx as nx

nx_graph = nx.from_numpy_array(sim_mat)
scores = nx.pagerank(nx_graph)

print("\n\nIf you don't have networkx, install it with pip")
print("type in 'pip install networkx'")

ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)

print("\n\nExtrainct top 10 sentences as the summary")
# Extract top 10 sentences as the summary
for i in range(10):
  print(ranked_sentences[i][1])




from collections import Counter
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
import hdbscan
import numpy as np


from topic_clustering_comparative import preprocess

#load data into a dataframe
raw_df = pd.read_csv('reddit_posts_2025-03-02_204135.csv')
raw_str = ' '.join(raw_df['text'].astype(str))

#clean data & preprocess by tokenizing input into chunks of information
corpus = sent_tokenize(raw_str)
preprocessed_data = [preprocess(sentence) for sentence in corpus]
tokens_str = ' '.join(word for doc in preprocessed_data for word in doc)

#tokens_str = ' '.join(preprocessed_data)


def wordfreq_analysis(tokens_str):
  """
  Generates frequencies of each word from the cleaned array and plots a
  word cloud. Most frequent words are shown as bigger with lower frequencies
  shown smaller. With higher frequency common words that are unecessary not
  part of the word cloud. Plots a wordcloud.

  Parameters:
  -----------
  cleaned_text: tokenized and pre-processed reddit posts in a string format
  """
  wordcloud = WordCloud(width=800, height=400, background_color='white').generate(tokens_str)
  plt.figure(figsize=(10,5))
  plt.imshow(wordcloud, interpolation='bilinear')
  plt.axis('off')
  plt.show()

#wordfreq_analysis(tokens_str)

def calculate_tfdif(corpus):
  """
  Calculates the TF-IDF value per word in the corpus and seperates similar
  words into clusters through KMeans Clustering. Outputs clustered posts to
  sentence_clusters.csv

  Parameters:
  -----------
  corpus: reddit posts in text format seperated into documents
  """
  #Calculating TF-IDF
  vectorizer = TfidfVectorizer()
  X = vectorizer.fit_transform(corpus)

  #Cluster seperation using KMeans
  kmeans = KMeans(n_clusters=5, random_state=0)
  kmeans.fit(X)
  labels = kmeans.labels_

  clustered = pd.DataFrame({'Text': corpus, 'Cluster': labels})
  clustered.to_csv("sentence_clusters.csv", index=False)

#calculate_tfdif(raw_str)

def Word2Vec_HDBScan(tokens, corpus):
  """
  Trains a Word2Vec model on the cleaned, tokenized Reddit posts and groups
  similar posts together using HDBSCAN. Outputs clustered posts to
  a hdb_clusters.csv.

  Parameters:
  -----------
  tokens[]: pre-processed reddit post data as tokenized chunks of information
  corpus[]: reddit posts in text format seperated into documents
  """
  model = Word2Vec(sentences=tokens, vector_size=50, window=5, min_count=1, workers=4)

  def get_vector(doc):
      vecs = [model.wv[word] for word in doc if word in model.wv]
      return np.mean(vecs, axis=0) if vecs else np.zeros(model.vector_size)

  doc_vectors = np.array([get_vector(doc) for doc in tokens])

  clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
  labels = clusterer.fit_predict(doc_vectors)

  df = pd.DataFrame({'Text': corpus, 'Cluster': labels})
  df.to_csv("hdb_clusters.csv", index=False)

Word2Vec_HDBScan(preprocessed_data, corpus)

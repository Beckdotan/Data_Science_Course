#!/usr/bin/env python
# coding: utf-8

# In the forthcoming sessions, you will discover how to solve the problem of CLUSTERING for data MARKETING with Kmeans algorithm 

# # 1 enviroment & data

# In the forthcoming sessions, you will discover how to solve the problem of CLUSTERING for data MARKETING with Kmeans algorithm 

# In[1]:


import scipy
import numpy as np
import matplotlib
import pandas as pd


# In[2]:


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


# Data preperation and Initialization

# In[5]:


input_file="marketing.csv"


# In[6]:


def get_df(file):
    dataset = pd.read_csv(file)
    df = pd.DataFrame(dataset)
    df=df.fillna(0)
    return df
df=get_df(input_file)


# In[7]:


def plot_properties(dataframe):
    print('head\n',dataframe.head(5))
    print('columns\n',dataframe.columns)
    print('shape\n',dataframe.shape)
plot_properties(df)


# In[8]:


distance_metric='euclidean'
K=5
k_range = np.arange(2, 11, 1)


# Build model

# In[9]:


def get_clusterer(points,k):
    clusterer = KMeans (n_clusters=k)
    preds = clusterer.fit_predict(points)
    return clusterer,preds
clusterer,preds=get_clusterer(df,K)


# In[5]:


def show_results (clusterer):
    print('labels:', clusterer.labels_, '\n') 
    centers = clusterer.cluster_centers_
    print('centroids:')
    for i in range(K):
        print(i,':',centers[i,:],'\n')
    for i in range(K):
        plt.plot(centers[i,:])
show_results(clusterer)


# Unit 4 :  Silheuette

# In[10]:


def print_silheuette (df,preds):
    sil=silhouette_score (df, preds, metric=distance_metric)
    print('silheuette:', sil, '\n')
print_silheuette(df,preds)


# In[13]:


def get_silheuettes(df,preds):
    Silhouettes = []
    for k in k_range:
        clusterer,preds=get_clusterer(df,k)
        Silhouettes.append(silhouette_score (df, preds, metric='euclidean'))
    return Silhouettes
Silhouettes = get_silheuettes(df,preds)


# In[14]:


def show_silhouettes (Silhouettes):
    plt.plot(k_range, Silhouettes, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Silheuettes')
    plt.title('Sillheuettes Vs, k')
    plt.show()
show_silhouettes(Silhouettes)


#!/usr/bin/env python
# coding: utf-8

# # 1 enviroment & data

# In this session, you will discover how to solve the problem of PATTERN RECOGNITION for data ABSENTEEISM_AT_WORK with STATISTICS &VISULAIZATION techniques 

# Import libraries

# In[1]:


import scipy
import numpy as np
import matplotlib
import pandas as pd


# In[2]:


import os
import math
import seaborn as sns #statistical data visualization
import matplotlib.pyplot as plt #matplotlib is a library, pyplot is a module
import mpl_toolkits #mpl_toolkits is a package
from mpl_toolkits.mplot3d import axes3d #axe3d is a module


# The data are composed ofthe followin attributes:
# 
# 1. Individual identification (ID)
# 2. Reason for absence ( by the categories of the International Code of Diseases (ICD))
# 3. Other (non ICD) categories (e.g. Month of absence, and Age)
# For further details see "Attribute absenteeism.doc"

# Data

# In[3]:


input_file = "absenteeism.csv"
pd.read_csv(input_file)


# In[4]:


def get_df(file):
    dataset = pd.read_csv(file)
    df = pd.DataFrame(dataset)
    df=df.fillna(0)
    return df
df=get_df(input_file)


# In[132]:


def plot_properties(dataframe):
    print('head\n',dataframe.head(5))
    print('columns\n',dataframe.columns)
    print('shape\n',dataframe.shape)
plot_properties(df)


# # 2 basic statistics (How spread out are the data?)

# Box and Whisker
# The box extends from the Q1 to Q3 quartile values of the data, with a line at the median (Q2). The whiskers extend from the edges of box to show the range of the data. By default, they extend no more than 1.5 * IQR (IQR = Q3 - Q1) from the edges of the box, ending at the farthest data point within that interval. Outliers are plotted as separate dots.

# In[133]:


#df.isna().sum() 
#df_dn = df.dropna(subset=['','']) if there had been attribute with Nan and we wished to drop them


# In[134]:


def plot_statistics(dataframe,selected_columns):
    print('7 numbers statistics\n',df.describe())
    print('How many Nans?\n', dataframe.isna().sum() )
    #print('Box and Whisker\n',dataframe.boxplot())
    print('Box and Whisker for selected columns\n',dataframe.boxplot(column=selected_columns) ) 
plot_statistics(df,['Transportation expense', 'Distance from Residence to Work', 'Age', 'Body mass index', 'Absenteeism time in hours'])


# # 3 histograms ( Are the data skewed?)

# In[135]:


def plot_histogram(dataframe,col):
        n_unique_categories = len(pd.unique(dataframe[col]))
        plt.hist(dataframe[col], bins=n_unique_categories)
        plt.xlabel("bins")
        plt.ylabel("counts")
        plt.title(col)
plot_histogram(df,'Month of absence')


# In[136]:


plot_histogram(df,'Reason for absence')


# In[137]:


def plot_sorted_histogram(dataframe,col):
    n_unique_categories = len(pd.unique(df[col]))
    category_hist = plt.hist( df[col], bins=n_unique_categories)
    category_hist_counts= category_hist[0] #category_hist_bins = category_hist[1]
    df1 = pd.DataFrame({'categories': pd.unique(df[col]),
                   'number': category_hist_counts})
    df1.set_index('categories', inplace=True)
    df1.sort_values('number', inplace=True)
    df1.plot(y='number', kind='bar', legend=False)
    plt.xlabel("bins")
    plt.ylabel("counts")
    plt.title(col)
plot_sorted_histogram(df,'Distance from Residence to Work')


# In[138]:


def plot_mean_groupedbycol (dataframe,col,grouping_col):
    grouping_set = set(df[grouping_col])
    grouping_list = list(grouping_set)
    mean_grouped = df.groupby(grouping_col)[col].mean()
    mean_list=mean_grouped.tolist()
    groupmean_df = pd.DataFrame({grouping_col:grouping_list,col:mean_list})
    print(groupmean_df)
    ax = groupmean_df.plot.bar(x=grouping_col,y=col, figsize=(8,6))
plot_mean_groupedbycol(df,"Absenteeism time in hours","Seasons")


# In[139]:


plot_mean_groupedbycol(df,"Absenteeism time in hours","Reason for absence")


# In[140]:


def plot_distribution(dataframe,col):
    df.plot.kde(y=col,figsize =(8,6))
plot_distribution(df,'Weight')


# In[141]:


def plot_violin(dataframe,colx,coly,colsplit):
    fig, ax = plt.subplots() 
    ax.set_title('VIOLIN SPLIT')
    sns.violinplot(data=dataframe,x=colx,y=coly,hue=colsplit, split='True')
plot_violin(df,'Day of the week','Absenteeism time in hours','Social smoker')


# # 4 line charts (Are there any outliers? any shifts in values? in variation?)  

# In[142]:


def plot_lines(dataframe,colx1,cost1):
    x_vals = dataframe[colx1]
    y1_vals = [cost1*i for i in x_vals]  
    
    plt.rcParams["figure.figsize"] = [12,8] #rcParams is a function, figure is a group
    plt.subplot(1,2,1)
    plt.plot(x_vals,y1_vals,'r*-', label ="linear") #label is a parameter,Line2D property
    plt.subplot(1,2,2)
   
plot_lines(df,'Absenteeism time in hours',5)


# # 5 scatterplots (Are there any outliers? Bivariate relations?)

# In[143]:


def plot_scatter(dataframe,col1,col2):
        plt.rcParams["figure.figsize"] = [8,6] #rcParams is a function, figure is a group
        plt.xlabel(col1)
        plt.ylabel(col2)
        plt.scatter(dataframe[col1], dataframe[col2],c="b", marker ="x")
plot_scatter(df,'Age', 'Absenteeism time in hours')


# In[144]:


def plot_scatter3d(dataframe,col1,col2,col3):
    figure1= plt.figure()
    axis1=figure1.add_subplot(projection='3d')
    axis1.scatter(df[col1], df[col2], df[col3])
plot_scatter3d(df,'Day of the week','Month of absence','Absenteeism time in hours')


# In[145]:


def plot_scatter3d(dataframe,col1,col2,col3):
    figure1= plt.figure()
    axis1=figure1.add_subplot(projection='3d')
    axis1.scatter(df[col1], df[col2], df[col3])
plot_scatter3d(df,'Age','Weight','Absenteeism time in hours')


# # 6 simple linear model

# In[146]:


def plot_lm(df,colx,coly,colsplit):
    ax2=sns.lmplot(data=df,x='Service time',y='Age') #lmplot linear model fit
    ax2=sns.lmplot(data=df,x=colx,y=coly, hue=colsplit) #lmplot linear model fit
plot_lm(df,'Service time','Age','Social drinker')


# # 7 correlations

# In[147]:


def plot_heatmap(dataframe,cols):
    plt.rcParams["figure.figsize"]=[10,8]
    sub_items = df[cols]
    sub_items_corr = sub_items.corr()
    ax=sns.heatmap(sub_items_corr,annot=True)
plot_heatmap(df,['Absenteeism time in hours', 'Body mass index','Weight','Distance from Residence to Work','Service time','Age','Social smoker'])


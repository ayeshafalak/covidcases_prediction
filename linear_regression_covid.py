#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor


# In[3]:


df=pd.read_csv('state_wise_daily.csv')


# In[4]:


df.info()


# In[5]:


df.head()


# In[6]:


df.describe()


# In[8]:


df.plot.scatter(x='Date_YMD',y='TT')


# In[9]:


df['Confirmed']=df['TT']


# In[10]:


df['Confirmed']


# In[11]:


df.sum(axis=0)


# In[12]:


df.Confirmed.mean()


# In[13]:


df=df[df.Status=='Confirmed']


# In[14]:


df=df.drop(columns='Status', axis=1)


# In[15]:


columns=['Date','Date_YMD', 'Confirmed']


# In[16]:


df=df[columns]


# In[17]:


df


# In[18]:


df.groupby('Date_YMD').mean()


# In[19]:


import datetime as dt
df['Date']=pd.to_datetime(df['Date'])
df['Date']=df['Date'].map(dt.datetime.toordinal)


# In[20]:


x=df['Date']
y=df['Confirmed']


# In[21]:


xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2, random_state=0)


# In[22]:


model=LinearRegression()


# In[23]:


model.fit(np.array(xtrain).reshape(-1,1), np.array(ytrain).reshape(-1,1))


# 

# In[24]:


df.tail()


# In[25]:


ytrain.head()


# In[26]:


prediction=model.predict(np.array(xtest).reshape(-1,1))


# In[27]:


model.predict(np.array([[737933]]))


# In[28]:


from sklearn.metrics import mean_squared_error
import math
rmse=mean_squared_error(ytest,prediction)
rmse


# In[29]:


model.score(np.array(x).reshape(-1,1), np.array(y).reshape(-1,1))*100


# In[ ]:





# In[30]:


df.plot.scatter(x='Date',y='Confirmed')


# In[ ]:





# In[ ]:





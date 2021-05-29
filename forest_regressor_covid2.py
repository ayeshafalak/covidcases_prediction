#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime,date,time
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor


# In[35]:


st.write("Covid-19 Cases Prediction")
predictdate= st.date_input('Input Date')
predictdate=pd.to_datetime(predictdate)
predictdate=predictdate.toordinal()


# In[5]:


df=pd.read_csv('state_wise_daily.csv')


# In[6]:


df.info()


# In[7]:


df.head()


# In[8]:


df.describe()


# In[ ]:





# In[9]:


df['Confirmed']=df.sum(axis=1)


# In[ ]:





# In[10]:


df.sum(axis=0)


# In[11]:


df.Confirmed.mean()


# In[12]:


df=df[df.Status=='Confirmed']


# In[13]:


df=df.drop(columns='Status', axis=1)


# In[14]:


columns=['Date','Date_YMD', 'Confirmed']


# In[15]:


df=df[columns]


# In[16]:


df.head()


# In[17]:


df.groupby('Date_YMD').mean()


# In[18]:



df['Date']=pd.to_datetime(df['Date'])
df['Date']=df['Date'].map(dt.datetime.toordinal)


# In[19]:


x=df['Date']
y=df['Confirmed']


# In[20]:


xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2, random_state=0)


# In[21]:


model=RandomForestRegressor()
#model=LinearRegression()


# In[22]:


model.fit(np.array(xtrain).reshape(-1,1), np.array(ytrain).reshape(-1,1))


# 

# In[23]:


df.tail()


# In[24]:


ytrain.head()


# In[25]:


prediction=model.predict(np.array(xtest).reshape(-1,1))


# In[26]:


model.predict(np.array([[737937]]))


# In[27]:


from sklearn.metrics import mean_squared_error
import math
rmse=mean_squared_error(ytest,prediction)


# In[28]:


score=model.score(np.array(x).reshape(-1,1), np.array(y).reshape(-1,1))*100


# In[36]:



st.write("Predicted cases")
result= model.predict(np.array([[predictdate]]))
r=result[0].astype(int)
r


# In[37]:


df.plot.scatter(x='Date',y='Confirmed')
st.line_chart(df['Confirmed'])


# In[31]:
st.write("Made by Ayesha Falak")




# In[ ]:





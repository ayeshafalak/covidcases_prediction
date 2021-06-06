#!/usr/bin/env python
# coding: utf-8

# Importing Modules:


import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime,date,time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


# Taking Input Date From User:


st.write("Covid-19 Cases Prediction")
predictdate= st.date_input('Input Date')
predictdate=pd.to_datetime(predictdate)
predictdate=predictdate.toordinal()

# Reading Dataset:


df=pd.read_csv('covid-data.csv')


# EDA:


df.info()
df.tail()
df.describe()


# Preprocessing:

df['Confirmed']=df['TT']

df.reset_index(drop=True, inplace=True)

df.sum(axis=0)

df.Confirmed.mean()

df=df[df.Status=='Confirmed']

df=df.drop(columns='Status', axis=1)

columns=['Date', 'Confirmed']

df=df[columns]

df.head()

df.groupby('Date').mean()

df['Date']=pd.to_datetime(df['Date'])
df['Date']=df['Date'].map(dt.datetime.toordinal)


# Model :


x=df['Date']
y=df['Confirmed']
x.reset_index(drop=True, inplace=True)
y.reset_index(drop=True, inplace=True)

l=len(x)
x[l]=predictdate
x=np.array(x).reshape(-1,1)
y=np.array(y).reshape(-1,1)



# Scaling:


sc=StandardScaler()
fX=sc.fit_transform(x)
fY=sc.fit_transform(y)

# training:

ans=fX[len(fX)-1][0]
fX=fX[:len(fX)-1]

model=SVR(kernel='rbf')
model.fit(fX,fY.ravel())

# Prediction:


pred=model.predict(fX)
pred=sc.inverse_transform(pred)
pred=np.array(pred).reshape(-1,1)

# Accuracy:

score=model.score(fX,fY)*100

#Predicting new data


ans=model.predict(np.array(ans).reshape(-1,1))
ans=sc.inverse_transform(ans)
st.write("Predicted cases ")
result=int(ans[0])
result

# Data Visualization:
fdf=pd.DataFrame(pred)
fdf.plot()

df.plot.scatter(x='Date',y='Confirmed')

st.write("Confirmed Cases")
st.line_chart(y)
st.write("Prediction Graph")
st.line_chart(fdf)


st.write("Made by Ayesha Falak using Supervised Machine Learning ")
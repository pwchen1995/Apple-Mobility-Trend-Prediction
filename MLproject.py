#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 18:03:50 2020

@author: kevinchen
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from pandasql import sqldf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
psql = lambda q: sqldf(q, globals())

import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error
    
def read_file():
    
    applePath = "/Users/kevinchen/Desktop/WSU_Fall2020/Machine Learning/ML-Project/applemobilitytrends-2020-10-18.csv"
    apple = pd.read_csv(applePath, header=0)
    
    return apple

def createFeatures(mobilityData, label=None):
    #print(mobilityData['date'].dt)
    mobilityData['day_of_week']= mobilityData.date.dt.dayofweek
    mobilityData['month'] = mobilityData['date'].dt.month
    mobilityData['day_of_year'] = mobilityData['date'].dt.dayofyear
    mobilityData['day_of_month'] = mobilityData['date'].dt.day
    mobilityData['week_of_year'] = mobilityData['date'].dt.isocalendar().week.astype(int) 
    mobilityData= mobilityData[['country', 'date', 'month', 'day_of_week', 'day_of_year', 'day_of_month','week_of_year' ,'mobility']]
    
    return mobilityData

def visualize_Mobility_data(mobilityData):
    
    color_pal = ["#F8766D", "#D39200", "#93AA00", "#00BA38", "#00C19F", "#00B9E3", "#619CFF", "#DB72FB"]
    _ = mobilityData['mobility'].plot(style='-', figsize=(15,5), color=color_pal[4], title='mobility')

def Split_data(mobilityData):
    
    X=  mobilityData.copy()
    y= X['mobility'].copy()
    X= X.drop(columns=['mobility'])
    X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.2, random_state= 42)
    
    return X_train, X_test, y_train, y_test

def monthWise_predictedResult(X_test):
    
    plotData= X_test.loc[(X_test['month'] == 7) & (X_test['Country_United States'] == 1)]
    f, ax = plt.subplots(1)
    f.set_figheight(5)
    f.set_figwidth(15)
    
    # Correlated degree between true value and predicted
    _ = plotData[['TrueValue','PredValue']].plot(ax=ax, style=['-','-'])
def weekWise_predictedResult(X_test):
    plotData= X_test.loc[X_test['day_of_week'] == 5]
    f, ax = plt.subplots(1)
    f.set_figheight(5)
    f.set_figwidth(15)
    
    _ = plotData[['TrueValue','PredValue']].plot(ax=ax, style=['-','-'])

if __name__ == "__main__":
   
    color_pal = ["#F8766D", "#D39200", "#93AA00", "#00BA38", "#00C19F", "#00B9E3", "#619CFF", "#DB72FB"]
    
    apple = read_file()
    countryList= ['Taiwan','India','Singapore','Russia','Saudi Arabia','Philippines','Norway','Mexico','Malaysia','Japan','Argentina','Austria','Canada','Brazil','Germany','France','Hong Kong','United States', 'Italy','United Kingdom', 'Spain']
    
    #Reorganize data
    mobilityData= apple.loc[apple.country.isin(countryList),:].sort_values(by=['country'])
    mobilityData= mobilityData.drop(['geo_type', 'region', 'sub-region'], axis=1)
    mobilityData= mobilityData.fillna(100.00)
    mobilityData= mobilityData.groupby(['country']).median().reset_index()
    mobilityData= mobilityData.melt(id_vars=['country'], value_vars= mobilityData.columns.values[1:], var_name= 'date', value_name= 'mobility')
    mobilityData.reset_index(inplace=True)
    mobilityData= mobilityData.drop(columns=['index'])
    mobilityData.date= pd.to_datetime(mobilityData.date, format= '%d-%m-%Y')
    mobilityData= mobilityData.sort_values(by=['date','country']).reset_index(drop=True)
    
    #Reorganzie data with specific features
    mobilityData= createFeatures(mobilityData)
    
    #Visualize mobility data
    visualize_Mobility_data(mobilityData)
    
    #Train test split
    X_train, X_test, y_train, y_test = Split_data(mobilityData)
    #_= y_train.plot(style='-', figsize=(15,5), color=color_pal[4], title='mobility')
    _1= y_test.plot(style='-', figsize=(15,5), color=color_pal[4], title='mobility')
    
    #encoding training countries
    encodedCountries= pd.get_dummies(X_train.country, prefix='Country')
    X_train= pd.concat([encodedCountries, X_train], axis=1)
    X_train= X_train.drop(columns=['country', 'date'])
    
    #encoding testingt countries
    encodedCountries= pd.get_dummies(X_test.country, prefix='Country')
    X_test= pd.concat([encodedCountries, X_test], axis=1)
    X_test= X_test.drop(columns=['country', 'date'])
    
    #training 
    reg = xgb.XGBRegressor(n_estimators= 2000)
    reg.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=False)
    #Feature importance is the best way to get a general idea about which features model is relying on to make its prediction. This is a metric that simply sums up how many times each feature is split on. We can see day_of_week is the most common feature used to split the tree followed by month closely.
    _ = plot_importance(reg, height=0.9)
    
    #predicting
    y_pred = reg.predict(X_test)
    acc_score = reg.score(X_test, y_test)
    X_test['TrueValue']= y_test
    X_test['PredValue']= y_pred
    #plot true value and predict value to observe the correlated degree
    X_test[['TrueValue', 'PredValue']].plot(figsize=(20,10))
    print(acc_score)
    #acc_score = reg.score(X_test, y_test)
    #acc_score = accuracy_score(y_true,y_pred)
    
    #taking a look at month wise prediction and day of the week prediction; correlated degree
    monthWise_predictedResult(X_test)
    weekWise_predictedResult(X_test)

    what = X_test.loc[(X_test['month'] == 7) & (X_test['day_of_month'] == 4) & (X_test['Country_United States'] == 1)]
    print(what)
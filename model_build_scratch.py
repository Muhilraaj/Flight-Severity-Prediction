# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 22:53:59 2020

@author: muhil
"""

import pandas as pd
import numpy as pandas
from sklearn.model_selection import train_test_split


def hpt(X,theta):
    h=1/(1+np.exp(X.dot(theta)))
    return h
    
def cost_grad(X,theta,y):
    x=X.values
    y=y.values
    cost=(-1/(2*len(X)))*(y.dot(np.log(hpt(X,theta)))+(1-y).dot(np.log(1-hpt(X,theta))))
    grad=(1/len(X))*(hpt(X,theta)-y).dot(X)
    return [cost,grad]



df=pd.read_csv('train1.csv')

y=df.iloc[:,0:4]
X=df.iloc[:,4:-1]
l=[1]*len(X)
X.insert(column='bias',value=l,loc=0)
X_train,X_test,y_train,y_test=train_test_split(X,y)
theta=[0]*10
print(cost_grad(X,theta,y.iloc[:,0]))
epochs=10000
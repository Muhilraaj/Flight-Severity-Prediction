# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 22:53:59 2020

@author: muhil
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.model_selection import train_test_split


def hpt(X,theta):
    h=1/(1+np.exp(-1*X.dot(theta)))
    return h
    
def cost_grad(X,theta,y):
    x=X.values
    y=y.values
    cost=(-1/(2*len(X)))*(y.transpose().dot(np.log(hpt(X,theta)))+(1-y).transpose().dot(np.log(1-hpt(X,theta)))).diagonal()
    grad=(1/len(X))*X.transpose().dot((hpt(X,theta)-y))
    return [cost,grad]

def poly(X):
    pol=PolynomialFeatures(degree=2)
    X=pol.fit_transform(X)
    X=pd.DataFrame(X)
    return X
    


df=pd.read_csv('train1.csv')

y=df.iloc[:,0:4]
X=df.iloc[:,4:]
l=[1]*len(X)
X=poly(X)
X.insert(column='bias',value=l,loc=0)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)
theta=np.zeros([67,4])

epochs=100000
alpha=1
for i in range(0,epochs):
    cg=cost_grad(X_train,theta,y_train)
    theta=theta-alpha*cg[1]
    if i%1000==0:
        print(cg[0])
    
cg=cost_grad(X_test,theta,y_test)
print("Test")
print(cg[0])







df2=pd.read_csv("test1.csv")
f3=pd.DataFrame()
f3.insert(column="Accident_ID",value=df2.iloc[:,0],loc=0)
l=[0]*len(f3)
df2=df2.iloc[:,1:]
l=[0]*len(df2)
df2=poly(df2)
#print(df2)
df2.insert(column="bias",value=l,loc=0)
y_mtest=hpt(df2,theta)
print(y_mtest)



f3.insert(column="Severity",value=l,loc=1)
for i in range(0,4):
    for j in range(0,len(df2)):
        if y_mtest.iloc[j,i]==y_mtest.max(axis=1).iloc[j]:
            f3.iloc[j,1]=y.columns[i]
    
print(f3)
f3.to_csv("testout.csv",index=False)

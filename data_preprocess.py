# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 21:04:04 2020

@author: muhil
"""

import pandas as pd


df=pd.read_csv("test.csv")
'''sev=df['Severity']
unq_sev=sev.unique()
print(unq_sev)
for i in range(0,len(unq_sev)):
    l=list(df.Severity.map(lambda a: 1 if a==unq_sev[i] else 0))
    df.insert(column=unq_sev[i],value=l,loc=i)
df.drop('Severity',axis=1,inplace=True)'''
l=df['Accident_ID'].values
df.drop('Accident_ID',axis=1,inplace=True)
#df.iloc[:,4:]=(df.iloc[:,4:]-df.iloc[:,4:].mean())/df.iloc[:,4:].std()
df=(df-df.mean())/df.std()
df.insert(column='Accident_ID',value=l,loc=0)
print(df)
df.to_csv('test1.csv',index=False)

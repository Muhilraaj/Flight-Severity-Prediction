# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 21:04:04 2020

@author: muhil
"""

import pandas as pd


df=pd.read_csv("train.csv")
sev=df['Severity']
unq_sev=sev.unique()
print(unq_sev)
for i in range(0,len(unq_sev)):
    l=list(df.Severity.map(lambda a: 1 if a==unq_sev[i] else 0))
    df.insert(column=unq_sev[i],value=l,loc=i)
df.drop('Severity',axis=1,inplace=True)
df.drop('Accident_ID',axis=1,inplace=True)
df.to_csv('train1.csv',index=False)
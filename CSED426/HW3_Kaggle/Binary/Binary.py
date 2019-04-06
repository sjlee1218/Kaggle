#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 14:13:48 2018

@author: leeseungjoon
"""

import sklearn.tree as sktree
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train_data=pd.read_csv('/Users/leeseungjoon/Desktop/2018-2/BigData/HW3/Binary/bank_train.csv',sep= ',')
test_data=pd.read_csv('/Users/leeseungjoon/Desktop/2018-2/BigData/HW3/Binary/bank_test.csv',sep= ',')
#print(read_data.shape)
# if you want only one column, read_data['age'].

train_x=train_data.loc[:,'age':'poutcome']
df2=train_x.copy()
# if you want only values, not containing column name, type train_data.values[:,0:16]
#X_train_val=train_data.values[:,1:16]
train_y=train_data['y']

test_x=test_data.loc[:,'age':'poutcome']
df1=test_x.copy()
test_id=test_data['id']
df_id=pd.DataFrame(test_id)

df2['marital']=df2['marital'].replace('single',0)
df2['marital']=df2['marital'].replace('married',1)
df2['marital']=df2['marital'].replace('divorced',2)

df2['loan']=df2['loan'].replace('no',0)
df2['loan']=df2['loan'].replace('yes',1)

df2['education']=df2['education'].replace('tertiary',0)
df2['education']=df2['education'].replace('unknown',1)
df2['education']=df2['education'].replace('secondary',2)
df2['education']=df2['education'].replace('primary',3)

df2['housing']=df2['housing'].replace('no',0)
df2['housing']=df2['housing'].replace('yes',1)

df2['default']=df2['default'].replace('no',0)
df2['default']=df2['default'].replace('yes',1)

df2['poutcome']=df2['poutcome'].replace('success',0)
df2['poutcome']=df2['poutcome'].replace('other',1)
df2['poutcome']=df2['poutcome'].replace('failure',2)
df2['poutcome']=df2['poutcome'].replace('unknown',3)

df1['marital']=df1['marital'].replace('single',0)
df1['marital']=df1['marital'].replace('married',1)
df1['marital']=df1['marital'].replace('divorced',2)

df1['loan']=df1['loan'].replace('no',0)
df1['loan']=df1['loan'].replace('yes',1)

df1['education']=df1['education'].replace('tertiary',0)
df1['education']=df1['education'].replace('unknown',1)
df1['education']=df1['education'].replace('secondary',2)
df1['education']=df1['education'].replace('primary',3)

df1['housing']=df1['housing'].replace('no',0)
df1['housing']=df1['housing'].replace('yes',1)

df1['default']=df1['default'].replace('no',0)
df1['default']=df1['default'].replace('yes',1)

df1['poutcome']=df1['poutcome'].replace('success',0)
df1['poutcome']=df1['poutcome'].replace('other',1)
df1['poutcome']=df1['poutcome'].replace('failure',2)
df1['poutcome']=df1['poutcome'].replace('unknown',3)


df3=df2.drop(columns=['age', 'job','balance','contact','day','month','duration','campaign','pdays','previous'])
# df3 이 train data
df4=df1.drop(columns=['age', 'job','balance','contact','day','month','duration','campaign','pdays','previous'])


clf=sktree.DecisionTreeClassifier(max_depth=4,random_state=0)
clf.fit(df3,train_y)

y_pred = clf.predict(df4)
y_col=pd.DataFrame(y_pred)

df_id['y']=y_col
df_id.to_csv('/Users/leeseungjoon/Desktop/2018-2/BigData/HW3/Binary/out.csv',index=False)

#print("훈련 세트 정확도: {:.3f}".format(tree.score(df3, train_y)))

#print("테스트 세트 정확도: {:.3f}".format(tree.score(df2, test_y)))

#f, ax = plt.subplots(1, 1, figsize=(7, 7))
#train_x[['poutcome','y']].groupby(['poutcome'],as_index=True).mean().sort_values(by='y', ascending=False).plot.bar(ax=ax)
#train_x[['job','y']].groupby(['job'],as_index=True).mean().sort_values(by='y', ascending=False).plot.bar(ax=ax)
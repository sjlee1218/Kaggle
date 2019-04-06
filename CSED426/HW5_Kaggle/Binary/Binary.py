#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 00:16:39 2018

@author: leeseungjoon
"""

import pandas as pd
from sklearn import model_selection
from sklearn.svm import SVC
from sklearn import preprocessing
import seaborn as sns
from sklearn import model_selection
from sklearn.metrics import fbeta_score
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score

train_data=pd.read_csv('/Users/leeseungjoon/Desktop/2018-2/BigData/HW5/Binary/bank_train.csv',sep= ',')
test_data=pd.read_csv('/Users/leeseungjoon/Desktop/2018-2/BigData/HW5/Binary/bank_test.csv',sep= ',')

train_x=train_data.loc[:,'age':'poutcome']
df2=train_x.copy()
train_y=train_data['y']

test_x=test_data.loc[:,'age':'poutcome']
df1=test_x.copy()
test_id=test_data['id']
df_id=pd.DataFrame(test_id)

names=df1.columns

df2['marital']=df2['marital'].replace('single',0)
df2['marital']=df2['marital'].replace('married',1)
df2['marital']=df2['marital'].replace('divorced',2)

df2['education']=df2['education'].replace('tertiary',0)
df2['education']=df2['education'].replace('unknown',1)
df2['education']=df2['education'].replace('secondary',2)
df2['education']=df2['education'].replace('primary',3)

df2['default']=df2['default'].replace('no',0) # 돈 안 갚은 게 있는가?
df2['default']=df2['default'].replace('yes',1)

df2['housing']=df2['housing'].replace('no',0) # 주택 담보 대출이 있는가?
df2['housing']=df2['housing'].replace('yes',1)

df2['loan']=df2['loan'].replace('no',0) # 개인 대출 받는 게 있는가?
df2['loan']=df2['loan'].replace('yes',1)

df2['job']=df2['job'].replace('student',0)
df2['job']=df2['job'].replace('retired',1)
df2['job']=df2['job'].replace('unemployed',2)
df2['job']=df2['job'].replace('management',3)
df2['job']=df2['job'].replace('unknown',4)
df2['job']=df2['job'].replace('self-employed',5)
df2['job']=df2['job'].replace('admin.',6)
df2['job']=df2['job'].replace('technician',7)
df2['job']=df2['job'].replace('housemaid',8)
df2['job']=df2['job'].replace('services',9)
df2['job']=df2['job'].replace('entrepreneur',10)
df2['job']=df2['job'].replace('blue-collar',11)

df1['marital']=df1['marital'].replace('single',0)
df1['marital']=df1['marital'].replace('married',1)
df1['marital']=df1['marital'].replace('divorced',2)

df1['education']=df1['education'].replace('tertiary',0)
df1['education']=df1['education'].replace('unknown',1)
df1['education']=df1['education'].replace('secondary',2)
df1['education']=df1['education'].replace('primary',3)

df1['default']=df1['default'].replace('no',0) # 돈 안 갚은 게 있는가?
df1['default']=df1['default'].replace('yes',1)

df1['housing']=df1['housing'].replace('no',0) # 주택 담보 대출이 있는가?
df1['housing']=df1['housing'].replace('yes',1)

df1['loan']=df1['loan'].replace('no',0) # 개인 대출 받는 게 있는가?
df1['loan']=df1['loan'].replace('yes',1)

df1['job']=df1['job'].replace('student',0)
df1['job']=df1['job'].replace('retired',1)
df1['job']=df1['job'].replace('unemployed',2)
df1['job']=df1['job'].replace('management',3)
df1['job']=df1['job'].replace('unknown',4)
df1['job']=df1['job'].replace('self-employed',5)
df1['job']=df1['job'].replace('admin.',6)
df1['job']=df1['job'].replace('technician',7)
df1['job']=df1['job'].replace('housemaid',8)
df1['job']=df1['job'].replace('services',9)
df1['job']=df1['job'].replace('entrepreneur',10)
df1['job']=df1['job'].replace('blue-collar',11)

df2.loc[df2['balance']>=35000,'balance']=35000

df2.loc[df2['duration']>=2500, 'duration']=2500

df2['contact']=df2['contact'].replace('cellular',0)
df2['contact']=df2['contact'].replace('telephone',1)
df2['contact']=df2['contact'].replace('unknown',2)

df2['month']=df2['month'].replace('mar',0)
df2['month']=df2['month'].replace('sep',1)
df2['month']=df2['month'].replace('dec',2)
df2['month']=df2['month'].replace('oct',3)
df2['month']=df2['month'].replace('apr',4)
df2['month']=df2['month'].replace('feb',5)
df2['month']=df2['month'].replace('aug',6)
df2['month']=df2['month'].replace('jun',7)
df2['month']=df2['month'].replace('jan',8)
df2['month']=df2['month'].replace('nov',9)
df2['month']=df2['month'].replace('jul',10)
df2['month']=df2['month'].replace('may',11)

df1['contact']=df1['contact'].replace('cellular',0)
df1['contact']=df1['contact'].replace('telephone',1)
df1['contact']=df1['contact'].replace('unknown',2)

df1['month']=df1['month'].replace('mar',0)
df1['month']=df1['month'].replace('sep',1)
df1['month']=df1['month'].replace('dec',2)
df1['month']=df1['month'].replace('oct',3)
df1['month']=df1['month'].replace('apr',4)
df1['month']=df1['month'].replace('feb',5)
df1['month']=df1['month'].replace('aug',6)
df1['month']=df1['month'].replace('jun',7)
df1['month']=df1['month'].replace('jan',8)
df1['month']=df1['month'].replace('nov',9)
df1['month']=df1['month'].replace('jul',10)
df1['month']=df1['month'].replace('may',11)

df2.loc[df2['campaign']>=30, 'campaign']=30

#df2['pdays']=preprocessing.robust_scale(df2['pdays'])

df2.loc[df2['previous']>=80, 'previous']=80

df2['poutcome']=df2['poutcome'].replace('success',0)
df2['poutcome']=df2['poutcome'].replace('other',1)
df2['poutcome']=df2['poutcome'].replace('failure',2)
df2['poutcome']=df2['poutcome'].replace('unknown',3)

df1['poutcome']=df1['poutcome'].replace('success',0)
df1['poutcome']=df1['poutcome'].replace('other',1)
df1['poutcome']=df1['poutcome'].replace('failure',2)
df1['poutcome']=df1['poutcome'].replace('unknown',3)


df2=df2.drop('day',1)
df1=df1.drop('day',1)

scaler = preprocessing.StandardScaler()
scaler = scaler.fit(df2)
new_train = pd.DataFrame(scaler.transform(df2))
new_test = pd.DataFrame(scaler.transform(df1))


X_train_sub, X_validation_sub, y_train_sub, y_validation_sub = model_selection.train_test_split(new_train, train_y, random_state=0)
"""

X_train_sub, X_validation_sub, y_train_sub, y_validation_sub = model_selection.train_test_split(new_train, train_y, random_state=0)
clf.fit(X_train_sub, y_train_sub)
print("Accuracy score (training): {0:.3f}".format(clf.score(X_train_sub, y_train_sub)))
print("Accuracy score (validation): {0:.3f}".format(clf.score(X_validation_sub, y_validation_sub)))
print("feautre importances")
print(clf.feature_importances_)
print()


class_weight : {dict, ‘balanced’}, optional


clf=LinearSVC(tol=0.001, C=1.5,random_state=0,fit_intercept=True,class_weight='balanced')
clf.fit(X_train_sub, y_train_sub)
y_pred=clf.predict(X_validation_sub)
fbeta_score(y_validation_sub, y_pred, beta=1)
Out[78]: 0.5304381543233812

clf=SVC(C=1.5, cache_size=2048,gamma="auto", random_state=0, degree=4, tol=0.001, class_weight='balanced')
clf.fit(X_train_sub, y_train_sub)
y_pred=clf.predict(X_validation_sub)
fbeta_score(y_validation_sub, y_pred, beta=1)
Out[88]: 0.5423728813559322


clf=SVC(C=1.5,kernel='poly',cache_size=2048,gamma="auto", random_state=0, degree=4, tol=0.001, class_weight='balanced')
clf.fit(X_train_sub, y_train_sub)
y_pred=clf.predict(X_validation_sub)
fbeta_score(y_validation_sub, y_pred, beta=1)
Out[9]: 0.5510563380281691

print("C=0.4, score: {0:.3f}",format(fbeta_score(y_validation_sub, y_pred, beta=1)))

clf=SVC(C=0.3,kernel='poly',cache_size=2048,gamma="auto", random_state=0, degree=4, tol=0.001, class_weight='balanced')
clf.fit(X_train_sub, y_train_sub)
y_pred=clf.predict(X_validation_sub)
print("C=0.3, score: {0:.3f}",format(fbeta_score(y_validation_sub, y_pred, beta=1)))
C=0.3, score: {0:.3f} 0.5663801337153773

"""
clf=SVC(C=0.3,kernel='poly',cache_size=2048,gamma="auto", random_state=0, degree=4, tol=0.001, class_weight='balanced')
clf.fit(X_train_sub, y_train_sub)
y_pred=clf.predict(X_validation_sub)
print("C=0.3, score: {0:.3f}",format(f1_score(y_validation_sub, y_pred)))

clf=SVC(C=0.3,kernel='poly',cache_size=2048,gamma="auto", random_state=0, degree=4, tol=0.001, class_weight='balanced')
clf.fit(new_train, train_y)

y_pred=clf.predict(new_test)
y_col=pd.DataFrame(y_pred)
df_id['y']=y_col
df_id.to_csv('/Users/leeseungjoon/Desktop/2018-2/BigData/HW5/Binary/without_date_SVC_balanced.csv',index=False)


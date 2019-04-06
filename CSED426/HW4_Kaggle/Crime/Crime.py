#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 20:32:44 2018

@author: leeseungjoon
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import graphviz
import matplotlib.pyplot as plt
from sklearn import preprocessing
import seaborn as sns
from sklearn import model_selection


train_data=pd.read_csv('/Users/leeseungjoon/Desktop/2018-2/BigData/HW4/Crime/crime_train.csv',sep= ',')
test_data=pd.read_csv('/Users/leeseungjoon/Desktop/2018-2/BigData/HW4/Crime/crime_test.csv',sep= ',')
sub=pd.read_csv('/Users/leeseungjoon/Desktop/2018-2/BigData/HW4/Crime/crime_sample_submission.csv',sep= ',')

sub=sub.drop('id',axis=1)
crime_category=sub.columns # 범죄 종류들의 열. 근데 배열은 아닌듯? Index(['ARSON', ... 'WEAPON LAWS'], dtype='object') 이렇게 생김

"""
crime_category
Out[27]: 
Index([, 'ASSAULT',
       'DISORDERLY CONDUCT', , 'DRUG/NARCOTIC',
       , 'EMBEZZLEMENT' , 'FAMILY OFFENSES',
       'FORGERY/COUNTERFEITING', 'FRAUD', , 'KIDNAPPING',
       'LARCENY/THEFT', 'LIQUOR LAWS', 'LOITERING', 'MISSING PERSON',
       'NON-CRIMINAL', 'OTHER OFFENSES', 'PORNOGRAPHY/OBSCENE MAT',
       'PROSTITUTION', 'RECOVERED VEHICLE', 'ROBBERY', 'RUNAWAY',
       'SECONDARY CODES', 'SEX OFFENSES FORCIBLE', 'SEX OFFENSES NON FORCIBLE',
       'STOLEN PROPERTY', 'SUICIDE', 'SUSPICIOUS OCC', 'TREA', 'TRESPASS',
       'VANDALISM', 'VEHICLE THEFT', 'WARRANTS', 'WEAPON LAWS'],
      dtype='object')
"""
"""
df=pd.DataFrame(train_data['Category'])
df['Descript']=train_data['Descript']

df[df['Category'].str.contains("THEFT")==True]
                                                                                                    Category    Descript

df.loc[df['Descript'].str.contains("ARSON")==True]=0                            #                       577     558
df.loc[df['Descript'].str.contains("CHECKS, NON-SUFFICIENT FUNDS")==True]=1      #                       264     264
df.loc[df['Descript'].str.contains("DISSUADING WITNESS, VICTIM")==True]=2
df.loc[df['Descript'].str.contains("BRIBERY OF WITNESSES")==True]=2               #                       191     177
df.loc[df['Descript'].str.contains("POSSESSION OF BURGLARY")==True]=3 # OTHER OFFENSES                     1450  1450
df.loc[df['Descript'].str.contains("BURGLARY")==True]=4 #                                                23508      23508
df.loc[df['Descript'].str.contains("UNDER INFLUENCE OF ALCOHOL IN A PUBLIC PLACE")==True]=5 #DRUNKNESS     2728  2728
df.loc[df['Descript'].str.contains("DRIVING WHILE UNDER THE INFLUENCE OF")==True]=6 #Driving            1435    1435
df.loc[df['Descript'].str.contains("EXTORTION")==True]=7
df.loc[df['Descript'].str.contains("GAMBLING")==True]=8

df.loc[df['Descript'].str.contains("THEFT")==True]=0     111663    119593
df.loc[df['Descript'].str.contains("STOLEN")==True]=1
df.loc[df['Descript'].str.contains("VEHICLE")==True]=1
"""
train_target=train_data['Category']

df1=train_data.drop('Category',1)
df3=test_data

test_id=pd.DataFrame(test_data['id'])


df1['DayOfWeek']=df1['DayOfWeek'].replace('Sunday',0)
df1['DayOfWeek']=df1['DayOfWeek'].replace('Monday',1)
df1['DayOfWeek']=df1['DayOfWeek'].replace('Tuesday',2)
df1['DayOfWeek']=df1['DayOfWeek'].replace('Wednesday',3)
df1['DayOfWeek']=df1['DayOfWeek'].replace('Thursday',4)
df1['DayOfWeek']=df1['DayOfWeek'].replace('Friday',5)
df1['DayOfWeek']=df1['DayOfWeek'].replace('Saturday',6)

df3['DayOfWeek']=df3['DayOfWeek'].replace('Sunday',0)
df3['DayOfWeek']=df3['DayOfWeek'].replace('Monday',1)
df3['DayOfWeek']=df3['DayOfWeek'].replace('Tuesday',2)
df3['DayOfWeek']=df3['DayOfWeek'].replace('Wednesday',3)
df3['DayOfWeek']=df3['DayOfWeek'].replace('Thursday',4)
df3['DayOfWeek']=df3['DayOfWeek'].replace('Friday',5)
df3['DayOfWeek']=df3['DayOfWeek'].replace('Saturday',6)


df1['PdDistrict']=df1['PdDistrict'].replace('RICHMOND',0)
df1['PdDistrict']=df1['PdDistrict'].replace('SOUTHERN',1)
df1['PdDistrict']=df1['PdDistrict'].replace('MISSION',2)
df1['PdDistrict']=df1['PdDistrict'].replace('CENTRAL',3)
df1['PdDistrict']=df1['PdDistrict'].replace('BAYVIEW',4)
df1['PdDistrict']=df1['PdDistrict'].replace('NORTHERN',5)
df1['PdDistrict']=df1['PdDistrict'].replace('TARAVAL',6)
df1['PdDistrict']=df1['PdDistrict'].replace('PARK',7)
df1['PdDistrict']=df1['PdDistrict'].replace('INGLESIDE',8)
df1['PdDistrict']=df1['PdDistrict'].replace('TENDERLOIN',9)

df3['PdDistrict']=df3['PdDistrict'].replace('RICHMOND',0)
df3['PdDistrict']=df3['PdDistrict'].replace('SOUTHERN',1)
df3['PdDistrict']=df3['PdDistrict'].replace('MISSION',2)
df3['PdDistrict']=df3['PdDistrict'].replace('CENTRAL',3)
df3['PdDistrict']=df3['PdDistrict'].replace('BAYVIEW',4)
df3['PdDistrict']=df3['PdDistrict'].replace('NORTHERN',5)
df3['PdDistrict']=df3['PdDistrict'].replace('TARAVAL',6)
df3['PdDistrict']=df3['PdDistrict'].replace('PARK',7)
df3['PdDistrict']=df3['PdDistrict'].replace('INGLESIDE',8)
df3['PdDistrict']=df3['PdDistrict'].replace('TENDERLOIN',9)


train_dates_series=pd.Series(train_data['Dates'])
train_year=pd.DataFrame(train_dates_series.str.split('-').str.get(0).astype(int))
train_month=pd.DataFrame(train_dates_series.str.split('-').str.get(1).astype(int))
temp=train_dates_series.str.split('-').str.get(2)
train_day=pd.DataFrame(temp.str.split(' ').str.get(0).astype(int))
temp2=temp.str.split(' ').str.get(1)
train_hour=pd.DataFrame(temp2.str.split(':').str.get(0).astype(int))
train_min=pd.DataFrame(temp2.str.split(':').str.get(1).astype(int))

train_year_temp=train_year[train_year.select_dtypes(include=['number']).columns] * 365
train_month_temp=train_month[train_month.select_dtypes(include=['number']).columns] * 30
train_date_col=train_year_temp.add(train_month_temp)
train_date_col=train_date_col.add(train_day)

train_hour_temp=train_hour[train_hour.select_dtypes(include=['number']).columns] * 60
train_time_col=train_hour_temp.add(train_min)

test_dates_series=pd.Series(test_data['Dates'])
test_year=pd.DataFrame(test_dates_series.str.split('-').str.get(0).astype(int))
test_month=pd.DataFrame(test_dates_series.str.split('-').str.get(1).astype(int))
temp=test_dates_series.str.split('-').str.get(2)
test_day=pd.DataFrame(temp.str.split(' ').str.get(0).astype(int))
temp2=temp.str.split(' ').str.get(1)
test_hour=pd.DataFrame(temp2.str.split(':').str.get(0).astype(int))
test_min=pd.DataFrame(temp2.str.split(':').str.get(1).astype(int))

test_year_temp=test_year[test_year.select_dtypes(include=['number']).columns] * 365
test_month_temp=test_month[test_month.select_dtypes(include=['number']).columns] * 30
test_date_col=test_year_temp.add(test_month_temp)
test_date_col=test_date_col.add(test_day)

test_hour_temp=test_hour[test_hour.select_dtypes(include=['number']).columns] * 60
test_time_col=test_hour_temp.add(test_min)

df1=df1.drop('Descript',1)
df1=df1.drop("Address",1)
df1=df1.drop("Dates",1)
df1=df1.drop('Resolution',1)

df3=df3.drop("Address",1)
df3=df3.drop("Dates",1)
df3=df3.drop('id',1)

"""
scaler = MinMaxScaler()
numerical = ['X','Y','Time','Date']
df1[numerical] = scaler.fit_transform(df1[numerical])
df3[numerical] = scaler.transform(df3[numerical])
"""
X_train_sub, X_validation_sub, y_train_sub, y_validation_sub = model_selection.train_test_split(df1, train_target, random_state=0)

clf = GradientBoostingClassifier(n_estimators=100, learning_rate = 0.1, min_samples_leaf=20, max_depth = 5, random_state = 0)
clf.fit(X_train_sub, y_train_sub)
print("Accuracy score (training): {0:.3f}".format(clf.score(X_train_sub, y_train_sub)))
print("Accuracy score (validation): {0:.3f}".format(clf.score(X_validation_sub, y_validation_sub)))
print()
    
    
    
"""

clf = GradientBoostingClassifier(n_estimators=100, learning_rate = 0.5, min_samples_leaf=10, max_depth = 5, random_state = 0)

X_train_sub, X_validation_sub, y_train_sub, y_validation_sub = model_selection.train_test_split(df1, train_target, random_state=0)

learning_rates = [0.05, 0.1, 0.25, 0.5, 0.75, 1]
for learning_rate in learning_rates:
    clf = GradientBoostingClassifier(n_estimators=300, learning_rate = learning_rate, min_samples_leaf=30, max_depth = 5, random_state = 0)
    clf.fit(X_train_sub, y_train_sub)
    print("Learning rate: ", learning_rate)
    print("Accuracy score (training): {0:.3f}".format(clf.score(X_train_sub, y_train_sub)))
    print("Accuracy score (validation): {0:.3f}".format(clf.score(X_validation_sub, y_validation_sub)))
    print()


"""

"""
clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2),learning_rate=learning_rate,n_estimators=500)
clf.fit(X_train_sub, y_train_sub)
print("Learning rate: ", learning_rate)
print("Accuracy score (training): {0:.3f}".format(clf.score(X_train_sub, y_train_sub)))
print("Accuracy score (validation): {0:.3f}".format(clf.score(X_validation_sub, y_validation_sub)))
print()



array_prob=clf.predict_proba(df3)

df_prob=pd.DataFrame(data=array_prob, columns=crime_category)
df_out=test_id.join(df_prob)
df_out['id']=df_out['id'].astype(float)

df_out.to_csv('/Users/leeseungjoon/Desktop/2018-2/BigData/HW4/Crime/insane.csv',index=False)
"""
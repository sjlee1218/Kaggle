# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 02:00:59 2018

@author: dltmd
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.svm import SVC
from sklearn.metrics import log_loss
import seaborn as sns

train_data=pd.read_csv('C:/Users/dltmd/iCloudDrive/Desktop/2018-2/BigData/HW5/Crime/crime_train.csv',sep= ',')
test_data=pd.read_csv('C:/Users/dltmd/iCloudDrive/Desktop/2018-2/BigData/HW5/Crime/crime_test.csv',sep= ',')
sub=pd.read_csv('C:/Users/dltmd/iCloudDrive/Desktop/2018-2/BigData/HW5/Crime/crime_sample_submission.csv',sep= ',')

sub=sub.drop('id',axis=1)
crime_category=sub.columns # 범죄 종류들의 열. 근데 배열은 아닌듯? Index(['ARSON', ... 'WEAPON LAWS'], dtype='object') 이렇게 생김

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
df1['Time']=train_time_col
df1['Date']=train_date_col
df1=df1.drop('Resolution',1)

df3=df3.drop("Address",1)
df3=df3.drop("Dates",1)
df3=df3.drop('id',1)
df3['Time']=test_time_col
df3['Date']=test_date_col

df1.loc[df1['X']>=-121, 'X']=-122.30
df3.loc[df3['X']>=-121, 'X']=-122.30

df1.loc[df1['Y']>=40, 'Y']=37.85
df3.loc[df3['Y']>=40, 'Y']=37.85

scaler = preprocessing.StandardScaler()
scaler = scaler.fit(df1)
new_train = pd.DataFrame(scaler.transform(df1))
new_test = pd.DataFrame(scaler.transform(df3))


"""

clf = GradientBoostingClassifier(n_estimators=100, learning_rate = 0.5, min_samples_leaf=10, max_depth = 5, random_state = 0)

X_train_sub, X_validation_sub, y_train_sub, y_validation_sub = model_selection.train_test_split(new_train, train_target, random_state=0)


model1=SVC(C=1, kernel='poly',cache_size=2048,gamma="auto", random_state=0, degree=4, tol=0.001, class_weight='balanced',probability=True)
model1.fit(X_train_sub, y_train_sub)
y_pred=model1.predict_proba(X_validation_sub)
print("C=1, balnaced, score: {0:.4f}".format(log_loss(y_validation_sub, y_pred)))
array_prob=model1.predict_proba(new_test)
df_prob=pd.DataFrame(data=array_prob, columns=crime_category)
df_out=test_id.join(df_prob)
df_out['id']=df_out['id'].astype(float)
df_out.to_csv('C:/Users/dltmd/iCloudDrive/Desktop/2018-2/BigData/HW5/Crime/balanced_poly_C15_degree4.csv',index=False)

model2=SVC(C=1,kernel='poly',cache_size=2048,gamma="auto", random_state=0, degree=4, tol=0.001,probability=True)
model2.fit(X_train_sub, y_train_sub)
y_pred=model2.predict_proba(X_validation_sub)
print("C=1, unbalanced, score: {0:.4f}".format(log_loss(y_validation_sub, y_pred)))
array_prob=model2.predict_proba(new_test)
df_prob=pd.DataFrame(data=array_prob, columns=crime_category)
df_out=test_id.join(df_prob)
df_out['id']=df_out['id'].astype(float)
df_out.to_csv('C:/Users/dltmd/iCloudDrive/Desktop/2018-2/BigData/HW5/Crime/unbalanced.csv',index=False)

model3=SVC(C=0.5, kernel='poly',cache_size=2048,gamma="auto", random_state=0, degree=4, tol=0.001, class_weight='balanced',probability=True)
model3.fit(X_train_sub, y_train_sub)
y_pred=model3.predict_proba(X_validation_sub)
print("C=0.5, balanced, score: {0:.4f}".format(log_loss(y_validation_sub, y_pred)))
array_prob=model3.predict_proba(new_test)
df_prob=pd.DataFrame(data=array_prob, columns=crime_category)
df_out=test_id.join(df_prob)
df_out['id']=df_out['id'].astype(float)
df_out.to_csv('C:/Users/dltmd/iCloudDrive/Desktop/2018-2/BigData/HW5/Crime/balanced_poly_C05_degree4.csv',index=False)

model4=SVC(C=1.5, kernel='poly',cache_size=2048,gamma="auto", random_state=0, degree=4, tol=0.001, class_weight='balanced',probability=True)
model4.fit(X_train_sub, y_train_sub)
y_pred=model4.predict_proba(X_validation_sub)
print("C=1.5, balnaced, score: {0:.4f}".format(log_loss(y_validation_sub, y_pred)))
array_prob=model4.predict_proba(new_test)
df_prob=pd.DataFrame(data=array_prob, columns=crime_category)
df_out=test_id.join(df_prob)
df_out['id']=df_out['id'].astype(float)
df_out.to_csv('C:/Users/dltmd/iCloudDrive/Desktop/2018-2/BigData/HW5/Crime/balanced_poly_C15_degree4.csv',index=False)


"""

X_train_sub, X_validation_sub, y_train_sub, y_validation_sub = model_selection.train_test_split(new_train, train_target, random_state=0)

clf=SVC(C=0.3,kernel='poly',cache_size=2048,gamma="auto", random_state=0, degree=4, tol=0.01, class_weight='balanced')

clf.fit(new_train, train_target)

array_prob=clf.predict_proba(new_test)
df_prob=pd.DataFrame(data=array_prob, columns=crime_category)
df_out=test_id.join(df_prob)
df_out['id']=df_out['id'].astype(float)

df_out.to_csv('C:/Users/dltmd/iCloudDrive/Desktop/2018-2/BigData/HW5/Crime/insane.csv',index=False)

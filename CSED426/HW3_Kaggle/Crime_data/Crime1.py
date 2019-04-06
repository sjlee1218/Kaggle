#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 04:47:47 2018

@author: leeseungjoon
"""

import sklearn.tree as sktree
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,LabelEncoder

train_data=pd.read_csv('/Users/leeseungjoon/Desktop/2018-2/BigData/HW3/Crime_data/crime_train.csv',sep= ',')
test_data=pd.read_csv('/Users/leeseungjoon/Desktop/2018-2/BigData/HW3/Crime_data/crime_test.csv',sep= ',')
sub=pd.read_csv('/Users/leeseungjoon/Desktop/2018-2/BigData/HW3/Crime_data/crime_sample_submission.csv',sep= ',')

sub=sub.drop('id',axis=1)
crime_category=sub.columns # 범죄 종류들의 열. 근데 배열은 아닌듯? Index(['ARSON', ... 'WEAPON LAWS'], dtype='object') 이렇게 생김

#우리가 해야하는 일은, 이 친구가 어떤 범죄를 일으켰는지 맞추는 것
"""
Dates: timestamp of the crime incident
Descript: detailed description of the crime incident (only in train.csv)
DayOfWeek: the day of the week
PdDistrict: name of the Police Department District
Resolution: how the crime incident was resolved (only in train.csv)
Address: the approximate street address of the crime incident
X: Longitude
Y: Latitude

얘네들이 우리가 봐야 할 feature들

"""
train_target=train_data['Category']

df1=train_data.drop('Category',1)
df3=test_data

test_id=pd.DataFrame(test_data['id'])

"""
    PdDistrict
0     RICHMOND
1     SOUTHERN
2      MISSION
5      CENTRAL
7      BAYVIEW
9     NORTHERN
10     TARAVAL
12        PARK
17   INGLESIDE
35  TENDERLOIN
"""

train_dates_series=pd.Series(train_data['Dates'])
train_year=pd.DataFrame(train_dates_series.str.split('-').str.get(0).astype(int))
train_month=pd.DataFrame(train_dates_series.str.split('-').str.get(1).astype(int))
temp=train_dates_series.str.split('-').str.get(2)
train_day=pd.DataFrame(temp.str.split(' ').str.get(0).astype(int))
temp2=temp.str.split(' ').str.get(1)
train_hour=pd.DataFrame(temp2.str.split(':').str.get(0).astype(int))
train_min=pd.DataFrame(temp2.str.split(':').str.get(1).astype(int))

train_year_temp=train_year[train_year.select_dtypes(include=['number']).columns] * 10000
train_month_temp=train_month[train_month.select_dtypes(include=['number']).columns] * 100
train_date_col=train_year_temp.add(train_month_temp)
train_date_col=train_date_col.add(train_day)

train_hour_temp=train_hour[train_hour.select_dtypes(include=['number']).columns] * 100
train_time_col=train_hour_temp.add(train_min)

test_dates_series=pd.Series(test_data['Dates'])
test_year=pd.DataFrame(test_dates_series.str.split('-').str.get(0).astype(int))
test_month=pd.DataFrame(test_dates_series.str.split('-').str.get(1).astype(int))
temp=test_dates_series.str.split('-').str.get(2)
test_day=pd.DataFrame(temp.str.split(' ').str.get(0).astype(int))
temp2=temp.str.split(' ').str.get(1)
test_hour=pd.DataFrame(temp2.str.split(':').str.get(0).astype(int))
test_min=pd.DataFrame(temp2.str.split(':').str.get(1).astype(int))

test_year_temp=test_year[test_year.select_dtypes(include=['number']).columns] * 10000
test_month_temp=test_month[test_month.select_dtypes(include=['number']).columns] * 100
test_date_col=test_year_temp.add(test_month_temp)
test_date_col=test_date_col.add(test_day)

test_hour_temp=test_hour[test_hour.select_dtypes(include=['number']).columns] * 100
test_time_col=test_hour_temp.add(test_min)

# train_year[train_year.select_dtypes(include=['number']).columns] *=3


"""
범죄니까, 날짜보다도 시간이랑 관련이 더 많을 듯. 날짜랑 시간이랑 컬럼을 나눠버리자.

근데 하고 나니까, Dates랑 Time 숫자가 너무 큰 듯. 정규화 필요할 수도

"""

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

"""
                                   Resolution
0                                        NONE
3                               ARREST, CITED
12                             ARREST, BOOKED
17                                  UNFOUNDED
30                                    LOCATED
32                          PSYCHOPATHIC CASE
96                          JUVENILE DIVERTED
143                       JUVENILE ADMONISHED
168                     EXCEPTIONAL CLEARANCE
259                           JUVENILE BOOKED
333                            JUVENILE CITED
465    DISTRICT ATTORNEY REFUSES TO PROSECUTE
771              PROSECUTED BY OUTSIDE AGENCY
988          COMPLAINANT REFUSES TO PROSECUTE
995                            NOT PROSECUTED
1578   CLEARED-CONTACT JUVENILE FOR MORE INFO
17649           PROSECUTED FOR LESSER OFFENSE
"""
"""
df1['Resolution']=df1['Resolution'].replace('NONE',0)
df1['Resolution']=df1['Resolution'].replace('ARREST, CITED',1)
df1['Resolution']=df1['Resolution'].replace('ARREST, BOOKED',2)
df1['Resolution']=df1['Resolution'].replace('UNFOUNDED',3)
df1['Resolution']=df1['Resolution'].replace('LOCATED',4)
df1['Resolution']=df1['Resolution'].replace('PSYCHOPATHIC CASE',5)
df1['Resolution']=df1['Resolution'].replace('JUVENILE DIVERTED',6)
df1['Resolution']=df1['Resolution'].replace('JUVENILE ADMONISHED',7)
df1['Resolution']=df1['Resolution'].replace('EXCEPTIONAL CLEARANCE',8)
df1['Resolution']=df1['Resolution'].replace('JUVENILE BOOKED',9)
df1['Resolution']=df1['Resolution'].replace('JUVENILE CITED',10)
df1['Resolution']=df1['Resolution'].replace('DISTRICT ATTORNEY REFUSES TO PROSECUTE',11)
df1['Resolution']=df1['Resolution'].replace('PROSECUTED BY OUTSIDE AGENCY',12)
df1['Resolution']=df1['Resolution'].replace('COMPLAINANT REFUSES TO PROSECUTE',13)
df1['Resolution']=df1['Resolution'].replace('NOT PROSECUTED',14)
df1['Resolution']=df1['Resolution'].replace('CLEARED-CONTACT JUVENILE FOR MORE INFO',15)
df1['Resolution']=df1['Resolution'].replace('PROSECUTED FOR LESSER OFFENSE',16)
"""
'''

df3['Resolution']=df3['Resolution'].replace('NONE',0)
df3['Resolution']=df3['Resolution'].replace('ARREST, CITED',1)
df3['Resolution']=df3['Resolution'].replace('ARREST, BOOKED',2)
df3['Resolution']=df3['Resolution'].replace('UNFOUNDED',3)
df3['Resolution']=df3['Resolution'].replace('LOCATED',4)
df3['Resolution']=df3['Resolution'].replace('PSYCHOPATHIC CASE',5)
df3['Resolution']=df3['Resolution'].replace('JUVENILE DIVERTED',6)
df3['Resolution']=df3['Resolution'].replace('JUVENILE ADMONISHED',7)
df3['Resolution']=df3['Resolution'].replace('EXCEPTIONAL CLEARANCE',8)
df3['Resolution']=df3['Resolution'].replace('JUVENILE BOOKED',9)
df3['Resolution']=df3['Resolution'].replace('JUVENILE CITED',10)
df3['Resolution']=df3['Resolution'].replace('DISTRICT ATTORNEY REFUSES TO PROSECUTE',11)
df3['Resolution']=df3['Resolution'].replace('PROSECUTED BY OUTSIDE AGENCY',12)
df3['Resolution']=df3['Resolution'].replace('COMPLAINANT REFUSES TO PROSECUTE',13)
df3['Resolution']=df3['Resolution'].replace('NOT PROSECUTED',14)
df3['Resolution']=df3['Resolution'].replace('CLEARED-CONTACT JUVENILE FOR MORE INFO',15)
df3['Resolution']=df3['Resolution'].replace('PROSECUTED FOR LESSER OFFENSE',16)

train에는 resolution이 있는데, test에는 없음
'''

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


df1=df1.drop('Descript',1)
df1=df1.drop("Address",1)
df1=df1.drop("Dates",1)

df3=df3.drop("Address",1)
df3=df3.drop("Dates",1)
df3=df3.drop('id',1)


"""
Dates 날리고, 날짜 컬럼 시간 컬럼 각각 추가.

train_year[train_year.select_dtypes(include=['number']).columns] *= 3

"""


#df1['Dates']=train_date_col
#df1['Time']=train_time_col
#df3['Dates']=test_date_col
#df3['Time']=test_time_col

# df1이랑 df2가 train
#df3은 test

"""
df.loc[df.First_name == 'Bill', 'name_match'] = 'Match'  
df.loc[df.First_name != 'Bill', 'name_match'] = 'Mis-Match'
"""


"""

clf_resol=sktree.DecisionTreeClassifier(max_depth=3,random_state=0)
clf_resol.fit(df4, df1['Resolution']) #df1에서 resolution 뺀거로 resolution column 유추
array_resol=pd.DataFrame(clf_resol.predict(df3))

df3.insert(loc=2, column='Resolution', value=array_resol)

"""

"""
add=train_data.Address.copy(True)
of=add.loc[add.str.contains("of")]
slash=add.loc[add.str.contains("/")]
temp1=of.str.split(' ').str.get(3)
temp1.loc[temp1.str.len()==2]=of.str.split(' ').str.get(3)+" "+of.str.split(' ').str.get(4)
temp1.loc[temp1.str.len()==3]=of.str.split(' ').str.get(3)+" "+of.str.split(' ').str.get(4)
temp2=slash.str.split(' ').str.get(0)
temp2.loc[temp2.str.len()==2]=slash.str.split(' ').str.get(0)+" "+slash.str.split(' ').str.get(1)
temp2.loc[temp2.str.len()==3]=slash.str.split(' ').str.get(0)+" "+slash.str.split(' ').str.get(1)
train_address=pd.DataFrame(temp1.append(temp2).sort_index())

add=test_data.Address.copy(True)
of=add.loc[add.str.contains("of")]
slash=add.loc[add.str.contains("/")]
temp1=of.str.split(' ').str.get(3)
temp1.loc[temp1.str.len()==2]=of.str.split(' ').str.get(3)+" "+of.str.split(' ').str.get(4)
temp1.loc[temp1.str.len()==3]=of.str.split(' ').str.get(3)+" "+of.str.split(' ').str.get(4)
temp2=slash.str.split(' ').str.get(0)
temp2.loc[temp2.str.len()==2]=slash.str.split(' ').str.get(0)+" "+slash.str.split(' ').str.get(1)
temp2.loc[temp2.str.len()==3]=slash.str.split(' ').str.get(0)+" "+slash.str.split(' ').str.get(1)
test_address=pd.DataFrame(temp1.append(temp2).sort_index())

df1['Address']=train_address
df3['Address']=test_address
# 주소 따옴. train에서.

label_encoder=LabelEncoder()
label_encoder.fit(df1['Address'])
df3['Address'] = df3['Address'].map(lambda s: '<unknown>' if s not in label_encoder.classes_ else s)
label_encoder.classes_ = np.append(label_encoder.classes_, '<unknown>')
df1['Address']=label_encoder.transform(df1['Address'])
df3['Address']=label_encoder.transform(df3['Address'])
"""

clf = sktree.DecisionTreeClassifier(max_depth=3,random_state=0)
clf.fit(df1.drop('Resolution',1),train_target)
array_prob=clf.predict_proba(df3)


df_prob=pd.DataFrame(data=array_prob, columns=crime_category)
df_out=test_id.join(df_prob)
df_out['id']=df_out['id'].astype(float)

df_out.to_csv('/Users/leeseungjoon/Desktop/2018-2/BigData/HW3/Crime_data/insane.csv',index=False)

#   /  ST


"""
array를 먼저 인덱스를 주면어 df로 바꾼 다음에, 합치자

"""


"""
위의 세 줄로, 트리를 생성하고, fitting하고, 각각에 대한 확률 계산
"""

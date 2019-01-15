#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 19:46:12 2019

@author: leeseungjoon
"""

"""
------train------

column: PassengerId      has percent of NaN value: 0.00%
column:   Survived       has percent of NaN value: 0.00%
column:     Pclass       has percent of NaN value: 0.00%
column:       Name       has percent of NaN value: 0.00%
column:        Sex       has percent of NaN value: 0.00%
column:        Age       has percent of NaN value: 19.87%
column:      SibSp       has percent of NaN value: 0.00%
column:      Parch       has percent of NaN value: 0.00%
column:     Ticket       has percent of NaN value: 0.00%
column:       Fare       has percent of NaN value: 0.00%
column:      Cabin       has percent of NaN value: 77.10%
column:   Embarked       has percent of NaN value: 0.22%

------test-------
column: PassengerId      has percent of NaN value: 0.00%
column:     Pclass       has percent of NaN value: 0.00%
column:       Name       has percent of NaN value: 0.00%
column:        Sex       has percent of NaN value: 0.00%
column:        Age       has percent of NaN value: 20.57%
column:      SibSp       has percent of NaN value: 0.00%
column:      Parch       has percent of NaN value: 0.00%
column:     Ticket       has percent of NaN value: 0.00%
column:       Fare       has percent of NaN value: 0.24%
column:      Cabin       has percent of NaN value: 78.23%
column:   Embarked       has percent of NaN value: 0.00%

Pclass
Sex
Age
Family
Embarked

"""
def category_age(x):
    if x < 10:
        return 0
    elif x < 20:
        return 1
    elif x < 30:
        return 2
    elif x < 40:
        return 3
    elif x < 50:
        return 4
    elif x < 60:
        return 5
    elif x < 70:
        return 6
    else:
        return 7    
    


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from pandas import Series
from xgboost import XGBClassifier


train_data=pd.read_csv('/Users/leeseungjoon/Desktop/Kaggle/KaKR_Titanic/train.csv',sep= ',')
test_data=pd.read_csv('/Users/leeseungjoon/Desktop/Kaggle/KaKR_Titanic/test.csv',sep= ',')

test_id = test_data['PassengerId']
df_id=pd.DataFrame(test_id)

train_y = train_data.Survived

train_data['Family']=train_data['SibSp']+train_data['Parch']+1
test_data['Family']=test_data['SibSp']+test_data['Parch']+1

test_data.loc[test_data['Fare'].isnull()==True, 'Fare']=test_data['Fare'].mean()

train_data['Fare'] = train_data['Fare'].map(lambda i: np.log(i) if i>0 else 0)
test_data['Fare']=test_data['Fare'].map(lambda i:np.log(i) if i>0 else 0)

train_data['Initial']= train_data['Name'].str.extract('([A-Za-z]+)\.')
test_data['Initial']=test_data['Name'].str.extract('([A-Za-z]+)\.')
train_data['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],
                        ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr', 'Miss'],inplace=True)
test_data['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],
                        ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr', 'Miss'],inplace=True)
train_data.loc[(train_data['Initial']=='Mr') & (train_data['Sex']=='female'),'Initial']='Mrs'

train_data.loc[(train_data.Age.isnull())&(train_data.Initial=='Mr'),'Age'] = 33
train_data.loc[(train_data.Age.isnull())&(train_data.Initial=='Mrs'),'Age'] = 36
train_data.loc[(train_data.Age.isnull())&(train_data.Initial=='Master'),'Age'] = 5
train_data.loc[(train_data.Age.isnull())&(train_data.Initial=='Miss'),'Age'] = 22
train_data.loc[(train_data.Age.isnull())&(train_data.Initial=='Other'),'Age'] = 46

test_data.loc[(test_data.Age.isnull())&(test_data.Initial=='Mr'),'Age'] = 33
test_data.loc[(test_data.Age.isnull())&(test_data.Initial=='Mrs'),'Age'] = 36
test_data.loc[(test_data.Age.isnull())&(test_data.Initial=='Master'),'Age'] = 5
test_data.loc[(test_data.Age.isnull())&(test_data.Initial=='Miss'),'Age'] = 22
test_data.loc[(test_data.Age.isnull())&(test_data.Initial=='Other'),'Age'] = 46

train_data['Embarked'].fillna('S', inplace=True)

df_train=train_data.drop(['PassengerId', 'Survived', 'SibSp','Parch','Cabin','Ticket','Name'],axis=1)
df_test=test_data.drop(['PassengerId', 'SibSp','Parch','Cabin','Ticket','Name'],axis=1)

df_train['Sex_pclass'] = df_train['Sex'].astype(str) + df_train['Pclass'].astype(str)
df_test['Sex_pclass'] = df_test['Sex'].astype(str) + df_test['Pclass'].astype(str)

df_train['ageCat'] = df_train['Age'].apply(category_age)
df_test['ageCat']=df_test['Age'].apply(category_age)

df_train['SexAgecat']= df_train['Sex'].astype(str) + df_train['ageCat'].astype(str)
df_test['SexAgecat']= df_test['Sex'].astype(str) + df_test['ageCat'].astype(str)
df_test.loc[df_test['SexAgecat']=='female7', 'SexAgecat']='female6'

df_train['Family']= df_train['Family'].map({4:0, 3:1,2:1, 7:2,1:2, 5:3,6:3, 8:4,11:4})
df_test['Family']= df_test['Family'].map({4:0, 3:1,2:1, 7:2,1:2, 5:3,6:3, 8:4,11:4})

train_Age=pd.DataFrame(df_train['Age'])
train_Fare=pd.DataFrame(df_train['Fare'])
df_train= df_train.drop(['Age', 'Fare','ageCat','Sex','Embarked'],axis=1)

test_Age=pd.DataFrame(df_test['Age'])
test_Fare=pd.DataFrame(df_test['Fare'])
df_test= df_test.drop(['Age', 'Fare','ageCat','Sex','Embarked'],axis=1)

for col in df_train.columns:
    le = preprocessing.LabelEncoder()
    le=le.fit(df_train[col])
    df_train[col]=le.transform(df_train[col])
    df_test[col]=le.transform(df_test[col])

df_train['Age']=train_Age
df_train['Fare']=train_Fare
df_test['Age']=test_Age
df_test['Fare']=test_Fare

X_train_sub, X_validation_sub, y_train_sub, y_validation_sub = model_selection.train_test_split(df_train, train_y, random_state=0,test_size=0.3)
learning_rates = [0.05, 0.1, 0.15, 0.25,0.5, 0.75, 1, 1.25, 1.5,2, 2.5,3]
for learning_rate in learning_rates:
    clf = XGBClassifier(max_depth=4, learning_rate=learning_rate, n_estimator=200)
    clf.fit(X_train_sub, y_train_sub)
    y_pred = clf.predict(X_validation_sub)
    print("Learning rate: ", learning_rate)
    print(accuracy_score(y_validation_sub, y_pred))


"""
XGBoost로 만든 거임
clf=XGBClassifier()
clf.fit(X_train_sub, y_train_sub)
y_pred = clf.predict(X_validation_sub)
print(accuracy_score(y_validation_sub, y_pred))
0.8470149253731343

clf = XGBClassifier(max_depth=2, learning_rate=learning_rate, n_estimator=100)
Learning rate:  0.25
0.8582089552238806

"""

clf.fit(df_train,train_y)
y_pred = clf.predict(df_test)
y_col=pd.DataFrame(y_pred)
df_id['Survived']=y_col
df_id.to_csv('/Users/leeseungjoon/Desktop/Kaggle/KaKR_Titanic/XGB_0.25.csv',index=False)


"""
colormap=plt.cm.RdBu
plt.figure(figsize=(14, 12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(df_train.astype(float).corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True, annot_kws={"size":16})

sns.heatmap(heatmap_data.astype(float).corr(), linewidths=0.1, vmax=1.0,
           square=True, cmap=colormap, linecolor='white', annot=True, annot_kws={"size": 16})
"""

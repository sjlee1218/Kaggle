#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 16:43:38 2018

@author: leeseungjoon
"""
import pandas as pd
from sklearn import model_selection
from sklearn.ensemble import GradientBoostingClassifier
import seaborn as sns


train_data=pd.read_csv('/Users/leeseungjoon/Desktop/2018-2/BigData/HW4/Binary/bank_train.csv',sep= ',')
test_data=pd.read_csv('/Users/leeseungjoon/Desktop/2018-2/BigData/HW4/Binary/bank_test.csv',sep= ',')

train_x=train_data.loc[:,'age':'poutcome']
df2=train_x.copy()
train_y=train_data['y']

test_x=test_data.loc[:,'age':'poutcome']
df1=test_x.copy()
test_id=test_data['id']
df_id=pd.DataFrame(test_id)

"""
bank client data
"""

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

#df.loc[df['sum'] != 0, 'sum'] = 1 이거 필요할까??

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


'''
balance에 대한 프리프로세스 필요함

transformer = RobustScaler().fit()
'''

df2.loc[df2['balance']>=35000,'balance']=35000
#df2['balance']=preprocessing.robust_scale(df2['balance'])
#scaler = preprocessing.MinMaxScaler()
#df2['balance'] = scaler.fit_transform(df2['balance'])

df1.loc[df1['balance']>=30000, 'balance']=30000
#df1['balance']=scaler.transform(df1['balance'])

"""
related with the last contact of the current campaign:
    
contact
day    -- 아예 빼는 게 좋아 보이기도 함
month
duration
"""

df2.loc[df2['duration']>=2500, 'duration']=2500
#df2['duration']=preprocessing.robust_scale(df2['duration'])
#df2['duration']=scaler.fit_transform(df2['duration'])

df1.loc[df1['duration']>=2000, 'duration']=2000
#df1['duration']=scaler.transform(df1['duration'])

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



"""
# other attributes:

campaign
pdays
previous
poutcome
"""

df2.loc[df2['campaign']>=30, 'campaign']=30

#df2['pdays']=preprocessing.robust_scale(df2['pdays'])

df2.loc[df2['previous']>=80, 'previous']=80

df2['poutcome']=df2['poutcome'].replace('success',0)
df2['poutcome']=df2['poutcome'].replace('other',1)
df2['poutcome']=df2['poutcome'].replace('failure',2)
df2['poutcome']=df2['poutcome'].replace('unknown',3)

df1.loc[df1['campaign']>=30, 'campaign']=30

df1.loc[df1['previous']>=20, 'previous']=20

df1['poutcome']=df1['poutcome'].replace('success',0)
df1['poutcome']=df1['poutcome'].replace('other',1)
df1['poutcome']=df1['poutcome'].replace('failure',2)
df1['poutcome']=df1['poutcome'].replace('unknown',3)


df2=df2.drop('day',1)
df1=df1.drop('day',1)


"""
X_train_sub, X_validation_sub, y_train_sub, y_validation_sub = model_selection.train_test_split(df2, train_y, random_state=0,test_size=0.3)

learning_rates = [0.05, 0.1, 0.25, 0.5, 0.75, 1]
for learning_rate in learning_rates:
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate = learning_rate, min_samples_leaf=20, max_depth = 5, random_state = 0)
    clf.fit(X_train_sub, y_train_sub)
    print("Learning rate: ", learning_rate)
    print("Accuracy score (training): {0:.3f}".format(clf.score(X_train_sub, y_train_sub)))
    print("Accuracy score (validation): {0:.3f}".format(clf.score(X_validation_sub, y_validation_sub)))
    print()
    
X_train_sub, X_validation_sub, y_train_sub, y_validation_sub = train_test_split(df2, train_y, random_state=0)
for learning_rate in learning_rates:
    model = AdaBoostClassifier(sktree.DecisionTreeClassifier(max_depth=2,min_samples_leaf=5) , algorithm="SAMME" , n_estimators=500,learning_rate=learning_rate)
    model = model.fit(X_train_sub, X_validation_sub)
    print ('learning rate is {:0.2f}'.format(learning_rate) )
    print ('Accuracy = {:0.2f}%'.format(100.0 * accuracy_score(train_y, model.predict(df2))))
    print ( 'On average, this model is correct {:0.2f}% (+/- {:0.2f}%) of the time.'.format(scores.mean() * 100.0, scores.std() * 2 * 100.0))
    print()
    
    
X_train_sub, X_validation_sub, y_train_sub, y_validation_sub = train_test_split(df2, train_y, random_state=0)
for learning_rate in learning_rates:
    model = RandomForestClassifier(n_estimators=500, max_depth=5,min_samples_leaf=10,random_state=0,warm_start=True)
    model = model.fit(X_train_sub, y_train_sub)
    print("Accuracy score (training): {0:.3f}".format(model.score(X_train_sub, y_train_sub)))
    print("Accuracy score (validation): {0:.3f}".format(model.score(X_validation_sub, y_validation_sub)))
    print("feautre importances")
    print(model.feature_importances_)
    print()


"""

#print(clf.feature_importances_)
#sample_weight=['age':1,'job':1,'marital':1,'education':1,'default':1,'balance':1,'housing':1,'loan':1,'contact':1,'month':2,'duration':2,'campaign':1,'pdays':1,'previous':1,'poutcome':2]

for i in range(0, 5): 
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate = 0.1, min_samples_leaf=20, max_depth = 5, random_state = 0,warm_start=True)

clf.fit(df2, train_y)

y_pred=clf.predict(df1)
y_col=pd.DataFrame(y_pred)
df_id['y']=y_col
df_id.to_csv('/Users/leeseungjoon/Desktop/2018-2/BigData/HW4/Binary/LIFE_without_day.csv',index=False)

print("feautre importances")
print(clf.feature_importances_)


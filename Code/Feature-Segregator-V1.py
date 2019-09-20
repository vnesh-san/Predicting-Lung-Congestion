# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 10:30:01 2017

@author: VIGNESH
"""
import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn import model_selection,cross_validation
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

##Read data and Preprocess

org_dataset_X = pd.read_csv('norm_data.csv',header=0)
org_dataset_y = pd.read_csv('label.csv',header=0)
data = pd.read_csv('data_set.csv',header=0)
data_anomaly = data.loc[data['label'] == 0].drop(['label'],1)
#%%
#org_dataset.replace('M',1,inplace=True)
#org_dataset.replace('B',0,inplace=True)
#org_dataset.drop(['id'],1,inplace=True)

##Shuffling data and printing 

#org_dataset_shuffled_features = org_dataset_X.sample(frac=1)

#train-split sample

org_dataset_features = org_dataset_X.values
#org_dataset_features = org_dataset_features.astype(int)

org_dataset_label = np.ravel(org_dataset_y.values)
#org_dataset_label = org_dataset_label.astype(int)

#input features

org_dataset_shuffled_X = np.array(org_dataset_features)

#output class

org_dataset_shuffled_y = np.ravel(np.array(org_dataset_label))
X_train, X_test, y_train, y_test = cross_validation. train_test_split(org_dataset_shuffled_X,org_dataset_shuffled_y,test_size = 0.30)

# Sample classifier in OC-SVM to test the data 

classifier = OneClassSVM(gamma=0.01,nu=0.3)

classifier.fit(data_anomaly)

inlier_and_outlier = classifier.predict(org_dataset_shuffled_X)

#%%
#splitting normal and anomaly data

i=0
count=0
count_2=0
normal_X=[]
normal_y=[]
anomaly_X=[]
anomaly_y=[]

for i in list(range(0,len(inlier_and_outlier))):
            if inlier_and_outlier[i]==1:
                count = count + 1
                normal_X.append(org_dataset_features[i])
                normal_y.append(org_dataset_label[i])
            else:
                count_2+=1
                anomaly_X.append(org_dataset_features[i])
                anomaly_y.append(org_dataset_label[i])


print('Distribution of Normal and Anomaly')
print(count,count_2)
#print(anomaly_X,anomaly_y)

#normal_data = np.array(normal)
#anomaly_data = np.array(anomaly)

#split normal and anomaly into train and test

#normal_X = normal_data[:,2:]
#normal_y = normal_data[:,1]

#anomaly_X = anomaly_data[:,2:]
#anomaly_y = anomaly_data[:,1]

normal_X_train,normal_X_test,normal_y_train,normal_y_test = train_test_split(normal_X,normal_y,test_size = 0.30)

anomaly_X_train,anomaly_X_test,anomaly_y_train,anomaly_y_test = train_test_split(anomaly_X,anomaly_y,test_size = 0.30)

normal_y_train=np.ravel(normal_y_train)
normal_y_test=np.ravel(normal_y_test)

anomaly_y_train=np.ravel(anomaly_y_train)
anomaly_y_test=np.ravel(anomaly_y_test)

#normal data classification
print ('Normal')
#logisticRegression
normal_clf_1 = LogisticRegression()
normal_clf_1.fit(normal_X_train,normal_y_train)
acc_normal_1 = normal_clf_1.score(normal_X_test, normal_y_test)
print(acc_normal_1)

#DecisionTree
normal_clf_2 = DecisionTreeClassifier()
normal_clf_2.fit(normal_X_train,normal_y_train)
acc_normal_2 = normal_clf_2.score(normal_X_test, normal_y_test)
print(acc_normal_2)

#bag
kfold = model_selection.KFold(n_splits=10, random_state=7)
cart = DecisionTreeClassifier()
model = BaggingClassifier(base_estimator=cart, n_estimators=100, random_state=7)
results = model_selection.cross_val_score(model, normal_X, normal_y, cv=kfold)
print(results.mean())

#boost
kfold = model_selection.KFold(n_splits=10, random_state=7)
model = AdaBoostClassifier(n_estimators=30, random_state=7)
results = model_selection.cross_val_score(model, normal_X, normal_y, cv=kfold)
print(results.mean())



#anomaly data classification
print ('Anomaly')
#logisticRegression
anomaly_clf_1 = LogisticRegression()
anomaly_clf_1.fit(anomaly_X_train,anomaly_y_train)
acc_anomaly_1 = anomaly_clf_1.score(anomaly_X_test,anomaly_y_test)
print(acc_anomaly_1)

#DecisionTree
anomaly_clf_2 = DecisionTreeClassifier()
anomaly_clf_2.fit(anomaly_X_train,anomaly_y_train)
acc_anomaly_2 = anomaly_clf_2.score(anomaly_X_test,anomaly_y_test)
print(acc_anomaly_2)

#bag
kfold = model_selection.KFold(n_splits=10, random_state=7)
cart = DecisionTreeClassifier()
model = BaggingClassifier(base_estimator=cart, n_estimators=100, random_state=7)
results = model_selection.cross_val_score(model, anomaly_X, anomaly_y, cv=kfold)
print(results.mean())

#boost
kfold = model_selection.KFold(n_splits=10, random_state=7)
model = AdaBoostClassifier(n_estimators=30, random_state=7)
results = model_selection.cross_val_score(model, anomaly_X, anomaly_y, cv=kfold)
print(results.mean())





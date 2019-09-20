# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 05:59:37 2017

@author: VIGNESH
"""

import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM, LinearSVC
from sklearn import cross_validation
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier


#READ AND PREPROCESS

org_dataset = pd.read_csv('data_set.csv')
org_dataset_features = org_dataset.drop(['label'],1)
org_dataset_label = org_dataset['label']
data_0 = org_dataset.loc[org_dataset['label'] == 0]
data_1 = org_dataset.loc[org_dataset['label'] == 1]
data_2 = org_dataset.loc[org_dataset['label'] == 2]
majority_class = data_2.append(data_0) 

org_dataset_X = np.array(org_dataset_features)
org_dataset_y = np.ravel(np.array(org_dataset_label))

X_train, X_test, y_train, y_test = cross_validation.train_test_split(org_dataset_X,org_dataset_y,test_size = 0.30)

#OC SVM

clf = OneClassSVM()
clf.fit(majority_class)
preds=clf.predict(org_dataset)
org = np.array(org_dataset)

#subset1 and larger SPLIT
i=0
count_1=0
count_2=0
subset1_X=[]
subset1_y=[]
larger_X=[]
larger_y=[]
larger = []
for i in list(range(0,len(preds))):
            if preds[i]==-1:
                count_1 = count_1 + 1
                subset1_X.append(org_dataset_X[i])
                subset1_y.append(org_dataset_y[i])
            else:
                count_2+=1
                larger_X.append(org_dataset_X[i])
                larger_y.append(org_dataset_y[i])
                larger.append(org[i])

#subset2 and subset3 SPLIT

clf.fit(data_1)
preds_2=clf.predict(larger)

j=0
count_3=0
count_4=0
subset2_X=[]
subset2_y=[]
subset3_X=[]
subset3_y=[]

for j in list(range(0,len(preds_2))):
            if preds_2[j]==-1:
                count_3 = count_3 + 1
                subset2_X.append(larger_X[j])
                subset2_y.append(larger_y[j])
            else:
                count_4+=1
                subset3_X.append(larger_X[j])
                subset3_y.append(larger_y[j])
               
print('Distribution of subsets')
print(count_1,count_3,count_4)                
subset1_X_train,subset1_X_test,subset1_y_train,subset1_y_test = cross_validation.train_test_split(subset1_X,subset1_y,test_size = 0.20)

subset2_X_train,subset2_X_test,subset2_y_train,subset2_y_test = cross_validation.train_test_split(subset3_X,subset3_y,test_size = 0.20)

subset3_X_train,subset3_X_test,subset3_y_train,subset3_y_test = cross_validation.train_test_split(subset2_X,subset2_y,test_size = 0.20)

#SUBSET 1
print ('SUBSET 1')

#DecisionTree
clf_2 = DecisionTreeClassifier()
clf_2.fit(subset1_X_train,subset1_y_train)
acc_sub1 = clf_2.score(subset1_X_test, subset1_y_test)
print(acc_sub1)


#SUBSET 2
print ('SUBSET 2')

#DecisionTree

clf_2.fit(subset2_X_train,subset2_y_train)
acc_sub2 = clf_2.score(subset2_X_test,subset2_y_test)
print(acc_sub2)

#SUBSET 3 
print ('SUBSET 3')

#DecisionTree
clf_2.fit(subset3_X_train,subset3_y_train)
acc_sub3 = clf_2.score(subset3_X_test, subset3_y_test)
print(acc_sub3)

print('Accuracy of Allocation')
acc = (acc_sub1 +  acc_sub2 + acc_sub3)/3
print(acc)

#DIRECT
print('Direct Method')
acc_total = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X_train,y_train).score(X_test,y_test)
print(acc_total)

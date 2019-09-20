# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 05:59:37 2017

@author: VIGNESH  
"""


import numpy as np
np.set_printoptions(threshold=np.nan)
import pandas as pd
from sklearn.svm import OneClassSVM, LinearSVC
from sklearn import cross_validation
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier
from scipy.spatial import distance

#READ AND PREPROCESS

org_dataset = pd.read_csv('data_set.csv')
org_dataset_features = org_dataset#.drop(['label'],1)
org_dataset_label = org_dataset['label']
data_0 = org_dataset.loc[org_dataset['label'] == 0]
data_1 = org_dataset.loc[org_dataset['label'] == 1]
data_2 = org_dataset.loc[org_dataset['label'] == 2]

whole_data = list(np.concatenate([org_dataset]))

data=[[] for _ in range(0,3)]
data[0]=np.concatenate([data_0])
data[1]=np.concatenate([data_1])
data[2]=np.concatenate([data_2])

def seperate_features(input_list,feature_id):
	index = feature_id
	sep_data = [[] for _ in range(len(input_list))]
	for i in range(0,len(input_list)):
		for j in range(0,16):
			sep_data[i].append(input_list[i][j])
	sorted_data = sorted(sep_data,key=lambda x: (x[index]))
	return sorted_data

outliers = [] 
count = [[] for _ in range(0,len(whole_data))]
for x in range(0,len(whole_data)):
    count[x]=0
    
for yin in list([0,1,2]):
    for feature_id in range(0,16):
        before_dropping=seperate_features(data[yin],feature_id)
        sum_of_diff = 0 
        for x in range(0,len(before_dropping)-1):           
            xplus = x + 1
            difference = distance.euclidean(before_dropping[xplus][feature_id],before_dropping[x][feature_id])
            sum_of_diff = sum_of_diff + difference
            before_dropping[x].append(difference)
        before_dropping[len(before_dropping)-1].append(0)
        mean_of_diff = sum_of_diff/len(before_dropping)
         
        for x in range(0,len(before_dropping)):
            if(before_dropping[x][16]>2*mean_of_diff):
                count[x]+=1
                
    for x in range(0,len(before_dropping)-1):
        if (count[x]>=4):
            if ((x < len(before_dropping)/2) and (x ==len(before_dropping))):
                before_dropping[x].pop(16)
                outliers.append(before_dropping[x])
                del whole_data[x]
            else:
                before_dropping[x+1].pop(16)
                outliers.append(before_dropping[x+1])
                del whole_data[x+1]
    print(len(outliers))

Outliers = set(tuple(element) for element in outliers) #eliminating duplicates in outliers 431*16
Inliers = set(tuple(element) for element in whole_data) #entire set of 1030*16

print('\n')
print(len(Outliers))
print(len(Inliers))





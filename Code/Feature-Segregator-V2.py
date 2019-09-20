# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 05:59:37 2017

@author: VIGNESH -- Mayiru.. 
"""


import pandas as pd
from scipy.spatial import distance
import numpy as np
np.set_printoptions(threshold=np.nan)

#READ AND PREPROCESS

org_dataset = pd.read_csv('data_set.csv')
org_dataset_features = org_dataset.drop(['label'],1)
org_dataset_label = org_dataset['label']
data_0 = org_dataset.loc[org_dataset['label'] == 0]
data_1 = org_dataset.loc[org_dataset['label'] == 1]
data_2 = org_dataset.loc[org_dataset['label'] == 2]


whole_data = np.concatenate([org_dataset_features])

def seperate_features(input_list,feature_id):
	index = feature_id
	sep_data = [[] for _ in range(len(input_list))]
	for i in range(0,len(input_list)):
		for j in range(0,16):
			sep_data[i].append(input_list[i][j])
	sorted_data = sorted(sep_data,key=lambda x: (x[index]))
	return sorted_data

outliers = [] 
dist = [[] for _ in range(len(whole_data))]
count = [[] for _ in range(len(whole_data))]

for feature_id in range(0,16):
	before_dropping=seperate_features(majority_class,feature_id)
	for x in range(0,len(before_dropping)-1):
		xplus = x + 1
		difference = distance.euclidean(before_dropping[xplus][feature_id],before_dropping[x][feature_id])
		dist[x].append(difference)
	dist[len(dist)].append(0)
	mean_of_diff = np.mean(dist)
	for x in range(0,len(dist)):
            if(dist[x] > 2*mean_of_diff):
                if (x==0):
                    count[x]+=1
                elif (x==len(dist)):
                    count[x+1]+=1
                else:
                    if (x < len(dist)/2):
                        count[x]+=1
                    else:
                        count[x+1]+=1
                        
for x in range(0,len(count)):
    if (count[x]>4):
        before_dropping[x].pop
        outliers.append(before_dropping[x])


final_outliers_tuples = list(set(tuple(element) for element in outliers)) #eliminating duplicates in outliers 431*16
whole_data_tuples = set(tuple(element) for element in whole_data) #entire set of 1030*16
final_inlier_tuples = list(set(whole_data_tuples) - set(final_outliers_tuples)) #difference --> 599*16

#proceed by transferring list to csv and so on. 

















				

	
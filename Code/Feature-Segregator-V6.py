# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 05:59:37 2017

@author: VIGNESH
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

data = [[] for _ in range(0,3)]
whole_data = np.concatenate([org_dataset])
Org_Data = whole_data
data[0] = np.concatenate([data_0])
data[1] = np.concatenate([data_1])
data[2] = np.concatenate([data_2])

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
for x in range(0,len(count)):
    count[x] = 0
    
for yin in range(0,2):        
    for feature_id in range(0,16):
    	before_dropping=seperate_features(data[yin],feature_id)
    	for x in range(0,len(before_dropping)-1):
    		xplus = x + 1
    		difference = distance.euclidean(before_dropping[xplus][feature_id],before_dropping[x][feature_id])
    		dist[x].append(difference)
    	dist[len(dist)].append(0)
    	mean_of_diff = np.mean(dist)
    	for x in range(0,len(dist)-1):
                if(dist[x] > 2*mean_of_diff):
                    if (x < len(dist)/2):
                        count[x]+=1
                    else:
                        count[x+1]+=1
                            
for x in range(0,len(count)):
    if (count[x]>4):
        if (x < len(dist)/2):
            before_dropping[x].pop
            outliers.append(before_dropping[x])
            del whole_data[x]
        else:
            before_dropping[x+1].pop
            outliers.append(before_dropping[x+1])
            del whole_data[x+1]
                    
        

Outliers = list(set(tuple(element) for element in outliers)) #eliminating duplicates in outliers 431*16
Org_Data = list(set(tuple(element) for element in Org_Data)) #entire set of 1030*16
Inliers = list(set(tuple(element) for element in whole_data) #difference --> 599*16

#proceed by transferring list to csv and so on. 

















				

	                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
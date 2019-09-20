from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from scipy.spatial import distance

#READ AND PREPROCESS

org_dataset = pd.read_csv('data_set.csv')
org_dataset_features = org_dataset.drop(['label'],1)
org_dataset_label = org_dataset['label']
data_0 = np.array(org_dataset.loc[org_dataset['label'] == 0].drop(['label'], 1))
data_1 = np.array(org_dataset.loc[org_dataset['label'] == 1].drop(['label'], 1))
data_2 = np.array(org_dataset.loc[org_dataset['label'] == 2].drop(['label'], 1))
   
def seperate_features(input_list,feature_id):
	index = feature_id
	print('Index:',index)
	sep_data = [[] for _ in range(len(input_list))]
	for i in range(0,len(input_list)):
		sep_data[i].append(input_list[i][index])
		sep_data[i].append(i)
	#print(sep_data)
	sorted_data = sorted(sep_data,key=lambda x: (x[0]))
	return sorted_data

sorted_data_0 = sorted(data_0,key=lambda x: (x[0]))
for feature_id in range(0,len(data_0[0])):
	before_dropping=seperate_features(data_0,feature_id)
	difference_list = [[] for _ in range(len(before_dropping)-1)]
	sum_of_diff = 0 
	for p in range(0,len(before_dropping)-1):
		q = p + 1
		difference = distance.euclidean(before_dropping[q][0],before_dropping[p][0])
		sum_of_diff = sum_of_diff + difference
		difference_list[p].append(difference)
		difference_list[p].append(p)
		difference_list[p].append(q)
	mean_of_diff = sum_of_diff/len(before_dropping)
	outlier_list = [] 
	for i in range(len(difference_list)):
		if(difference_list[i][0]>2*mean_of_diff):
			outlier = difference_list[i].pop(1)
			outlier_list.append(outlier)

print(outlier_list)

        for i in range(len(data_0)):
                if(

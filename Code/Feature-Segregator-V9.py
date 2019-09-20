import pandas as pd
from scipy.spatial import distance
import numpy as np
from sklearn.svm import OneClassSVM,LinearSVC
from sklearn import cross_validation
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier
from keras.models import Sequential
from keras.layers import Dense

# fix random seed for reproducibility
np.random.seed(7)
np.set_printoptions(threshold=np.nan)

#READ AND PREPROCESS

org_dataset = pd.DataFrame(pd.read_csv('data_set.csv'))
data_0 = org_dataset.loc[org_dataset['label'] == 0]
data_1 = org_dataset.loc[org_dataset['label'] == 1]
data_2 = org_dataset.loc[org_dataset['label'] == 2]

data = [[] for _ in range(0,3)]
whole_data = list(np.concatenate([org_dataset]))

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
count = [[] for _ in range(len(whole_data)+1)]
for x in range(0,len(count)):
    count[x] = 0
    
for yin in range(0,2):        
    for feature_id in range(0,16):        
        before_dropping=seperate_features(data[yin],feature_id)
        sum_of_diff = 0
        for x in range(0,len(before_dropping)-1):
            xplus = x + 1
            difference = distance.euclidean(before_dropping[xplus][feature_id],before_dropping[x][feature_id])
            sum_of_diff+= difference
            before_dropping[x].append(difference)
        before_dropping[len(before_dropping)-1].append(0)
        mean_of_diff = sum_of_diff/len(before_dropping)
        for x in range(0,len(before_dropping)-1):
            if(before_dropping[x][16] > 2*mean_of_diff):
                if (x < len(before_dropping)/2):
                    count[x]+=1
                else:
                    count[x+1]+=1
                            
    for x in range(0,len(before_dropping)-1):
        if (count[x]>4):
            if (x < len(before_dropping)/2):
                before_dropping[x].pop(16)
                outliers.append(before_dropping[x])
                del whole_data[x]
            else:
                before_dropping[x].pop(16)
                outliers.append(before_dropping[x+1])
                del whole_data[x+1]
        
            
Org_Data = pd.DataFrame(list(set(tuple(element) for element in list(np.concatenate([org_dataset])))))
Inliers = pd.DataFrame(list(set(tuple(element) for element in whole_data)))
Outliers = pd.concat([Org_Data, Inliers]).drop_duplicates(keep=False)

print('',len(Outliers),'\n',len(Inliers),'\n',len(Org_Data))


dataset = [[] for _ in range(0,2)]
dataset[0] = Inliers
dataset[1] = Outliers
acc = [[] for _ in range(0,2)]
acc_sub1 = [[] for _ in range(0,2)]
acc_sub2 = [[] for _ in range(0,2)]
acc_sub3 = [[] for _ in range(0,2)]

yang = 1
for yang in range(0,2):
        
    org_dataset = dataset[yang]
    org_dataset_features = org_dataset.drop([len(org_dataset.columns)-1],1)
    org_dataset_label = org_dataset[len(org_dataset.columns)-1]
    data_0 = org_dataset.loc[org_dataset[len(org_dataset.columns)-1] == 0]
    data_1 = org_dataset.loc[org_dataset[len(org_dataset.columns)-1] == 1]
    data_2 = org_dataset.loc[org_dataset[len(org_dataset.columns)-1] == 2]
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
    
    subset1_X_train,subset1_X_test,subset1_y_train,subset1_y_test = train_test_split(subset1_X,subset1_y,test_size = 0.30)
        
    subset2_X_train,subset2_X_test,subset2_y_train,subset2_y_test = train_test_split(subset3_X,subset3_y,test_size = 0.30)
        
    subset3_X_train,subset3_X_test,subset3_y_train,subset3_y_test = train_test_split(subset2_X,subset2_y,test_size = 0.30)
        
        #ANN
        # create model
    model = Sequential()
    model.add(Dense(24, input_dim=16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(48, activation='relu'))
    model.add(Dense(72, activation='relu'))
    model.add(Dense(3, activation='sigmoid'))
    
    #subset1_X = np.array(subset1_X)
    #subset1_y = np.array(subset1_y)
    #subset2_X = np.array(subset2_X)
    #subset2_y = np.array(subset2_y)
    subset3_X = np.array(subset3_X)
    subset3_y = np.array(subset3_y)
    
    # Compile model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
         #DecisionTree
    clf_2 = DecisionTreeClassifier()
    
    
    #SUBSET 1
    clf_2.fit(subset1_X_train,subset1_y_train)
    acc_sub1[yang] = clf_2.score(subset1_X_test, subset1_y_test)
    
        
        
        #SUBSET 2
    clf_2.fit(subset2_X_train,subset2_y_train)
    acc_sub2[yang] = clf_2.score(subset2_X_test,subset2_y_test)
   
        
        #SUBSET 3 
    model.fit(subset3_X,subset3_y, epochs=10, batch_size=10)
    acc_sub3[yang] = model.evaluate(subset3_X, subset3_y)
        
    acc[yang] = (acc_sub1[yang]+  acc_sub2[yang] + acc_sub3[yang][1])/3
    
print ('SUBSET 1')
print(acc_sub1)
print ('SUBSET 2')
print(acc_sub2)
print ('SUBSET 3')
print(acc_sub3)
       
print('Accuracy of Allocation')
print((acc_sub1[0] +  acc_sub2[0] + acc_sub3[0][1] + acc_sub1[1]+  acc_sub2[1] + acc_sub3[1][1])/6)

#DIRECT
print ('Direct Method')
acc_total = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X_train,y_train).score(X_test,y_test)
print(acc_total)

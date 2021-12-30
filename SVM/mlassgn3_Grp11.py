
from math import exp
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt
import datetime


data = pd.read_csv("datatraining.txt")
data2 = pd.read_csv("datatest.txt")
data3 = pd.read_csv("datatest2.txt")
data = data.append(data2,ignore_index = True)
data = data.append(data3,ignore_index = True)

features = ['date','Temperature','Humidity','Light','CO2','HumidityRatio','time']

def prep_data(data):
    data['date'] = pd.to_datetime(data['date'])
    #data = data.drop(columns =['date'])

    data['time'] = [tm.hour+tm.minute/60+tm.second/3600 for tm in data['date']]
    data['date'] = [tm.day for tm in data['date']]
    for col in features:
        data[col] = (data[col]-data[col].mean())/data[col].std()
    return data

data = prep_data(data)


#split into train  validation  test in 7:1:2 ratio
train = data.sample(frac=0.7)

data = data.drop(train.index)

val_data = data.sample(frac=1/3)

test_data = data.drop(val_data.index)


#----------DO PCA----------------------

pca = PCA(n_components=2)
pca.fit(train[features])

#Determine reduced dimensional dataset
X_train = np.dot(pca.components_,train[features].to_numpy().T).T
y_train = train['Occupancy'].to_numpy()
X_val = np.dot(pca.components_,val_data[features].to_numpy().T).T
y_val = val_data['Occupancy'].to_numpy()
X_test = np.dot(pca.components_,test_data[features].to_numpy().T).T
y_test = test_data['Occupancy'].to_numpy()

#Reduced dimension Data Visualisation of train split

fig = plt.figure(figsize=(10,10))
colors = ['r','b']#r for occupancy = 0 b for 1
i = 0
ax = fig.add_subplot()
temp = pd.DataFrame(X_train,columns=['0','1'])
temp['op'] = y_train
#print(temp)
for color in colors:
    ind = temp['op']== i    
    #print(ind)
    ax.scatter(temp.loc[ind,'0'],temp.loc[ind,'1'],c=color)
    i+=1
ax.legend(['zero','one'])
ax.grid()
plt.show()

#train model on PCA
kernals = ['linear','poly','rbf','sigmoid']
deg = [2,3,4]
accuracy = {}
for kernal in kernals:
   
    if(kernal=='poly'):
        for deg_ in deg:
            model = SVC(kernel = kernal,degree=deg_)
            model.fit(X_train,y_train)
            y_pred = model.predict(X_val)
            accuracy[(kernal,deg_)] = np.sum(y_pred==y_val)/len(y_val)
    else:

        model = SVC(kernel = kernal)
        model.fit(X_train,y_train)
        y_pred = model.predict(X_val)
        accuracy[kernal] = np.sum(y_pred==y_val)/len(y_val)


maxDeg = 0
accMaxKer = max(accuracy,key = accuracy.get)
print('Accuracy with different kernals on validation (PCA) :',accuracy)
print('Kernal taken for test data: ',accMaxKer)
if(type(accMaxKer) is tuple):
    maxDeg = accMaxKer[1]
    accMaxKer = accMaxKer[0]

#Now run that on test data
if(maxDeg == 0):
    model = SVC(kernel = accMaxKer)
else:
    model = SVC(kernel = accMaxKer,degree=maxDeg)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
Acc = np.sum(y_pred==y_test)/len(y_test)

print('Test Accuracy with PCA: ',Acc)

#Now using LDA
lda = LDA(n_components = 1)
lda.fit(train[features],train['Occupancy'])

#Get reduced data
X_train = lda.transform(train[features])
y_train = train['Occupancy'].to_numpy()
X_val = lda.transform(val_data[features])
y_val = val_data['Occupancy'].to_numpy()
X_test = lda.transform(test_data[features])
y_test = test_data['Occupancy'].to_numpy()


#Visualize reduced data

fig = plt.figure(figsize=(10,10))
colors = ['r','b']#r for occupancy = 0 b for 1
i = 0
ax = fig.add_subplot()
temp = pd.DataFrame(X_train,columns=['0'])# 1 dimension 
temp['op'] = y_train
#print(temp)
for color in colors:
    ind = temp['op']== i    
    
    ax.scatter(temp.loc[ind,'0'],[0]*len(temp.loc[ind,'0']),c=color)
    i+=1
ax.legend(['zero','one'])
ax.grid()
plt.show()
#train model on LDA
kernals = ['linear','poly','rbf','sigmoid']
deg = [2,3,4]
accuracy = {}
for kernal in kernals:
    if(kernal=='poly'):
        for deg_ in deg:
            model = SVC(kernel = kernal,degree=deg_)
            model.fit(X_train,y_train)
            y_pred = model.predict(X_val)
            accuracy[(kernal,deg_)] = np.sum(y_pred==y_val)/len(y_val)
    else:

        model = SVC(kernel = kernal)
        model.fit(X_train,y_train)
        y_pred = model.predict(X_val)
        accuracy[kernal] = np.sum(y_pred==y_val)/len(y_val)


maxDeg = 0
accMaxKer = max(accuracy,key = accuracy.get)
print('Accuracy with different kernals on validation (LDA) :',accuracy)
print('Kernal taken for test data: ',accMaxKer)
if(type(accMaxKer) is tuple):
    maxDeg = accMaxKer[1]
    accMaxKer = accMaxKer[0]

#Now run that on test data
if(maxDeg == 0):
    model = SVC(kernel = accMaxKer)
else:
    model = SVC(kernel = accMaxKer,degree=maxDeg)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
Acc = np.sum(y_pred==y_test)/len(y_test)

print('Test Accuracy with LDA: ',Acc)



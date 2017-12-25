## Import Library
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.linear_model
import pickle
np.random.seed(1)

## Read Data
train=pd.read_csv("new_train_set.csv")

## Plot Input
#plt.scatter(train.cpu_util,train.cpu_util,c=train.dev_status)
#plt.show()

## Input/Output Configuration
X=train.cpu_util
Y=train.dev_status
X= X.values.reshape(-1,1)
Y = Y.values.reshape(-1,1)

## Model Training
clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X,Y)
y_pred=clf.predict(159)

## Save the Model
pickle.dump(clf,open("clf","wb"))
print(y_pred[0])

## Plot the output prediction classes
#y_pred = y_pred.reshape(-1,1)
#plt.scatter(X,Y,c=y_pred )
#plt.show()
#print(X)

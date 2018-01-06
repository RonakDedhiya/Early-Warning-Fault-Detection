## Import Library
import pandas as pd
import pickle
from keras.models import load_model
import numpy as np
from flask import Flask,request,jsonify
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM,Dense
from math import sqrt
import tensorflow as tf
from random import randint
import sklearn.linear_model

app = Flask(__name__)

## Definitions

## Removing trend- Making Stationary
def difference(dataset, interval=1):
    diff = list()
    #diff.append(0)
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return pd.Series(diff)

## Frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df
## Inverse scailing for forecasted value
def inverse_scale(scaler,X,value):
    new_row = [x for x in X] + [value]
    array = np.array(new_row)
    array = array.reshape(1,len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0,-1]

## Invert diffferenced value - Get Orignial Value
def inverse_difference(history, yhat, interval=1):
	return yhat + history

## Calling Classifier
def predict_alert(data):
    ## Load Model
    clf=pickle.load(open("clf","rb"))
    ## Predict Class
    y_pred=clf.predict(data)
    if np.array(y_pred[0]) == 0 :
        return 1
    else:
        return 0

## Retraining the classifier for new threshold
def retrain(data):
    print("#"*33,"Retraining","#"*33)
    ## Data Preparation for Training
    data=pd.DataFrame(data,columns=["cpu_util"])
    temp_dataframe= pd.DataFrame(0, index=np.arange(len(data)), columns=["dev_status"])
    data=data.join(temp_dataframe)

    ## Augmenting training data for Off state Label
    for row in data.iterrows():
        if 0 == row[1][0] :
            row[1][0]=randint(trigger_value, int(trigger_value*1.4))
            row[1][1]=0
        elif row[1][0] >= trigger_value  :
            row[1][1]=0
        else:
            row[1][1]=1

    if zero_count < len(data)*0.5 :
        for i in range(int(len(data)*0.5) - zero_count):
            temp_dataframe=pd.DataFrame([[randint(trigger_value, int(trigger_value*1.4)), 0]], columns=["cpu_util","dev_status"])
            data = data.append(temp_dataframe, ignore_index=True)

    ## Data ready for retraining
    X=data.cpu_util
    Y=data.dev_status
    X= X.values.reshape(-1,1)
    Y = Y.values.reshape(-1,1)
    ## Model Trained for new data
    clf = sklearn.linear_model.LogisticRegressionCV()
    clf.fit(X,Y)
    ## Save the model
    pickle.dump(clf,open("clf","wb"))
    print("#"*33,"Retraining completed","#"*33)

    return 0

## Predict output
def predict_data(raw_data):
    global prev,trigger_count,trigger_value,zero_count
    ## Input Transformation
    arr=np.array([prev,raw_data])
    arr=arr.reshape(arr.shape[0],1)
    data=difference(arr,1)
    data=timeseries_to_supervised(data,1)
    data=scaler.transform(data)
    X= data[0,1]
    X=X.reshape(1,1,1)
    ## Predicting the next possible value
    yhat = lstm_model.predict(X,1)
    ## Inverse Transformation
    yhat=inverse_scale(scaler,X,yhat)
    yhat=inverse_difference(raw_data,yhat,1)
    print("Current Value is:",raw_data,end=' ')
    print("Naxt value could be ",yhat, end=' ')
    ## Calling Classifier to decide device on or off
    predict_alert(yhat)
    num_list.append(raw_data)
   ## Retraining Logic
    if raw_data == 0:
            if 0 != prev:
                trigger_count+=1
                if trigger_count == 1:
                    trigger_value=prev
                if trigger_value > prev:
                    trigger_value=prev
                if trigger_count >= trig:
                    retrain(num_list)
                    trigger_count=0
            zero_count+=1
    prev=raw_data

    return [predict_alert(yhat),yhat]

## Loading Model( LSTM Model, Classifier Model )
lstm_model = load_model('my_model.h5')
scaler=pickle.load(open("scaler","rb"))

## Some Initialization
raw_data=120
prev=raw_data
trigger_value=0
trigger_count=0
trig=5
zero_count=0
num_list=[]
#predict_data(raw_data)


@app.route('/predict',methods=['POST'])
def prediction():
    data = request.get_json()
    ## Calling Prediction - Result :- Early alert or no alert
    alert,result=predict_data(int(data[0]['data']))
    if alert == 1:
            return jsonify({"Current Value": str(data[0]['data']),
                            "Next Value could be":str(result),
                            "Alert":"Alert:Chance of Over Memory Consumption"})
    else:
       return jsonify({"Current Value": str(data[0]['data']),
                        "Next Value could be":str(result),
                        "Alert":"No Alert"})

if __name__ == "__main__":
	app.run(host='0.0.0.0', port=9000)

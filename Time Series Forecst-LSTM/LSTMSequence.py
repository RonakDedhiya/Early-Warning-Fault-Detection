"""
Steps to be Followed:-
Load the dataset from CSV file.
Transform the dataset to make it suitable for the LSTM model, including:
Transforming the data to a supervised learning problem.
Transforming the data to be stationary.
Transforming the data so that it has the scale -1 to 1.
Fitting a stateful LSTM network model to the training data.
Evaluating the static LSTM model on the test data.
Report the performance of the forecasts.

"""
## Load Librraies
import pickle
import numpy as np
import pandas as pd
from math import sqrt
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import LSTM,Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

## Definitions

## Frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df

## Removing trend- Making Stationary
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return pd.Series(diff)

## Invert diffferenced value - Get Orignial Value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

## Scale Data
def scale(data):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(data)
    data = data.reshape(data.shape[0],data.shape[1])
    scaled_X = scaler.transform(data)
    return scaler,scaled_X

## Inverse scailing for forecasted value
def inverse_scale(scaler,X,value):
    new_row = [x for x in X] + [value]
    array = np.array(new_row)
    array = array.reshape(1,len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0,-1]

## Fit an lstm network to training data
def fit_lstm(train,batch_size,nb_epochs,neurons):
    X,y = train[:,0:-1],train[:,-1]
    X = X.reshape(X.shape[0],1,X.shape[1])
    model = Sequential()
    model.add(LSTM(neurons,batch_input_shape=(batch_size,X.shape[1],X.shape[2]),stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_logarithmic_error', optimizer='Nadam')
    for i in range(nb_epochs):
        model.fit(X,y,epochs=1,batch_size=batch_size,verbose=0,shuffle=False)
        model.reset_states()
    return model

## Make a one-step forecast
def forecast_lstm(model,batch_size,X):
    X=X.reshape(1,1,len(X))
    yhat=model.predict(X,batch_size=batch_size)
    return yhat[0,0]

## Transform data and do prediction
def predict_data(data,scaler,model):
    data=data.reshape(data.shape[0],1)
    raw_values=data
    data=difference(data,1)
    data=timeseries_to_supervised(data,1)
    data=scaler.transform(data)
    for i in range(len(data)):
        X= data[i,0]
        X=X.reshape(1,1,1)
        yhat = model.predict(X,1)
        yhat=inverse_scale(scaler,X,yhat)
        yhat=inverse_difference(raw_values,yhat,len(data)+1-i)
        print(yhat)
    return yhat

## Load Data
data = pd.read_csv("data.csv",header=None,names=['col1','col2','col3','col4','col5'])
np.random.seed(1337)

## Differenced Series - Stationary Series
raw_values = data.col5.values
raw_values = raw_values.reshape(raw_values.shape[0],1)
differenced = difference(raw_values, 1)

## Transform data to supervised data
tsupervised = timeseries_to_supervised(differenced, 1)
supervised_values= tsupervised.values

## Transform scale
scaler,data = scale(supervised_values)
pickle.dump(scaler,open("scaler","wb"))

## Split Data
Len=len(data)
validation_size=0.1
Size = int(validation_size*Len)
train, test = data[:-Size],data[-Size:]

## Fit the model
lstm_model=fit_lstm(train,1,10,1)
lstm_model.save('my_model.h5')
## Forecast entire trainig set
#lstm_model.predict(train[:,0].reshape(len(train),1,1),batch_size=1)

## walk forward validation on test data
predictions=list()
for i in range(len(test)):
    #make one - step forecast
    X,y=test[i,0:-1],test[i,-1]
    yhat=forecast_lstm(lstm_model,1,X)
    #invert scaling
    yhat=inverse_scale(scaler,X,yhat)
    #invert differencing
    yhat=inverse_difference(raw_values,yhat,len(test)+1-i)
    #store forecast
    predictions.append(yhat)
    expected=raw_values[len(train) + i + 1]
    #print('predicted=%f, Expected=%f' % (yhat,expected))

## report performance
rmse = sqrt(mean_squared_error(raw_values[-Size:],predictions))
print('Test RMSE: %0.3f' % rmse)

## line plot of observed vs predicted
pyplot.plot(raw_values[-Size:])
pyplot.plot(predictions)
pyplot.legend(['desired','predicted'])
pyplot.show()

# Time Series Forecast Using LSTM

Problem Statement :- Memory consumption of certain Edge Device is provided.
Here task is to learn from its past data and do prediction of next value in future

Tools :- Python 3.6, Tensorflow 1.4, Keras, Scikit (Sklearn)

Implementation :-
Time series forecast can  be done using statistical methods but here we approched
to use neural network to build a time series forecast model.
 After initial research, we found it to go with LSTM model which is good at detecting
 sequences and can  be used for time series forecasts.

 I have referred [Time Series Forecast Using LStM](https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/)
 Tutorial from Machine Learning mastery website which has explaned explicitly in
 detail about building LSTM model from scratch. I have used very simple model of
 LSTM. There are lot many variations that can be done for enhancing the model.

 Please refer the tutorial [Link](https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/) to follow on LSTM.
 Codes are annotated with comments for better understanding.

Steps to be Followed :-  
1.Load the dataset from CSV file.  
2.Transform the dataset to make it suitable for the LSTM model    
3.Transforming the data to a supervised learning problem.  
4.Transforming the data to be stationary.  
5.Transforming the data so that it has the scale -1 to 1.  
6.Fitting a stateful LSTM network model to the training data.  
7.Evaluating the static LSTM model on the test data.  
8.Report the performance of the forecasts.  

This completes the training part.  
For Using the trained model directly, I have wriiten another function predict_data which takes in
data and model. Perform all preprocessing/ transformation of data, Loads the model and gives
the forecasted output. Output is inversed/ transformed to original form.

In short, call predict_data and get the output.

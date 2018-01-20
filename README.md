# Early-Warning-Fault-Detection  
Problem statement/Background:- Build a Tensorflow Model to do early detection of machine failure ( on basis of memory consumption)  With proper machine monitoring, we could achieve preventive maintenance, improved safety and cost savings.

High Level Requirement :-Python 3.6,Tensorflow 1.4, sklearn, keras  

Implementation:-  

We have a Edge device which is continuously monitored. We have to early detect the possible failure of machine and alert the system.Approach is to do time series forecast using Deep Learning LSTM model where it will  predict the future value beforehand and then using classifier to decide to give alert or not.  

Steps to follow :-  

1. Build a LSTM model to predict future values. Refer  [here](https://github.com/RonakDedhiya/Early-Warning-Fault-Detection/tree/master/Time%20Series%20Forecst-LSTM)  
2. Build a classifier to accurately decide/classify (Alert/ No Alert). Refer [here](https://github.com/RonakDedhiya/Early-Warning-Fault-Detection/tree/master/Logistic%20Classifier)  
3. Integration & Modification. Refer [here](https://github.com/RonakDedhiya/Early-Warning-Fault-Detection/tree/master/Fault%20Prediction)

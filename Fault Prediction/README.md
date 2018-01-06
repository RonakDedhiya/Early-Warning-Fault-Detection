# Fault Prediction
Problem Statement :- Build a Tensorflow Model to do early detection of machine failure ( on basis of memory consumption)

High Level Requirement :- Python 3.6,Tensorflow 1.4, sklearn, keras

Implementation:-  

We have a Edge device which is continuously monitored. We have to early detect the possible failure of machine and alert the system.Approach is to do time series forecast using Deep Learning LSTM model where it will  predict the future value beforehand and then using classifier to decide to give alert or not.

Steps to follow :-  
1.Build a LSTM model to predict future values. Refer [here](https://github.com/RonakDedhiya/Early-Warning-Fault-Detection/tree/master/Time%20Series%20Forecst-LSTM)                                                          2.Build a classifier to accurately decide/classify (Alert/ No Alert). Refer [here](https://github.com/RonakDedhiya/Early-Warning-Fault-Detection/tree/master/Logistic%20Classifier)  
3.Integration  
4.Modification  

Integration :-  
Soo we had this LSTM model and a classifier model ready. We wrote another code which loaded both of this model and configure input/ output accordingly. We got a single current memory consumption value as our input. We passed it to LSTM model which predicted the future value and passed the predicted value to classifier. This classifier decide if value is above prescribed machine failure threshold or not and generates status describing Alert or No Alert. Refer Prediction_final.py.

Modification :-  
Here we have trained our classifier model on certain threshold but what if device threshold changes or what if we are monitoring machine whose threshold is different. Soo we come with idea of retraining. Whenever our model is giving false alarm or no alarm (when device fails ), It is trigger point that our model is inaccurately configured for machine failure threshold. Soo if such event occur consistently and crosses above inaccurate result limit, we will retrain our network.  

Retraining of model will be done with the past data which will capture the incorrect classification and train the model for appropriate threshold value. The appropriate threshold value will be chosen by seeing past failure event(memory consumption value). The whole process is automatic and no manual training will be required.  

The Data will be continuously monitored for any alerts and retrained in case of false alarm.  

For detecting False Alarm/ No alarm, an logic is written and incorporated in predict_data function. On retraining event, retrain function is called when conditions become true. This retrain function detects the lowest value for which machine got off and adds dummy data accordingly to have on/off labels almost equal in number for proper training. This data augmentation is done in retrain code wherein ahead it is trained with logistic regression method.  

After retraining, model runs with new threshold.  
RestPrediction.py is similar code as prediction_final.py but additionally with flask to allow it to be hit from the API's

# Fault Prediction
Problem Statement :- Build a Tensorflow Model to do early detection of machine failure ( on basis of memory consumption)

High Level Requirement :- Python 3.6,Tensorflow 1.4, sklearn, keras

Implementation:- 

<img src='/Fault Prediction/Image.PNG'>

We have a Edge device which is continuously monitored. We have to early detect the possible failure of machine and alert the system.Approach is to do time series forecast using Deep Learning LSTM model where it will  predict the future value beforehand and then using classifier to decide to give alert or not.

Steps to follow :-  
1. Build a LSTM model to predict future values. Refer [here](https://github.com/RonakDedhiya/Early-Warning-Fault-Detection/tree/master/Time%20Series%20Forecst-LSTM)  
2. Build a classifier to accurately decide/classify (Alert/ No Alert). Refer [here](https://github.com/RonakDedhiya/Early-Warning-Fault-Detection/tree/master/Logistic%20Classifier)  
3. Integration  
4. Modification  

Integration :-  
Soo we had this LSTM model and a classifier model ready. We wrote another code which loaded both of this model and configure input/ output accordingly. We got a single current memory consumption value as our input. We passed it to LSTM model which predicted the future value and passed the predicted value to classifier. This classifier decide if value is above prescribed machine failure threshold or not and generates status describing Alert or No Alert. Refer Prediction_final.py.

Modification :-  
Here we have trained our classifier model on certain threshold but what if device threshold changes or what if we are monitoring machine whose threshold is different. Soo we come with idea of retraining. Whenever our model is giving false alarm or no alarm (when device fails ), It is trigger point that our model is inaccurately configured for machine failure threshold. Soo if such event occur consistently and crosses above inaccurate result limit, we will retrain our network.  

Retraining of model will be done with the past data which will capture the incorrect classification and train the model for appropriate threshold value. The appropriate threshold value will be chosen by seeing past failure event(memory consumption value). The whole process is automatic and no manual training will be required.  

The Data will be continuously monitored for any alerts and retrained in case of false alarm.  

For detecting False Alarm/ No alarm, an logic is written and incorporated in predict_data function. On retraining event, retrain function is called when conditions become true. This retrain function detects the lowest value for which machine got off and adds dummy data accordingly to have on/off labels almost equal in number for proper training. This data augmentation is done in retrain code wherein ahead it is trained with logistic regression method.  

After retraining, model runs with new threshold.  
RestPrediction.py is similar code as prediction_final.py but additionally with flask to allow it to be hit from the API's


# MQTT :-

Problem Statement :- Integrate Predictive Incident Management Model with Codex IoT.

Description :- 

All details about model is provided here.

This model is already updated to work with REST API and model is deployed in Raspberry Pi [Link].

Our model predict the future memory consumption value and alert the system if crossed above threshold. Here now with alerting the system, we are creating an event and can be viewed from codex IoT platform. We can create event from model which is running in raspberry pi by using MQTT protocol.

MQTT :-

MQTT is a machine-to-machine (M2M)/"Internet of Things" connectivity protocol. It was designed as an extremely lightweight publish/subscribe messaging transport. It is useful for connections with remote locations where a small code footprint is required and/or network bandwidth is at a premium.

We wrote a small code in python to send event using MQTT protocol.

steps:-  
1.pip install paho-mqtt.  
2.This package provides a client class which enable applications to connect to an MQTT broker to publish messages, and to subscribe to topics and receive published messages. It also provides some helper functions to make publishing one off messages to an MQTT server very straightforward.  
3.We have to first make a payload, which is nothing but a json packet and we have to include all information required to register an event.  
payload={    
                  "customer_id" : "####",  
                  "Project_id" : "MI_WM_EAST_WINDMILL_FARM",  
                   "asset_id" : "MI_WM_EDGEG1",  
                   "event_name":"Device Failure Prediction",  
                   "event_description":"Device will fail Take Action",  
                   "event_type":"Failure Prediction",  
                   "severity":"high",  
                   "status":"open",  
                    "created_date":"2018-01-19 21:10:58.786"  
                }  
4.Invoke MQTT Client and connect to host.  
client=mqtt.client()  
client.connect("host_ip",1884,60)  
5. And finally publish payload to assigned topic  
client.publish("codexidm_iot_mi_event_data",payload)  
  

# Creating Classifier model

Problem Statement :- We have a data from edge device related to its memory consumption.
This device have a threshold above which device may fail. Thus we need to make a Classifier
where it will distinguish wheather device is in fail or active mode.
Here ultimately what we are doing is given a memory consumption value, we need to decide
device is on or off

Tools :- Python 3.6, Sklearn

Implementation :- Here, We have a data with 2 columns (i.e cpu_util, device_status). Device_status
is like a label denotes classes.
With this training data, we train a sklearn model ( Logistic regression) and classify output quite accurately.

Refer code for details(Logistic.py).

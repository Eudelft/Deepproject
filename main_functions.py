# -*- coding: utf-8 -*-
"""


@author: Abenezer
"""


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.style.use('ggplot')
import os
import tensorflow as tf
from tensorflow.python.client import device_lib
from keras.models import load_model
from sklearn.preprocessing import StandardScaler


def pre_process_encoder(df):

    'Features and label'
    X = df.drop('Class',axis=1)
    y = df.Class
   
    if "Car_Id" in X.columns:
        X.drop('Car_Id', axis=1, inplace=True)
    if 'Trip'in X.columns:
        X.drop('Trip', axis=1, inplace=True)
        
    return X,y
    

'Split the data set into window samples'

from sklearn.preprocessing import LabelEncoder

def window(X1, y1):
    '''
    returns the time-series data in sequential order and window with overlap.
    Essential we get an array of X_samples.

    X_samples[0] represents the first window which will have a 2D shape. The number of rows (or timesteps) it contains 
    will depend on the size of the window. The number of columns = total features - class feature.

    If we have 100,000 datapoints and 100 windows. Then each window will have a window_size = 1000 timesteps.    
    '''

    # instantiate lists
    X_samples = []
    y_samples = []
    
    # one hot encode class labels 
    encoder = LabelEncoder()
    encoder.fit(y1)
    y1 = encoder.transform(y1)
    
    # window_size (length of the window)
    length = 16
    overlapsize = length // 2
    n = y1.size    
 
    Xt = np.array(X1)
    yt= np.array(y1).reshape(-1,1)

    # for over the 263920  310495in jumps of 64
    # i = 0, 8, 16, 24... --> function will store 0-->16, 8-->24, 16-->32 etc. (so there is overlap between the timeseries data)
    # overlapsize is crucial as X_sample[0] and X_sample[1] will contain shared data
    for i in range(0, n , length-overlapsize):
        
        # grab from i to i+length 
        sample_x = Xt[i:i+length,:]

        # assures that the windows all have the same size and does not include other part of the code
        if (np.array(sample_x).shape[0]) == length: 
            X_samples.append(sample_x)

        sample_y = yt[i:i+length]
        if (np.array(sample_y).shape[0]) == length:
            y_samples.append(sample_y)

    return np.array(X_samples),  np.array(y_samples)


'for the label Select the maximum occuring value in the given array'
def max_occuring_label(sample):

    # stores which values occur and how many of the class
    values, counts = np.unique(sample, return_counts=True)

    # returns the index of the maximum value of a Numpy array
    ind = np.argmax(counts)
    
    return values[ind] 


'Creating y_sample label by taking only the maximum'

def label_y(y_value):
    """
    Essentially, there a number of windows and we would like to associate each window to one specific class. (e.g. window[5] = class["A"] )
    Most of the time the windows will have the same class at each timestep, especially if there are very small window size of < 60 seconds. 
    However, there will be a small number of windows which will contain two classes and we then would like to select the most occuring class to train the window on.
    """
    y_samples_1 = []
    for i in range(len(y_value)):
        y_samples_1.append(max_occuring_label(y_value[i]))
        
    return np.array( y_samples_1 ).reshape(-1,1)


from sklearn.model_selection import train_test_split
# from keras.utils import to_categorical (author wrote like this)
from tensorflow.keras.utils import to_categorical 

def rnn_dimension(X,y):

    # creates windows out of the data
    # Note, y_samples.shape = (windows, 1, window_size)
    X_samples, y_samples = window(X, y)

    # finds the most occuring class in each window and stores it. 
    # converts y_samples --->  y_samples.shape(windows, 1)
    y_samples = label_y(y_samples)

    # shuffles the windows in random fashion
    from sklearn.utils import shuffle
    X_samples,  y_samples = shuffle(X_samples, y_samples)

    # onehot encoding of the data. Right now each window is associate with a class integer. We'd like to onehot encode this before training the model
    # e.g. if a window has class 2 --> [0, 0, 1, 0, 0 ... 0]
    y_samples_cat = to_categorical(y_samples)

    # split the train, validation, and test data
    X_train_rnn, X_test_rnn, y_train_rnn, y_test_rnn = train_test_split(X_samples, y_samples_cat, train_size=0.85)

    # shuffle the data again for ultimate shuffling (idk why it's being done again)
    X_train,  y_train = shuffle(X_train_rnn, y_train_rnn)
    
    return X_train, y_train, X_test_rnn, y_test_rnn



def normalizing(X_test):
            
    dim1=X_test.shape[1]
    dim2=X_test.shape[2]

    X_test_2d = X_test.reshape(-1,dim2)
    scale = StandardScaler()
    scale.fit(X_test_2d)

    X_test_scaled = scale.transform(X_test_2d)
    X_test_scaled = X_test_scaled.reshape(-1,dim1,dim2)

    return X_test_scaled
        
        



anomality_level = [0,0.05,0.1,0.2,0.4,0.6,0.8]

def LSTM_anomality(X_test_rnn,y_test_rnn ):
    acc_noise_test = []
    acc_noise_test_rf_box = []
    for anomaly in anomality_level:
        print("="*5)
        print("for anomaly percentage = ",anomaly)

        def anomality(X, ): 
            orgi_data = np.copy(X_test_5.reshape(-1,21))
            mask = np.random.choice( orgi_data.shape[0], int(len(orgi_data)* .5), replace=False)
            # orgi_data[mask].shape

            orgi_data[mask] = orgi_data[mask]+orgi_data[mask]*anomaly
            
            return orgi_data
        
        def normalizing(X_test):
            
            dim1=X_test.shape[1]
            dim2=X_test.shape[2]

            X_test_2d = X_test.reshape(-1,dim2)
            scale = StandardScaler()
            scale.fit(X_test_2d)

            X_test_scaled = scale.transform(X_test_2d)
            X_test_scaled = X_test_scaled.reshape(-1,dim1,dim2)

            return X_test_scaled

           
        iter_score = []    
        for i in range(5):
            
            X_test_rnn_anomal = np.copy(anomality(X_test_rnn).reshape(-1,X_test_5.shape[1],X_test_5.shape[2]))
            
            X_test_rnn_noise_scaled = normalizing(X_test_rnn_anomal)
           
            #pd.DataFrame(noising2(X_train.reshape(-1,49)))[1].head(1000).plot(kind='line')

            score_1 = clean_model.evaluate(X_test_rnn_noise_scaled, y_test_rnn, batch_size=50,verbose=0)
            iter_score.append(score_1[1])
            # print(score_1[1])

        dif = max(iter_score) - min(iter_score)
        score_2 = sum(iter_score)/len(iter_score)
        acc_noise_test.append(score_2)
        print('Avg Test loss:', score_2)
        print('Avg Test accuracy:', score_2)
        acc_noise_test_rf_box.append(dif)
        
    return acc_noise_test,acc_noise_test_rf_box
        
        

def normalizing_2d(X):              
           
    scale = StandardScaler()
    scale.fit(X)

    X = scale.transform(X)
    
    return X
        
def anomality_2d(X, anomaly): 

    X = np.array(X).reshape(-1,21)
    mask = np.random.choice( X.shape[0], int(len(X)* .4), replace=False)
    # orgi_data[mask].shape

    X[mask] = X[mask]+X[mask]*anomaly

    return X




from keras.utils import np_utils 
from sklearn.preprocessing import StandardScaler


from keras.models import Sequential
from keras.layers import Dense
from sklearn.utils import shuffle
from keras import layers
from sklearn.model_selection import train_test_split


def mlp_acc_test(X_test, y_test):
    acc_noise_test = []
    acc_noise_test_rf_box = []
    
    # anomality_level = [0,0.2,0.4,0.6,0.8,1]
        
    for anomal in anomality_level:      

        i = 0
        iter_score = []
        while i < 5:
            X_test_anomal = np.copy(anomality_2d(X_test, anomal))
            X_test_normalized = normalizing_2d(X_test_anomal)


            score_1 = mlp.evaluate(X_test_normalized, y_test, batch_size=50)
            iter_score.append(score_1[1])
            i += 1
            # print(i)
  
        dif = max(iter_score) - min(iter_score) 
        score_2 = sum(iter_score)/len(iter_score)
        acc_noise_test.append(score_2)
        print('Avg Test loss:', score_2)
        print('Avg Test accuracy:', score_2)
        acc_noise_test_rf_box.append(dif)

    return acc_noise_test, acc_noise_test_rf_box




from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix  
from sklearn.tree import DecisionTreeClassifier 
from sklearn import metrics



def acc_noise_test_dt(X_train, y_train ,X_test , y_test):
    
    dt = DecisionTreeClassifier()
    dt.fit(X_train,y_train)

    acc_noise_test_dt = []
    acc_noise_test_rf_box = []



    
    
    for anomal in anomality_level:
       
        iter_score=[]
        for i in range(10):
            
            X_test_anomal = np.copy(anomality_2d(X_test, anomal))
            X_test_normalized = normalizing_2d(X_test_anomal)
           

            'Decision Tree'
            y_pred_dt = dt.predict(X_test_normalized)   
            acc_n = metrics.accuracy_score(y_test, y_pred_dt)
            
            iter_score.append(acc_n)
            
        dif = max(iter_score) - min(iter_score)    
        score_2 = sum(iter_score)/len(iter_score)
        acc_noise_test_dt.append(score_2)
        print('Avg Test loss:', score_2)
        print('Avg Test accuracy:', score_2)
        acc_noise_test_rf_box.append(dif)
            

        
    return  acc_noise_test_dt, acc_noise_test_rf_box




from sklearn.metrics import classification_report, confusion_matrix  
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics



def acc_noise_test_rf(X_train, y_train ,X_test , y_test):
    
    rf = RandomForestClassifier(n_estimators=20)
    rf.fit(X_train, y_train)

    acc_noise_test_rf = []
    acc_noise_test_rf_box = []
    
    for anomal in anomality_level:
       
        iter_score=[]
        for i in range(10):
            
            X_test_anomal = np.copy(anomality_2d(X_test, anomal))
            X_test_normalized = normalizing_2d(X_test_anomal)           

        
            'Random Forest'
            y_pred_rf =rf.predict(X_test_normalized) 
            acc_n = metrics.accuracy_score(y_test, y_pred_rf)
            iter_score.append(acc_n)
            # print(acc_n)
        
        dif = max(iter_score) - min(iter_score)
        acc_noise_test_rf_box.append(dif)
        score_2 = sum(iter_score)/len(iter_score)
        acc_noise_test_rf.append(score_2)
        
        print("=")
        print(score_2)
        
        
    return (acc_noise_test_rf,acc_noise_test_rf_box)



import pickle
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model, neighbors, datasets
from sklearn import svm
import scipy.signal as signal
import numpy as np
from sklearn import svm
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import LeavePOut
from sklearn.preprocessing import StandardScaler
import sys
import time
import os
import math
import shutil
from sklearn.externals import joblib
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import numpy as np
import pickle as pk
import sys
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score

import keras as ke
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, UpSampling2D,Conv2DTranspose,ZeroPadding2D,MaxPooling1D,MaxPooling2D, CuDNNLSTM, Dropout, Reshape, PReLU, ELU, BatchNormalization, Flatten
from keras import optimizers
from keras import initializers
from keras.callbacks import ReduceLROnPlateau, CSVLogger, ModelCheckpoint
from keras.utils import to_categorical
from library.utilities import *
from library.test_pipeline_matrix import test_features
from keras.models import load_model





# parameters
window_size = 900
step_size=50
num_channels=5
num_of_features=200




########### loading trained model##########
model = load_model('models/autoencoder_rawsignal.h5')
############################################


############# loading input ###############
path='input/'
#subfolder
train_data_name='day1'
##########################################à



# run prediction
[data_matrix,labels]=get_data(path+train_data_name,num_channels,window_size,step_size)
print(data_matrix.shape)
reshaped_data_matrix = data_matrix.reshape(data_matrix.shape[0], int(window_size), num_channels, 1)


#### create features #####################
from keras.models import Model
layer_name = 'dense_layer_encode1'
intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
# predict features
intermediate_output = intermediate_layer_model.predict(reshaped_data_matrix)
print('Feature Matrix of shape:')
print(intermediate_output.shape)



# ettore analysis
#test_features(intermediate_output, 
       #      labels, day='day label', visit='visit label', patient='patient label', train_step=500)



#svm classifier
clf = svm.SVC(C = 1, kernel = 'linear', gamma = 'auto')
Model_Name='SVM_classifier_1'
labels=labels.reshape(len(labels),)
performance_assesment_fraction_std(intermediate_output, labels, 10, clf,Model_Name)






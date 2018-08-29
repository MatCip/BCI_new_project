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

# INPUT FOLDER
path='input/'

#INPUT SUBFOLDERS (IF PRESENT) ( ex: day1, day2 ..)
autoenconder_data=['']



# prepare data for autoencoder
train_data=get_autoencoder_data(path,autoenconder_data,num_channels,window_size,step_size)
reshaped_train=train_data;





size_of_kernel_1= (5,2)
size_of_kernel_2= (9,1)

kernel_strides = 1
num_filters = 4

dropout_prob = 0.5
inputshape = (window_size, num_channels, 1)

# BUILDING MODEL USING KERAS AND TENSORFLOW BACKEND
print('Building Model...')
model = Sequential()

model.add(Conv2D(num_filters, kernel_size=size_of_kernel_1, strides=kernel_strides,
                 activation='relu', input_shape=inputshape, name='1_conv_layer'))

model.add(Conv2D(num_filters, kernel_size=size_of_kernel_2, strides=kernel_strides,
                 activation='relu', input_shape=inputshape, name='2_conv_layer'))
conv_1_shape=model.get_layer(name='1_conv_layer').output_shape
conv_2_shape=model.get_layer(name='2_conv_layer').output_shape



model.add(MaxPooling2D(pool_size=(4, 2), strides=None, padding='valid', data_format=None,name='max_pooling_1'))
model.add(MaxPooling2D(pool_size=(2, 1), strides=None, padding='valid', data_format=None,name='max_pooling_2'))
max_pool_1_shape=model.get_layer(name='max_pooling_1').output_shape
max_pool_2_shape=model.get_layer(name='max_pooling_2').output_shape

print(max_pool_1_shape)
print(conv_1_shape)
print(conv_2_shape)
#
model.add(Flatten(name='flatten'))

model.add(Dense(num_of_features, activation='relu', name='dense_layer_encode1'))

#decoding
flatten=model.get_layer(name='flatten')
model.add(Dense(flatten.output_shape[1], activation='relu', name='dense_layer_decode1'))

model.add(Reshape((max_pool_2_shape[1],max_pool_2_shape[2],max_pool_2_shape[3])))
model.add(UpSampling2D(size=(2,1), data_format=None))
model.add(UpSampling2D(size=(4,2), data_format=None))
model.add(Conv2DTranspose(4, (9,1), strides=1, padding='valid',  activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
model.add(Conv2DTranspose(1, (5,2), strides=1, padding='valid',  activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))


Adam=optimizers.Adam(lr=0.001)
model.compile(loss='mean_squared_error', optimizer=Adam, metrics=['accuracy'])


print(model.summary())




batchSize = 50
train_epoches = 30
model.fit(reshaped_train,reshaped_train,epochs=train_epoches,batch_size=batchSize,verbose=1)
model.save('autoencoder_model.h5')

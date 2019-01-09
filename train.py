import pickle
import numpy as np

import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Reshape, Dropout, BatchNormalization, ZeroPadding2D, Dense, Activation, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv1D
from keras.layers.merge import Concatenate
import tensorflow as tf
from keras.backend import tensorflow_backend
from keras.utils import np_utils

np.random.seed(0)
np.random.RandomState(0)
tf.set_random_seed(0)

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)

iterations = 2000
batch_size = 1024
test_interval = 200
save_interval = 2000

FILENAME = './model/kinect.model'


k_input = Input((4,)) #x,y,sin(theta),cos(theta)

model = Dense(256, activation='relu')(k_input)
model = Dropout(0.25)(model)
model = Dense(128, activation='relu')(model)
model = Dropout(0.25)(model)
model = Dense(32, activation='relu')(model)
model = Dropout(0.25)(model)
model = Dense(16, activation='relu')(model)
model = Dropout(0.25)(model)

model = Dense(3, activation='linear')(model)

kinect = Model(inputs=k_input, outputs=model)

kinect.summary()

kinect.compile(loss='mean_squared_error',optimizer='adam')

def theta2ans(theta):
    s = np.sin(theta).T
    c = np.cos(theta).T
    s12 = s[0]*c[1] + c[0]*s[1]
    c12 = c[0]*c[1] - s[0]*s[1]
    s123 = s12*c[2] + c12*s[2]
    c123 = c12*c[2] - s12*s[2]
    x = s[0] + s12 + s123
    y = c[0] + c12 + c123
    th = theta.sum(axis=1)

    return np.array([x,y,np.cos(th),np.sin(th)]).T

for iteration in range(iterations):
    theta = np.random.uniform(-np.pi/2, np.pi/2, (batch_size,3))
    ans = theta2ans(theta)
    loss = kinect.train_on_batch(ans,theta)

test_size = 10
theta = np.random.uniform(-np.pi/2, np.pi/2, (test_size,3))
ans = theta2ans(theta)

predict = kinect.predict(ans)
ans2 = theta2ans(predict)

for i in range(test_size):
  print(i)
  print('\t'.join(map(str,ans[i])))
  print('\t'.join(map(str,ans2[i])))

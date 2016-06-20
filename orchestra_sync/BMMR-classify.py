import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import pickle
import os
import glob
import sys
import gzip

import numpy as np

import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum

from nolearn.lasagne import NeuralNet
from nolearn.lasagne import visualize

from sklearn.preprocessing import normalize
from sknn.mlp import Classifier, Layer, Convolution
from sklearn import cross_validation
from sklearn.metrics import roc_auc_score

from PIL import Image
import codecs

import csv
import xlrd


# Constants
SIZE = 128
NUM = 500
BATCH = 32
EPOCH = 20
TRAIN_NUM = 9000
VAL_NUM = 10000
TEST_NUM = 11432
TRAIN = "conv_patches_" + str(SIZE) + "";
TRAINCLASS = "conv_patches_class_" + str(SIZE) + "";
TEST = "test_patches";
TESTCLASS = "test_patches_class";

# Utils
def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

#Neural Net model

net1 = NeuralNet(
    layers=[('input', layers.InputLayer),
            ('conv2d1', layers.Conv2DLayer),
            ('maxpool1', layers.MaxPool2DLayer),
            ('conv2d2', layers.Conv2DLayer),
            ('maxpool2', layers.MaxPool2DLayer),
            ('dropout1', layers.DropoutLayer),
            ('dense', layers.DenseLayer),
            ('dropout2', layers.DropoutLayer),
            ('output', layers.DenseLayer),
            ],
    # input layer
    input_shape=(None, 1, 128, 128),
    # layer conv2d1
    conv2d1_num_filters=256,
    conv2d1_filter_size=(8, 8),
    conv2d1_nonlinearity=lasagne.nonlinearities.rectify,
    conv2d1_W=lasagne.init.GlorotUniform(),  
    # layer maxpool1
    maxpool1_pool_size=(3, 3),    
    # layer conv2d2
    conv2d2_num_filters=192,
    conv2d2_filter_size=(6, 6),
    conv2d2_nonlinearity=lasagne.nonlinearities.rectify,
    # layer maxpool2
    maxpool2_pool_size=(2, 2),
    # dropout1
    dropout1_p=0.4,    
    # dense
    dense_num_units=256,
    dense_nonlinearity=lasagne.nonlinearities.rectify,    
    # dropout2
    dropout2_p=0.4,    
    # output
    output_nonlinearity=lasagne.nonlinearities.softmax,
    output_num_units=10,
    # optimization method params
    update=nesterov_momentum,
    update_learning_rate=0.002,
    update_momentum=0.9,
    max_epochs=20,
    verbose=1,
    )

# Read in images/classes

trainImg = np.zeros((NUM, 1, SIZE, SIZE))
trainVal = np.zeros((NUM))

num = 0
for filename in glob.iglob('cancer_images_updated/*.jpg'):
    #print(filename)
    img = Image.open(filename);
    img = img.resize((SIZE, SIZE))
    pix = img.load()
    for i in range(0, SIZE):
    	for j in range(0, SIZE):
    		trainImg[num, 0, i, j] = float(pix[i, j])/256
    num = num + 1
    trainVal[num] = 0

for filename in glob.iglob('normal_images_updated/*.jpg'):
    #print(filename)
    img = Image.open(filename);
    img = img.resize((SIZE, SIZE))
    pix = img.load()
    for i in range(0, SIZE):
        for j in range(0, SIZE):
            trainImg[num, 0, i, j] = float(pix[i, j])/256
    num = num + 1
    trainVal[num] = 1





NUM = num
trainImg = trainImg[0:NUM, ]
trainVal = trainVal[0:NUM]
trainImg = trainImg.astype(np.float32)
trainVal = trainVal.astype(np.uint8)

print(trainImg)
print(num)

TRAIN_NUM = NUM*0.7
TEST_NUM = NUM*1.0

print('Train Image shape:', trainImg.shape)

print("Train Validation:")

print(trainVal)
print(sum(trainVal))
print(len(trainVal))


nn = net1.fit(trainImg[0:TRAIN_NUM], trainVal[0:TRAIN_NUM])
preds = net1.predict_proba(trainImg[TRAIN_NUM:TEST_NUM])
actual = trainVal[TRAIN_NUM:TEST_NUM]

score = roc_auc_score(preds, actual, average='macro')

print score





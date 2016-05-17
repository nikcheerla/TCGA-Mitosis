import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import cPickle as pickle
import os
import glob
import sys

import numpy as np

sys.setrecursionlimit(80000)

from sklearn.utils import shuffle
from scipy.spatial import distance
from sklearn import cross_validation

import sknn
from sknn.platform import gpu32


# Constants
SIZE = 2084
PATCH_SIZE = 101
PATCH_GAP = 50
RADIUS = 10

print "Here, we train a Lasagne network to predict whether or not a pixel is mitotic.t"


"""Important GPU testing! Will verify whether GPU or CPU is being used!"""

from theano import function, config, shared, sandbox
import theano.tensor as T
import numpy
import time


vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
iters = 100

rng = numpy.random.RandomState(22)
x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
f = function([], T.exp(x))
print(f.maker.fgraph.toposort())
t0 = time.time()
for i in range(iters):
    r = f()
t1 = time.time()
print("Looping %d times took %f seconds" % (iters, t1 - t0))
print("Result is %s" % (r,))
if numpy.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]):
    print('Used the cpu')
else:
    print('Used the gpu')










"""Building Lasagne Network"""

import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum

from nolearn.lasagne import NeuralNet
from nolearn.lasagne import visualize

print("Building Classifier:")
nn = NeuralNet(
    layers=[('input', layers.InputLayer),
            ('conv2d1', layers.Conv2DLayer),
            ('maxpool1', layers.MaxPool2DLayer),
            ('conv2d2', layers.Conv2DLayer),
            ('maxpool2', layers.MaxPool2DLayer),
            ('dropout1', layers.DropoutLayer),
            ('conv2d3', layers.Conv2DLayer),
            ('maxpool3', layers.MaxPool2DLayer),
            ('dropout2', layers.DropoutLayer),
            ('conv2d4', layers.Conv2DLayer),
            ('maxpool4', layers.MaxPool2DLayer),
            ('dense', layers.DenseLayer),
            ('dropout3', layers.DropoutLayer),
            ('output', layers.DenseLayer),
            ],
    # input layer
    input_shape=(None, 3, PATCH_SIZE, PATCH_SIZE),
    # layer conv2d1
    conv2d1_num_filters=192,
    conv2d1_filter_size=(4, 4),
    conv2d1_nonlinearity=lasagne.nonlinearities.rectify,
    conv2d1_W=lasagne.init.GlorotUniform(),
    # layer maxpool1
    maxpool1_pool_size=(2, 2),
    # layer conv2d2
    conv2d2_num_filters=128,
    conv2d2_filter_size=(4, 4),
    conv2d2_nonlinearity=lasagne.nonlinearities.rectify,
    # layer maxpool2
    maxpool2_pool_size=(2, 2),
    # dropout1
    dropout1_p=0.5,
    #conv2d3
    conv2d3_num_filters=96,
    conv2d3_filter_size=(4, 4),
    conv2d3_nonlinearity=lasagne.nonlinearities.rectify,
    conv2d3_W=lasagne.init.GlorotUniform(),
    # layer maxpool1
    maxpool3_pool_size=(2, 2),
    #conv2d4
    conv2d4_num_filters=96,
    conv2d4_filter_size=(4, 4),
    conv2d4_nonlinearity=lasagne.nonlinearities.rectify,
    conv2d4_W=lasagne.init.GlorotUniform(),
    # layer maxpool1
    maxpool4_pool_size=(2, 2),
    dropout2_p=0.5,
    # dense
    dense_num_units=128,
    dense_nonlinearity=lasagne.nonlinearities.rectify,
    # dropout2
    dropout3_p=0.3,
    # output
    output_nonlinearity=lasagne.nonlinearities.softmax,
    output_num_units=2,
    # optimization method params
    update=nesterov_momentum,
    update_learning_rate=0.007,
    update_momentum=0.9,
    max_epochs=16,
    verbose=1,
    )






"""Loading data and training Lasagne network using nolearn"""

trainImg = np.load('trainImg_stage1.npy')
trainVal2 = np.load('trainVal_stage1.npy')
trainImg2 = trainImg.astype(np.float32).swapaxes(1, 3)
trainVal2 = trainVal2.astype(np.uint8)

print "Training Classifier: 70/30 split"
nn.fit(trainImg2, trainVal2)


print "Saving Classifier"
pickle.dump(nn, open("nn_stage1.pkl", "wb"))
nn.save_weights_to('weights_stage1')

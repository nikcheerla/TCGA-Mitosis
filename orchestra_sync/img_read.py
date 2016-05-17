import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from urllib import urlretrieve
import cPickle as pickle
import os
import glob
import sys
import gzip

import numpy as np

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import progressbar

# Constants
SIZE = 2084
PATCH_SIZE = 101
PATCH_GAP = 50
RADIUS = 10

from contextlib import contextmanager
import warnings
import sys, os
import theano

import sklearn
import sknn

from sklearn.utils import shuffle
from scipy.spatial import distance
from sklearn import cross_validation

from nolearn.lasagne import BatchIterator
from sklearn.metrics import roc_auc_score



# command line args

imgfile = sys.argv[1]
print "IMAGE: " + imgfile
outfile = imgfile[:-4] + ".out"
imgoutfile = imgfile[:-4] + ".png"






#loading network

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
    conv2d1_num_filters=128,
    conv2d1_filter_size=(4, 4),
    conv2d1_nonlinearity=lasagne.nonlinearities.rectify,
    conv2d1_W=lasagne.init.GlorotUniform(),
    # layer maxpool1
    maxpool1_pool_size=(2, 2),
    # layer conv2d2
    conv2d2_num_filters=64,
    conv2d2_filter_size=(4, 4),
    conv2d2_nonlinearity=lasagne.nonlinearities.rectify,
    # layer maxpool2
    maxpool2_pool_size=(2, 2),
    # dropout1
    dropout1_p=0.4,
    # conv2d3
    conv2d3_num_filters=64,
    conv2d3_filter_size=(4, 4),
    conv2d3_nonlinearity=lasagne.nonlinearities.rectify,
    conv2d3_W=lasagne.init.GlorotUniform(),
    # layer maxpool1
    maxpool3_pool_size=(2, 2),
    # conv2d4
    conv2d4_num_filters=32,
    conv2d4_filter_size=(4, 4),
    conv2d4_nonlinearity=lasagne.nonlinearities.rectify,
    conv2d4_W=lasagne.init.GlorotUniform(),
    # layer maxpool1
    maxpool4_pool_size=(2, 2),
    dropout2_p=0.3,
    # dense
    dense_num_units=128,
    dense_nonlinearity=lasagne.nonlinearities.rectify,
    # dropout2
    dropout3_p=0.4,
    # output
    output_nonlinearity=lasagne.nonlinearities.softmax,
    output_num_units=2,


    # optimization method params
    update=nesterov_momentum,
    update_learning_rate=0.0052,
    update_momentum=0.84,
    #on_epoch_finished=[
    #    AdjustVariable('update_learning_rate', start=0.018, stop=0.0001),
    #    AdjustVariable('update_momentum', start=0.9, stop=1.09),
    #],
    max_epochs=1,
    verbose=1
)

nn.load_params_from("cachednn0.986.params");

num = 0;

patch_probs = np.zeros((SIZE, SIZE));
patch_probs = patch_probs.astype(np.float32);

SKIP = 4;

try:
    with progressbar.ProgressBar(max_value=(2084/SKIP*2084/SKIP + 2)) as bar:
        img = plt.imread(imgfile);
        patch = np.zeros((1, PATCH_SIZE, PATCH_SIZE, 3))
        patch = patch.astype(np.float32)
        patch = np.swapaxes(patch, 1, 3)

        y1, y2 = PATCH_SIZE, SIZE - PATCH_SIZE
        x1, x2 = PATCH_SIZE, SIZE - PATCH_SIZE

        mmx = SIZE/8

        for i in range(x1, x2, SKIP):
            for j in range(y1, y2, SKIP):
                sx = i - PATCH_SIZE/2
                sy = j - PATCH_SIZE/2
                patch = np.swapaxes(patch, 1, 3)
                patch[0] = np.divide(img[sx:sx + PATCH_SIZE, sy:sy+PATCH_SIZE], 255.0)
                patch = np.swapaxes(patch, 1, 3)
                prob = nn.predict_proba(patch)
                patch_probs[i, j] = prob[0, 1]
                bar.update(num)
                num += 1
except:
    pass
np.save(outfile, patch_probs)
plt.imsave(imgoutfile, patch_probs)

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

import theano
from sknn.platform import gpu32

from nolearn.lasagne import NeuralNet, TrainSplit
from lasagne.updates import adam
from nolearn.lasagne import BatchIterator
import cPickle as pickle
import lasagne
import PosterExtras as phf

from nolearn_utils.iterators import (
    ShuffleBatchIteratorMixin,
    AffineTransformBatchIteratorMixin,
    RandomFlipBatchIteratorMixin,
    LCNBatchIteratorMixin,
    MeanSubtractBatchiteratorMixin,
    make_iterator
)

from nolearn_utils.hooks import (
    SaveTrainingHistory, PlotTrainingHistory,
    EarlyStopping
)

from nolearn.lasagne import BatchIterator

from contextlib import contextmanager


# Constants
SIZE = 600





### TODO 1: Read in data ####
## Final result: trainImg and trainVal contain data ##


# Read in images/classes

trainImg = []
trainVal = []

num = 0
for filename in glob.iglob('BMMR-cancer/*.jpg'):
    img = Image.open(filename)
    img = img.resize((SIZE,SIZE))
    img = np.asarray(img)
    trainImg.append(img)
    trainVal.append(0)

for filename in glob.iglob('BMMR-normal/*.jpg'):
    img = Image.open(filename)
    img = img.resize((SIZE,SIZE))
    img = np.asarray(img)
    trainImg.append(img)
    trainVal.append(0)

trainImg = np.array([trainImg, trainImg, trainImg]).swapaxes(0, 1)
print trainImg.shape
trainVal = np.array(trainVal)
trainVal = trainVal.astype(np.uint8)


















### TODO 2: CREATE AND DESIGN CNN ####
## Final result: nn contains desired CNN ##




print("\n\nCreating and Designing CNN.")

def roc_robust(y_true, y_proba):
    if sum(y_true) == 0 or sum(y_true) == len(y_true):
        return 0.0
    else:
        return roc_auc_score(y_true, y_proba)

print("Building Image Perturbation Models/Callbacks:")

train_iterator_mixins = [
    RandomFlipBatchIteratorMixin,
    AffineTransformBatchIteratorMixin,
    #MeanSubtractBatchiteratorMixin
]
TrainIterator = make_iterator('TrainIterator', train_iterator_mixins)

test_iterator_mixins = [
    RandomFlipBatchIteratorMixin,
    #MeanSubtractBatchiteratorMixin
]
TestIterator = make_iterator('TestIterator', test_iterator_mixins)

mean_value = np.mean(np.mean(np.mean(img)))

train_iterator_kwargs = {
    'batch_size': 20,
    'affine_p': 0.5,
    'affine_scale_choices': np.linspace(start=0.85, stop=1.2, num=10),
    'flip_horizontal_p': 0.5,
    'flip_vertical_p': 0.5,
    'affine_translation_choices': np.arange(-5, 6, 1),
    'affine_rotation_choices': np.linspace(start=-20.0, stop=20.0, num=10),
    #'mean': mean_value,
}
train_iterator_tmp = TrainIterator(**train_iterator_kwargs)

test_iterator_kwargs = {
    'batch_size': 20,
    'flip_horizontal_p': 0.5,
    'flip_vertical_p': 0.5,
    #'mean': mean_value,
}
test_iterator_tmp = TestIterator(**test_iterator_kwargs)






class CustomBatchIterator(BatchIterator):

    def __init__(self, batch_size, built_iterator):
        super(CustomBatchIterator, self).__init__(batch_size=batch_size)
        self.iter = built_iterator

    def transform(self, Xb, yb):
        Xb = Xb.astype(np.float32)
        yb = yb.astype(np.uint8)
        Xb, yb = self.iter.transform(Xb, yb)
        #for i in range(0, len(yb)):
        #    plt.imsave("img" + str(yb[i]) + "num" + str(i) + ".png", Xb[i].swapaxes(0, 2))
        return Xb, yb

train_iterator = CustomBatchIterator(
    batch_size=20, built_iterator=train_iterator_tmp)
test_iterator = CustomBatchIterator(
    batch_size=20, built_iterator=test_iterator_tmp)

# Model Specifications
net = phf.build_GoogLeNet(SIZE, SIZE)
values = pickle.load(open('blvc_googlenet.pkl', 'rb'))['param values'][:-2]

nn = NeuralNet(
    net['softmax'],
    max_epochs=1,
    update=adam,
    update_learning_rate=.0001, #start with a really low learning rate
    #objective_l2=0.0001,

    # batch iteration params
    batch_iterator_train=train_iterator,
    batch_iterator_test=test_iterator,

    train_split=TrainSplit(eval_size=0.2),
    verbose=3,
)




nn.fit(trainImg, trainVal)

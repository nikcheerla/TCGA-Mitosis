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



import lasagne


import PosterExtras as phf



from nolearn.lasagne import NeuralNet, TrainSplit
from lasagne.updates import adam
from nolearn.lasagne import BatchIterator



# Constants
SIZE = 2084
PATCH_SIZE = 139
PATCH_GAP = 69
RADIUS = 10



radius = PATCH_GAP
kernel = np.zeros((2*radius+1, 2*radius+1))
y,x = np.ogrid[-radius:radius+1, -radius:radius+1]
mask = x**2 + y**2 >= radius**2



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

class RadialBatchIterator(BatchIterator):

    def __init__(self, batch_size):
        super(RadialBatchIterator, self).__init__(batch_size=batch_size)

    def transform(self, Xb, yb):
        Xb = Xb.astype(np.float32).swapaxes(1, 3)
        for i in range(0, Xb.shape[0]):
            for c in range(0, 3):
                Xb[i, c][mask] = 0.0
        if yb != None:
            yb = yb.astype(np.uint8)
        #for i in range(0, len(yb)):
        #    plt.imsave("img" + str(yb[i]) + "num" + str(i) + ".png", Xb[i].swapaxes(0, 2))
        return Xb, yb


test_iterator = RadialBatchIterator(batch_size=1)

net = phf.build_GoogLeNet(PATCH_SIZE, PATCH_SIZE)


nn = NeuralNet(
    net['softmax'],
    max_epochs=1,
    update=adam,
    update_learning_rate=.00014, #start with a really low learning rate
    #objective_l2=0.0001,

    # batch iteration params
    batch_iterator_test=test_iterator,

    train_split=TrainSplit(eval_size=0.2),
    verbose=3,
)

nn.load_params_from("cachedgooglenn.params");


patch_probs = np.zeros((SIZE, SIZE));
patch_probs = patch_probs.astype(np.float32);

SKIP = 2;

num = 0;
with progressbar.ProgressBar(max_value=(2084/SKIP*2084/SKIP + 2)) as bar:
    img = plt.imread(imgfile);
    patch = np.zeros((1, PATCH_SIZE, PATCH_SIZE, 3))
    patch = patch.astype(np.float32)

    y1, y2 = PATCH_SIZE, SIZE - PATCH_SIZE
    x1, x2 = PATCH_SIZE, SIZE - PATCH_SIZE

    mmx = SIZE/8

    for i in range(x1, x2, SKIP):
        for j in range(y1, y2, SKIP):
            sx = i - PATCH_SIZE/2
            sy = j - PATCH_SIZE/2
            patch[0] = np.divide(img[sx:sx + PATCH_SIZE, sy:sy+PATCH_SIZE], 255.0)
            prob = nn.predict_proba(patch)
            patch_probs[i, j] = prob[0, 1]
            bar.update(num)
            num += 1
            if num % 4000 == 0:
                print ("\n\nReloading network params!")
                nn.load_params_from("cachedgooglenn.params");

np.save(outfile, patch_probs)
plt.imsave(imgoutfile, patch_probs)

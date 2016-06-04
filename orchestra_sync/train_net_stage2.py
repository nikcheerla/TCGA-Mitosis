import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm

import cPickle as pickle
import os
import glob
import sys

import numpy as np
import csv
import random

from sklearn.utils import shuffle
from scipy.spatial import distance
from sklearn import cross_validation

from sknn.platform import gpu32

from nolearn_utils.iterators import (
    ShuffleBatchIteratorMixin,
    AffineTransformBatchIteratorMixin,
    RandomFlipBatchIteratorMixin,
    make_iterator
)

from nolearn_utils.hooks import (
    SaveTrainingHistory, PlotTrainingHistory,
    EarlyStopping
)

from nolearn.lasagne import BatchIterator
from sklearn.metrics import roc_auc_score

def roc_robust(y_true, y_proba):
    if sum(y_true) == 0 or sum(y_true) == len(y_true):
        return 0.0
    else:
        return roc_auc_score(y_true, y_proba)


print "Loading and morphing data"


# Constants
SIZE = 2084
PATCH_SIZE = 101
MAX_PATCH_SIZE = 101
PATCH_GAP = 50
RADIUS = 10

img = []
img2 = []
coords = []

img_aux = []
coords = []
num_images = 0
max_images = 296
img_file_loc = "mitoses_image_data"

for root, dirnames, filenames in os.walk('mitoses_aux_ground'):
    if (num_images >= max_images):
            break
    for filename in filenames:
        if filename[-3:] != "csv":
            break

        full_path_csv = os.path.join(root, filename);
        print full_path_csv,
        full_path_img = os.path.join(img_file_loc + root[-3:], filename[:-3] + "tif")
        print full_path_img,

        csvReader = csv.reader(open(full_path_csv))
        for row in csvReader:
            xv = int(row[0])
            yv = int(row[1])
            for di in range(-7, 7):
                for dj in range(-7, 7):
                    coords.append( ( xv + di, yv + dj, -num_images - 1))

        img_aux.append(plt.imread(full_path_img))
        num_images += 1
        if (num_images >= max_images):
            break

img_aux = np.array(img_aux)
print (img_aux.shape)

img = []

cnt = 0

for imgfile in glob.iglob("train/*.bmp"):
    print "\n" + imgfile,
    annotfile = imgfile[:-3] + "csv"
    img.append(plt.imread(imgfile))
    csvReader = csv.reader(open(annotfile, 'rb'))
    tot = 0
    imgMask = np.zeros((SIZE, SIZE))
    for row in csvReader:
        minx, miny, maxx, maxy = (SIZE, SIZE, 0, 0)
        random_coords = []
        for i in range(0, len(row) / 2):
            xv, yv = (int(row[2 * i]), int(row[2 * i + 1]))
            if xv > PATCH_SIZE / 2 + 1 and yv > PATCH_SIZE / 2 + 1 and xv < SIZE - PATCH_SIZE / 2 - 1 and yv < SIZE - PATCH_SIZE / 2 - 1:
                random_coords.append([yv, xv, cnt])

        centroid = np.array(random_coords).mean(axis=0).astype(int)
        print centroid,
        for i in range(0, len(row) / 2):
            xv, yv = (int(row[2 * i]), int(row[2 * i + 1]))
            if distance.euclidean([yv, xv, cnt], centroid) <= RADIUS:
                if xv > PATCH_SIZE / 2 + 1 and yv > PATCH_SIZE / 2 + 1 and xv < SIZE - PATCH_SIZE / 2 - 1 and yv < SIZE - PATCH_SIZE / 2 - 1:
                    coords.append((yv, xv, cnt))
                    tot = tot + 1
    cnt += 1
img = np.array(img)

print img.shape
print img_aux.shape
print len(coords)

def get_patches(coords, patchsize=PATCH_SIZE):
    patches = np.zeros((len(coords), patchsize, patchsize, 3))
    i = 0
    for (x, y, img_num) in coords:
        # print x, y
        #print (x - patchsize/2), (x + patchsize/2 + 1), (y - patchsize/2), (y + patchsize/2 + 1)
        if img_num >= 0:
            patches[i] = img[img_num, (x - patchsize / 2):(x + patchsize / 2 + 1),
                             (y - patchsize / 2):(y + patchsize / 2 + 1), :]
        else:
            patches[i] = img_aux[-img_num - 1, (x - patchsize / 2):(x + patchsize / 2 + 1),
                             (y - patchsize / 2):(y + patchsize / 2 + 1), :]
        patches[i] = np.divide(patches[i], 255.0)
        i += 1
    return patches

lookup = set(coords)

import random

def make_normal_set(length):
    norm_coords = []
    while len(norm_coords) < length:
        triple = (random.randint(MAX_PATCH_SIZE/2, 1999 - MAX_PATCH_SIZE/2 - 1),
                     random.randint(MAX_PATCH_SIZE/2, 1999 - MAX_PATCH_SIZE/2 - 1),
                     random.randint(-len(img_aux), len(img) - 1))
        if triple not in lookup:
            norm_coords.append(triple)
    return norm_coords

surplus = 500000 - len(coords)
trainVal = np.append(np.ones(len(coords)), np.zeros(surplus))
coords_normal = make_normal_set(surplus)

def get_patches(coords, patchsize=PATCH_SIZE):
    patches = np.zeros((len(coords), patchsize, patchsize, 3))
    i = 0
    for (x, y, img_num) in coords:
        # print x, y
        #print (x - patchsize/2), (x + patchsize/2 + 1), (y - patchsize/2), (y + patchsize/2 + 1)
        patches[i] = img[img_num, (x - patchsize / 2):(x + patchsize / 2 + 1),
                         (y - patchsize / 2):(y + patchsize / 2 + 1), :]
        patches[i] = np.divide(patches[i], 255.0)
        i += 1
    return patches


def get_visual_patches(coords, patchsize=PATCH_SIZE):
    patches = np.zeros((len(coords), patchsize, patchsize, 3))
    i = 0
    for (x, y, img_num) in coords:
        # print x, y
        #print (x - patchsize/2), (x + patchsize/2 + 1), (y - patchsize/2), (y + patchsize/2 + 1)
        patches[i] = img2[img_num, (x - patchsize / 2):(x + patchsize / 2 + 1),
                          (y - patchsize / 2):(y + patchsize / 2 + 1), :]
        patches[i] = np.divide(patches[i], 255.0)
        i += 1
    return patches


trainImg2, trainVal2 = shuffle(np.array(coords + coords_normal), trainVal)

print trainImg2.shape


print "Here, we train a Lasagne network to predict whether or not a pixel is mitotic."
print "This is a stage 2 network so there will be more training time and more accuracy needed."


"""Building Image Perturbation Models/Callbacks"""

train_iterator_mixins = [
    RandomFlipBatchIteratorMixin,
    AffineTransformBatchIteratorMixin,
]
TrainIterator = make_iterator('TrainIterator', train_iterator_mixins)

test_iterator_mixins = [
    RandomFlipBatchIteratorMixin,
]
TestIterator = make_iterator('TestIterator', test_iterator_mixins)

train_iterator_kwargs = {
    'batch_size': 20,
    'affine_p': 0.5,
    'affine_scale_choices': np.linspace(0.75, 1.25, 1.5),
    'flip_horizontal_p': 0.5,
    'flip_vertical_p': 0.5,
    'affine_translation_choices': np.arange(-5, 6, 1),
    'affine_rotation_choices': np.arange(-45, 50, 5),
}
train_iterator_tmp = TrainIterator(**train_iterator_kwargs)

test_iterator_kwargs = {
    'batch_size': 20,
    'flip_horizontal_p': 0.5,
    'flip_vertical_p': 0.5,
}
test_iterator_tmp = TestIterator(**test_iterator_kwargs)


class CustomBatchIterator(BatchIterator):

    def __init__(self, batch_size, built_iterator):
        super(CustomBatchIterator, self).__init__(batch_size=batch_size)
        self.iter = built_iterator

    def transform(self, Xb, yb):
        if sum(yb) == 0:
            Xb = Xb[0:1]
            yb = yb[0:1]
            Xb = get_patches(Xb)
            Xb = Xb.astype(np.float32).swapaxes(1, 3)
            yb = yb.astype(np.uint8)
            Xb, yb = self.iter.transform(Xb, yb)
            return Xb, yb

        num_normal = len(yb) - sum(yb)

        mitosis = np.nonzero(yb)

        while sum(yb) < num_normal:
            yb = np.append(yb, 1)
            Xb = np.append(Xb, [random.choice(Xb[mitosis])], axis=0)

        Xb = get_patches(Xb)
        Xb = Xb.astype(np.float32).swapaxes(1, 3)
        yb = yb.astype(np.uint8)
        Xb, yb = self.iter.transform(Xb, yb)
        return Xb, yb

train_iterator = CustomBatchIterator(
    batch_size=20, built_iterator=train_iterator_tmp)
test_iterator = CustomBatchIterator(
    batch_size=20, built_iterator=test_iterator_tmp)


class AdjustVariable(object):

    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)

import theano


def float32(k):
    return np.cast['float32'](k)


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
    custom_scores=[('auc', lambda y_true, y_proba: roc_robust(y_true, y_proba[:, 1]))],
    max_epochs=1,
    verbose=1,


    # batch iteration params
    batch_iterator_train=train_iterator,
    batch_iterator_test=test_iterator,
)


"""Loading data and training Lasagne network using nolearn"""

trainVal2 = trainVal2
print trainImg2.shape

print "Ratio: " + str(1.0 - float(sum(trainVal2)) / float(len(trainVal2)))

best_accuracy = 0.0
print "Training Classifier: 80/20 split"
for i in [1, 2, 3, 4, 6, 8, 10, 40, 100, 250]:
    saved_accuracy = 0.0
    print "Size: " + str(i*2000)
    for epoch in range(0, 25):
        nn = nn.fit(trainImg2[0:2000*i], trainVal2[0:2000*i])
        cur_accuracy = nn.train_history_[-1]['valid_accuracy']
        best_accuracy = max(cur_accuracy, best_accuracy)
        #print "Current Accuracy: " + str(cur_accuracy)
        #print "Saved Accuracy: " + str(saved_accuracy)
        if cur_accuracy + 0.04 < saved_accuracy or cur_accuracy + 0.12 < best_accuracy:
            print "Accuracy Drop! Loading previous network!"
            nn.load_params_from("cachednn.params")
        else:
            nn.save_params_to('cachednn.params')
            saved_accuracy = cur_accuracy

nn.save_params_to('nn_stage2.params')

#pickle.dump(nn, open( "nn_stage2.pkl", "wb" ))

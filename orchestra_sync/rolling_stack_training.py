from __future__ import print_function

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

import theano
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



### CONSTANTS ###

SIZE = 2084
SIZE2 = 2000
PATCH_SIZE = 101
MAX_PATCH_SIZE = 101
PATCH_GAP = 50
RADIUS = 10

N = 10000
K = 1.5
EPOCHS = 12
DECAY = 1.0
KGROWTH = 0.5
EGROWTH = 1.0

def inbounds(x, y):
    return x < SIZE2 - PATCH_SIZE and x > PATCH_SIZE and y < SIZE2 - PATCH_SIZE and y > PATCH_SIZE

def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a


def tolist(a):
    try:
        return list(totuple(i) for i in a)
    except TypeError:
        return a









### TODO 1: READ ALL DATA ###
## Final result: coords contains indexed coords of all mitotic cells ##

print("\n\nData Reading.")



img = []
img_aux = []
coords = []

num_images = 0
max_images = 0
img_file_loc = "mitoses_image_data2"

print("Reading in auxilary image data:")

for root, dirnames, filenames in os.walk('mitoses_aux_ground2'):
    if (num_images >= max_images):
            break
    for filename in filenames:
        if filename[-3:] != "csv":
            break

        full_path_csv = os.path.join(root, filename);
        print(full_path_csv, end=", ")
        full_path_img = os.path.join(img_file_loc + root[-3:], filename[:-3] + "tif")
        print(full_path_img, end=" ::: ")

        csvReader = csv.reader(open(full_path_csv))
        for row in csvReader:
            xv = int(row[0])
            yv = int(row[1])
            for di in range(-RADIUS - 1, RADIUS + 1):
                for dj in range(-RADIUS - 1, RADIUS + 1):
                    if distance.euclidean([di, dj, 0], [0, 0, 0]) <= RADIUS and inbounds(xv + di, yv + dj):
                        coords.append( ( xv + di, yv + dj, -num_images - 1))

        img_aux.append(plt.imread(full_path_img))
        num_images += 1
        if (num_images >= max_images):
            break

img_aux = np.array(img_aux)
print (img_aux.shape)

cnt = 0

print("\nReading in original image files:")
for imgfile in glob.iglob("train/*.bmp"):
    print("\n" + imgfile, end="")
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
        print(centroid, end="")
        for i in range(0, len(row) / 2):
            xv, yv = (int(row[2 * i]), int(row[2 * i + 1]))
            if distance.euclidean([yv, xv, cnt], centroid) <= RADIUS:
                if xv > PATCH_SIZE / 2 + 1 and yv > PATCH_SIZE / 2 + 1 and xv < SIZE - PATCH_SIZE / 2 - 1 and yv < SIZE - PATCH_SIZE / 2 - 1:
                    coords.append((yv, xv, cnt))
                    tot = tot + 1
    cnt += 1
img = np.array(img)

print(img.shape)
print(img_aux.shape)
print(len(coords))



def get_patches(coords, patchsize=PATCH_SIZE):
    patches = np.zeros((len(coords), patchsize, patchsize, 3))
    i = 0
    for (x, y, img_num) in coords:
        #print x, y
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















### TODO 2: CREATE AND DESIGN CNN ####
## Final result: net contains desired CNN ##





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
]
TrainIterator = make_iterator('TrainIterator', train_iterator_mixins)

test_iterator_mixins = [
    RandomFlipBatchIteratorMixin,
]
TestIterator = make_iterator('TestIterator', test_iterator_mixins)

train_iterator_kwargs = {
    'batch_size': 20,
    'affine_p': 0.5,
    'affine_scale_choices': np.linspace(0.75, 1.1, 1.25, 1.5),
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




def color_transform(image):
    if random.uniform(0.0, 1.0) < 0.15:
        image[0] = np.multiply(image[0], random.uniform(0.8, 1.15))
    if random.uniform(0.0, 1.0) < 0.15:
        image[1] = np.multiply(image[0], random.uniform(0.8, 1.15))
    if random.uniform(0.0, 1.0) < 0.15:
        image[1] = np.multiply(image[0], random.uniform(0.8, 1.15))
    return np.clip(image, -1.0, 1.0).astype(np.float32)





class CustomBatchIterator(BatchIterator):

    def __init__(self, batch_size, built_iterator):
        super(CustomBatchIterator, self).__init__(batch_size=batch_size)
        self.iter = built_iterator

    def transform(self, Xb, yb):
        Xb = get_patches(Xb)
        Xb = Xb.astype(np.float32).swapaxes(1, 3)
        for i in range(0, len(yb)):
            Xb[i] = color_transform(Xb[i])
        yb = yb.astype(np.uint8)
        Xb, yb = self.iter.transform(Xb, yb)
        return Xb, yb

train_iterator = CustomBatchIterator(
    batch_size=20, built_iterator=train_iterator_tmp)
test_iterator = CustomBatchIterator(
    batch_size=20, built_iterator=test_iterator_tmp)


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
    update_learning_rate=0.0048,
    update_momentum=0.83,
    #on_epoch_finished=[
    #    AdjustVariable('update_learning_rate', start=0.018, stop=0.0001),
    #    AdjustVariable('update_momentum', start=0.9, stop=1.09),
    #],
    max_epochs=1,
    verbose=1,


    # batch iteration params
    batch_iterator_train=train_iterator,
    batch_iterator_test=test_iterator,
)














### TODO 3: DEFINE METHODS TO WORK WITH NORMAL_STACKS ###
## Final result: update_stack(stack, times) ###

lookup = set(coords)
proba_before = 0.0
proba_after = 0.0
overlap = 0.0

def prob(coord, net):
    patch = get_patches([coord]).swapaxes(1, 3).astype(np.float32)
    patch2 = patch.swapaxes(2, 3)
    saved_iter = net.batch_iterator_test
    net.batch_iterator_test = BatchIterator(batch_size=1)
    prob = net.predict_proba(patch)[0, 1]
    prob2 = net.predict_proba(patch2)[0, 1]
    net.batch_iterator_test = saved_iter
    return (prob + prob2)/2.0

def create_stack(length):
    global lookup
    normal_stack = []
    while len(normal_stack) < length:
        triple = (random.randint(MAX_PATCH_SIZE/2, SIZE2 - MAX_PATCH_SIZE/2 - 1),
                     random.randint(MAX_PATCH_SIZE/2, SIZE2 - MAX_PATCH_SIZE/2 - 1),
                     random.randint(-len(img_aux), len(img) - 1))
        if triple not in lookup:
            normal_stack.append(triple)
            lookup.add(triple)
    return normal_stack

def update_stack(normal_stack, iters, net):
    global lookup, proba_before, proba_after, overlap
    init_len = len(normal_stack)
    probs = []
    for i in range(0, len(normal_stack)):
        probs.append(prob(normal_stack[i], net))
    proba_before = np.mean(probs)

    for i in range(0, iters):
        triple = (random.randint(MAX_PATCH_SIZE/2, SIZE2 - MAX_PATCH_SIZE/2 - 1),
                     random.randint(MAX_PATCH_SIZE/2, SIZE2 - MAX_PATCH_SIZE/2 - 1),
                     random.randint(-len(img_aux), len(img) - 1))
        if triple not in lookup:
            normal_stack.append(triple)
            probs.append(prob(triple, net))
            lookup.add(triple)

    sort_idx = np.argsort(probs)[::-1]
    normal_stack = np.array(normal_stack)[sort_idx, :]
    normal_stack = normal_stack[0:init_len]
    normal_stack = tolist(normal_stack)

    probs = np.array(probs)
    probs = probs[sort_idx]
    probs = probs[0:init_len]

    proba_after = np.mean(probs)

    overlap = 0.0
    for i in sort_idx[0:init_len]:
        if i < init_len:
            overlap += 1.0

    overlap /= init_len

    return normal_stack



















### TODO 4: Train network on normal stacks ###
## Final result: done! ###

print("\n\nTraining Network!")

normal_stack = create_stack(N)

for i in range(0, 1000):
    saved_accuracy = 0.0
    data = np.array(normal_stack + random.sample(coords, N))
    val = np.append(np.zeros(N), np.ones(N))
    data, val = shuffle(data, val)
    for i in range(0, int(EPOCHS)):
        nn.fit(data, val)
        cur_accuracy = nn.train_history_[-1]['valid_accuracy']
        if cur_accuracy + 0.08 < saved_accuracy:
            print("Accuracy Drop! Loading previous network!")
            nn.load_params_from("cachednn.params")
        else:
            nn.save_params_to('cachednn.params')
            saved_accuracy = cur_accuracy
        nn.update_learning_rate *= DECAY

    normal_stack = update_stack(normal_stack, int(K*N), nn)

    print("Data Report: K={3:.2f}, Prob Before={0}, Prob After={1}, Overlap={2}".format(proba_before, proba_after, overlap, K))
    nn.train_history_ = []
    K += KGROWTH
    EPOCHS *= EGROWTH


nn.save_params_to('rollingnn.params')

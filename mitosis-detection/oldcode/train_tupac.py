from __future__ import print_function

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm

import cPickle as pickle
import sys, os, glob, csv, random

import numpy as np

from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score
from sklearn import cross_validation
from scipy.spatial import distance
from scipy.misc import imresize

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

import spams
from staining.stainingController import Controller
from staining.method.macenko import macenko

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout



lookup, proba_before, proba_after, overlap = set(), 0, 0, 0

def train_tupac(params_dict):
    global lookup, proba_before, proba_after, overlap

    ### CONSTANTS ###

    SIZE = params_dict['size']
    PATCH_SIZE = params_dict['patch_size']
    PATCH_GAP = int(PATCH_SIZE/2)
    RADIUS = params_dict['radius']
    normalization = params_dict['normalization']

    net_name = params_dict['net_name']

    print ("PATCH_SIZE: ", PATCH_SIZE)
    print ("Network name: ", net_name)



    N = params_dict['N']
    MN = params_dict['MN']
    K = params_dict['K']
    EPOCHS = params_dict['epochs']
    DECAY = params_dict['decay']
    KGROWTH = params_dict['kgrowth']
    EGROWTH = params_dict['egrowth']

    VALID = 4000

    def inbounds(x, y):
        return x < SIZE - PATCH_SIZE and x > PATCH_SIZE and y < SIZE - PATCH_SIZE and y > PATCH_SIZE

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

    total_coords = []

    cnt = 0

    print("\nReading in original image files:")
    for imgfile in glob.iglob("data/train_tupac/*.jpg"):
        print("\n" + imgfile, end="")
        annotfile = imgfile[:-3] + "csv"
        img_vals = plt.imread(imgfile)

        if normalization:
            cntr = Controller(img)
            img_norm, _, __, __ = macenko(cntr)
            img.append(img_norm)
        else:
            img.append(img_vals)

        csvReader = csv.reader(open(annotfile, 'rb'))
        for row in csvReader:
            minx, miny, maxx, maxy = (SIZE, SIZE, 0, 0)
            random_coords = []
            for i in range(0, len(row) / 2):
                xv, yv = (int(row[2 * i]), int(row[2 * i + 1]))
                if xv > PATCH_SIZE / 2 + 1 and yv > PATCH_SIZE / 2 + 1 and xv < SIZE - PATCH_SIZE / 2 - 1 and yv < SIZE - PATCH_SIZE / 2 - 1:
                    coords.append((yv, xv, cnt))
                    total_coords.append((yv, xv, cnt))

        cnt += 1

    print("\n")
    print('Num images: ', len(img))
    print(len(coords))

    print('not synthesizing image through reflection')


    def get_patches(coords, patchsize=PATCH_SIZE):
        patches = np.zeros((len(coords), patchsize, patchsize, 3))
        i = 0
        for (x, y, img_num) in coords:
            #print x, y
            #print (x - patchsize/2), (x + patchsize/2 + 1), (y - patchsize/2), (y + patchsize/2 + 1)
            patches[i] = img[img_num, (x - patchsize / 2):(x + patchsize / 2 + 1),
                             (y - patchsize / 2):(y + patchsize / 2 + 1), :]
            patches[i] = np.divide(patches[i], 255.0)
            i += 1
        return patches


























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
        'affine_scale_choices': np.linspace(start=0.85, stop=1.6, num=10),
        'flip_horizontal_p': 0.5,
        'flip_vertical_p': 0.5,
        'affine_translation_choices': np.arange(-5, 6, 1),
        'affine_rotation_choices': np.linspace(start=-30.0, stop=30.0, num=20),
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




    def color_transform(image):
        if random.uniform(0.0, 1.0) < 0.15:
            image[0] = np.multiply(image[0], random.uniform(0.95, 1.05))
        if random.uniform(0.0, 1.0) < 0.15:
            image[1] = np.multiply(image[1], random.uniform(0.95, 1.05))
        if random.uniform(0.0, 1.0) < 0.15:
            image[2] = np.multiply(image[2], random.uniform(0.95, 1.05))
        return np.clip(image, -1.0, 1.0).astype(np.float32)


    radius = PATCH_GAP
    kernel = np.zeros((2*radius+1, 2*radius+1))
    y,x = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = x**2 + y**2 >= radius**2


    class CustomBatchIterator(BatchIterator):

        def __init__(self, batch_size, built_iterator):
            super(CustomBatchIterator, self).__init__(batch_size=batch_size)
            self.iter = built_iterator

        def transform(self, Xb, yb):
            Xb = get_patches(Xb)
            Xb = Xb.astype(np.float32).swapaxes(1, 3)
            for i in range(0, len(yb)):
                Xb[i] = color_transform(Xb[i])
                for c in range(0, 3):
                    Xb[i, c][mask] = 0.0
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
    net = phf.build_GoogLeNet(PATCH_SIZE, PATCH_SIZE)

























    ### TODO 3: DEFINE METHODS TO WORK WITH NORMAL_STACKS ###
    ## Final result: update_stack(stack, times) ###




    proba_before = 0.0
    proba_after = 0.0
    overlap = 0.0

    def prob(coord, net):
        patch = get_patches([coord]).swapaxes(1, 3).astype(np.float32)
        patch2 = patch.swapaxes(2, 3)
        saved_iter = net.batch_iterator_test
        net.batch_iterator_test = test_iterator_tmp
        prob = net.predict_proba(patch)[0, 1]
        prob2 = net.predict_proba(patch2)[0, 1]
        net.batch_iterator_test = saved_iter
        return (prob + prob2)/2.0

    def create_stack(length):
        global lookup
        normal_stack = []
        while len(normal_stack) < length:
            triple = (random.randint(PATCH_SIZE/2, SIZE - PATCH_SIZE/2 - 1),
                         random.randint(PATCH_SIZE/2, SIZE - PATCH_SIZE/2 - 1),
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
            triple = (random.randint(PATCH_SIZE/2, SIZE - PATCH_SIZE/2 - 1),
                         random.randint(PATCH_SIZE/2, SIZE - PATCH_SIZE/2 - 1),
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



    lookup = set(total_coords)

    coords = shuffle(coords)
    valid_sample_mitosis = coords[0:(VALID/2)]
    coords = coords[(VALID/2):(len(coords))]
    valid_sample_normal = create_stack(VALID/2)

    valid_sample = valid_sample_mitosis + valid_sample_normal

    valid_sample_y = np.append(np.ones(VALID/2), np.zeros(VALID/2))

    lookup = set(np.append(total_coords, valid_sample_normal))

    def get_validation(train_X, train_y, net):
        return train_X, valid_sample, train_y, valid_sample_y



    nn = NeuralNet(
        net['softmax'],
        max_epochs=1,
        update=adam,
        update_learning_rate=.0001, #start with a really low learning rate
        #objective_l2=0.0001,

        # batch iteration params
        batch_iterator_train=train_iterator,
        batch_iterator_test=test_iterator,

        train_split=get_validation,
        verbose=3,
    )






















    ### TODO 4: Train network on normal stacks ###
    ## Final result: done! ###

    #print('\nLoading Data from Previous Network')

    #nn.load_params_from("cachedgooglenn2.params")

    print("\n\nTraining Network!")

    normal_stack = create_stack(N)

    print("Made stack!")

    for k in range(0, 1000):
        saved_accuracy = 10011.0
        print ("Length of coords:", len(coords), "length of sample", MN)
        data = np.array(normal_stack + random.sample(coords, MN))
        val = np.append(np.zeros(N), np.ones(MN))
        data, val = shuffle(data, val)
        for i in range(0, int(EPOCHS)):
            nn.fit(data, val)
            cur_accuracy = nn.train_history_[-1]['valid_loss']
            if cur_accuracy - 0.004 > saved_accuracy:
                print("Test Loss Jump! Loading previous network!")
                with suppress_stdout():
                    nn.load_params_from("data/" + str(net_name) + ".params")
            else:
                nn.save_params_to("data/" + net_name + ".params")
                saved_accuracy = cur_accuracy
            nn.update_learning_rate *= DECAY

        normal_stack = update_stack(normal_stack, int(K*N), nn)

        print("Data Report: K={3:.2f}, Prob Before={0}, Prob After={1}, Overlap={2}".format(proba_before, proba_after, overlap, K))

        accuracy = nn.train_history_[-1]['valid_accuracy']
        nn.save_params_to('checkpoints/' + str(net_name) + '-checkpoint' + str(k) + '-validacc' + str(accuracy) + '.params')

        K += KGROWTH
        EPOCHS *= EGROWTH
        for r in range(len(nn.train_history_)):
            nn.train_history_[r]['train_loss'] = 10011.0

        
    nn.save_params_to("data/" + str(net_name) + ".params")

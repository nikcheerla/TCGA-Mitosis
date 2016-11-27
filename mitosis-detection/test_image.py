import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import cPickle as pickle
import sys, os, glob, csv, random


import numpy as np

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import progressbar

import lasagne
import PosterExtras as phf

from nolearn.lasagne import NeuralNet, TrainSplit
from lasagne.updates import adam
from nolearn.lasagne import BatchIterator

from sklearn.utils import shuffle
from scipy.spatial import distance
from sklearn import cross_validation
from sklearn.metrics import roc_auc_score

from contextlib import contextmanager

from nuclei_detect import nuclei_detect_pipeline

import spams
from staining.stainingController import Controller
from staining.method.macenko import macenko

def limit(num, minn, maxx):
    return min(max(num, minn), maxx)

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

# Constants
PATCH_SIZE = 159
PATCH_GAP = int(PATCH_SIZE/2)
RADIUS = 10

radius = PATCH_GAP
kernel = np.zeros((2*radius+1, 2*radius+1))
y,x = np.ogrid[-radius:radius+1, -radius:radius+1]
mask = x**2 + y**2 >= radius**2

normalized = False

print ("Local variables: ", locals())


# command line args

imgfile = sys.argv[1]
print "IMAGE: " + imgfile
outfile = imgfile[:-4] + ".out"
imgoutfile = imgfile[:-4] + ".png"
netfile = sys.argv[2]



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

nn.load_params_from(netfile);

img = plt.imread(imgfile);

SIZE = img.shape[0]

if normalized:
    cntr = Controller(img)
    img, _, __, __ = macenko(cntr)

print ("making nuclei map")
nimg, nuclei_map = nuclei_detect_pipeline(img)
print ('created')

print('synthesizing image through reflection')
img2 = np.append(img, np.append(img, img, axis = 0), axis = 0)
img3 = np.append(img2, np.append(img2, img2, axis = 1), axis = 1)
print('synthesized -- ', img3.shape)

def get_patches(coords, patchsize=PATCH_SIZE):
    patches = np.zeros((len(coords), patchsize, patchsize, 3))
    i = 0
    for (x, y) in coords:
        x += SIZE
        y += SIZE
        #print x, y
        #print (x - patchsize/2), (x + patchsize/2 + 1), (y - patchsize/2), (y + patchsize/2 + 1)
        patches[i] = img3[(x - patchsize / 2):(x + patchsize / 2 + 1),
                         (y - patchsize / 2):(y + patchsize / 2 + 1), :]
        patches[i] = np.divide(patches[i], 255.0)
        i += 1
    return patches


patch_probs = np.zeros((SIZE, SIZE));
patch_probs = patch_probs.astype(np.float32);

SKIP = 3;

num = 0;

patch = np.zeros((1, PATCH_SIZE, PATCH_SIZE, 3))
patch = patch.astype(np.float32)

y1, y2 = 1, SIZE - 1
x1, x2 = 1, SIZE - 1

coords = []

annotfile = imgfile[:-3] + "csv"
csvReader = csv.reader(open(annotfile, 'rb'))
tot = 0
imgMask = np.zeros((SIZE, SIZE))
for row in csvReader:
    minx, miny, maxx, maxy = (SIZE, SIZE, 0, 0)
    random_coords = []
    for i in range(0, len(row) / 2):
        xv, yv = (int(row[2 * i]), int(row[2 * i + 1]))
        if xv > PATCH_SIZE / 2 + 1 and yv > PATCH_SIZE / 2 + 1 and xv < SIZE - PATCH_SIZE / 2 - 1 and yv < SIZE - PATCH_SIZE / 2 - 1:
            random_coords.append([xv, yv])

    centroid = np.array(random_coords).mean(axis=0).astype(int)
    print (centroid)
    for i in range(0, len(row) / 2):
        xv, yv = (int(row[2 * i]), int(row[2 * i + 1]))
        if distance.euclidean([yv, xv], centroid) <= RADIUS:
            if xv > PATCH_SIZE / 2 + 1 and yv > PATCH_SIZE / 2 + 1 and xv < SIZE - PATCH_SIZE / 2 - 1 and yv < SIZE - PATCH_SIZE / 2 - 1:
                coords.append((xv, yv))
                tot = tot + 1



print ("First nuclei map " + str(nuclei_map[0, 0]))
with progressbar.ProgressBar(max_value=(2084/SKIP*2084/SKIP + 1)) as bar:
    patches = []
    for i in range(x1, x2, SKIP):
        for j in range(y1, y2, SKIP):
            bar.update(num)
            num += 1

            sx = i - PATCH_SIZE/2
            sy = j - PATCH_SIZE/2

            cover = False
            for dx in range(-6, 6):
                for dy in range(-6, 6):
                    if nuclei_map[limit(i + dx, 0, SIZE - 1), limit(j + dy, 0, SIZE - 1)]:
                        cover = True

            for xx, yy in coords:
                if i == xx and j == yy:
                    print ("Found coordinate of mitosis: ", (xx, yy))
                    print ("Cover = ", cover)

            if cover:
                patches.append((i, j))
                if len(patches) >= 500:
                    print ("Evaluating!")
                    patches2 = get_patches(patches)
                    prob = nn.predict_proba(patches2)
                    for k in range(0, len(patches)):
                        sx, sy = patches[k]
                        patch_probs[sx, sy] = prob[k, 1]

                        for xx, yy in coords:
                            if sx == xx and sy == yy:
                                print ("Found coordinate of mitosis: ", (xx, yy))
                                print ("Prob = ", prob[k, 1])
                    patches = []
                    with suppress_stdout():
                        nn.load_params_from(netfile);

print ("Evaluating!")
patches2 = get_patches(patches)
prob = nn.predict_proba(patches2)
for k in range(0, len(patches)):
    sx, sy = patches[k]
    patch_probs[sx, sy] = prob[k, 1]

    for xx, yy in coords:
        if sx == xx and sy == yy:
            print ("Found coordinate of mitosis: ", (xx, yy))
            print ("Prob = ", prob[k, 1])
patches = []

np.save(outfile, patch_probs)
plt.imsave(imgoutfile, patch_probs)

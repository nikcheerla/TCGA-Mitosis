import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import cPickle as pickle
import os
import glob
import sys


import numpy as np
import random
from heapq import heappush, heappop, heappushpop, nlargest, heapify

from sklearn.utils import shuffle
from scipy.spatial import distance

import csv



sys.setrecursionlimit(80000)





print "Here, we train a Lasagne network to predict whether or not a pixel is mitotic."
print "This is a stage 2 network so there will be more training time and more accuracy needed."


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





# Constants
SIZE = 2084
PATCH_SIZE = 101
PATCH_GAP = 50
RADIUS = 10

img = []
coords = []

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
print len(coords)






def get_patches(coords, patchsize=PATCH_SIZE):
    patches = np.zeros((len(coords), patchsize, patchsize, 3))
    i = 0
    heap = []
    for (x, y, img_num) in coords:
        # print x, y
        #print (x - patchsize/2), (x + patchsize/2 + 1), (y - patchsize/2), (y + patchsize/2 + 1)
        patches[i] = img[img_num, (x - patchsize / 2):(x + patchsize / 2 + 1),
                         (y - patchsize / 2):(y + patchsize / 2 + 1), :]
        patches[i] = np.divide(patches[i], 255.0)
        i += 1
    return patches







nn = pickle.load(open("nn_stage1.pkl", "rb"))

def make_normal_difficult(length, iters, nn_ref):
    lookup = set(coords)
    norm_coords = []
    heap_norm_coords = []
    print "Initial Seeding Stage:"
    while len(heap_norm_coords) < length:
        #print len(heap_norm_coords)
        #print heap_norm_coords
        if len(heap_norm_coords) % 2000 == 0:
            print str(len(heap_norm_coords)) + "/" + str(length)
            sys.stdout.flush()
        triple = (random.randint(PATCH_SIZE / 2, SIZE - PATCH_SIZE / 2 - 1),
                  random.randint(PATCH_SIZE / 2, SIZE - PATCH_SIZE / 2 - 1),
                  random.randint(0, len(img) - 1))
        if triple not in lookup:
            patch = get_patches([triple]).swapaxes(1, 3).astype(np.float32)
            prob = nn_ref.predict_proba(patch)[0, 1]
            heappush(heap_norm_coords, (prob, triple))
            #print triple
            #print len(heap_norm_coords)
            #print heap_norm_coords
            lookup.add(triple)

    for i in range(0, iters):
        #print len(heap_norm_coords)
        #print heap_norm_coords
        if i % 5000 == 0:
            print str(i) + "/" + str(iters) + " with lower bound " + str(heap_norm_coords[0][0])
        triple = (random.randint(PATCH_SIZE / 2, SIZE - PATCH_SIZE / 2 - 1),
                  random.randint(PATCH_SIZE / 2, SIZE - PATCH_SIZE / 2 - 1),
                  random.randint(0, len(img) - 1))
        if triple not in lookup:
            patch = get_patches([triple]).swapaxes(1, 3).astype(np.float32)
            prob = nn_ref.predict_proba(patch)[0, 1]
            #print len(heap_norm_coords)
            #print heap_norm_coords
            heappushpop(heap_norm_coords, (prob, triple))
            #print len(heap_norm_coords)
            #print heap_norm_coords
            lookup.add(triple)
    #print heap_norm_coords
    print len(heap_norm_coords)
    for i in range(0, length):
        smallest = heappop(heap_norm_coords)
        norm_coords.append(smallest[1])
    return norm_coords

print make_normal_difficult(5, 50, nn)


surplus = 1000000 - len(coords)

trainCoords = coords + make_normal_difficult(surplus, 40*surplus, nn)
np.save("intermediate_data.npy", np.array(trainCoords))

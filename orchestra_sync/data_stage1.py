

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import cPickle as pickle
import os
import glob
import sys

sys.setrecursionlimit(80000)

import numpy as np

from sklearn.utils import shuffle
from scipy.spatial import distance
from sklearn import cross_validation

import csv

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
    imgMask = np.zeros((SIZE, SIZE));
    for row in csvReader:
        minx, miny, maxx, maxy = (SIZE, SIZE, 0, 0)
        random_coords = []
        for i in range(0, len(row)/2):
            xv, yv = (int(row[2*i]), int(row[2*i+1]))
            if xv > PATCH_SIZE/2 + 1 and yv > PATCH_SIZE/2 + 1 and xv < SIZE - PATCH_SIZE/2 - 1 and yv < SIZE - PATCH_SIZE/2 - 1:
                random_coords.append([yv, xv, cnt])

        centroid = np.array(random_coords).mean(axis=0).astype(int)
        print centroid,
        for i in range(0, len(row)/2):
            xv, yv = (int(row[2*i]), int(row[2*i+1]))
            if distance.euclidean([yv, xv, cnt], centroid) <= RADIUS:
                if xv > PATCH_SIZE/2 + 1 and yv > PATCH_SIZE/2 + 1 and xv < SIZE - PATCH_SIZE/2 - 1 and yv < SIZE - PATCH_SIZE/2 - 1:
                    coords.append((yv, xv, cnt))
                    tot = tot + 1
    cnt += 1
img = np.array(img)

print img.shape
print len(coords)

def get_patches(coords, patchsize = PATCH_SIZE):
    patches = np.zeros((len(coords), patchsize, patchsize, 3))
    i = 0
    for (x, y, img_num) in coords:
        #print x, y
        #print (x - patchsize/2), (x + patchsize/2 + 1), (y - patchsize/2), (y + patchsize/2 + 1)
        patches[i] = img[img_num, (x - patchsize/2):(x + patchsize/2 + 1), (y - patchsize/2):(y + patchsize/2 + 1),:]
        patches[i] = np.divide(patches[i], 255.0)
        i += 1
    return patches


import random

def make_normal_set(length):
    lookup = set(coords)
    norm_coords = []
    while len(norm_coords) < length:
        triple = (random.randint(PATCH_SIZE/2, SIZE - PATCH_SIZE/2 - 1),
                     random.randint(PATCH_SIZE/2, SIZE - PATCH_SIZE/2 - 1),
                     random.randint(0, len(img) - 1))
        if triple not in lookup:
            norm_coords.append(triple)
    return norm_coords

trainCoords = coords + make_normal_set(len(coords))
trainVal = np.append(np.ones(len(coords)), np.zeros(len(coords)))
trainCoords, trainVal2 = shuffle(trainCoords, trainVal)

trainImg2 = get_patches(trainCoords)

print(trainVal2)

np.save("trainImg_stage1.npy", trainImg2)
np.save("trainVal_stage1.npy", trainVal2)

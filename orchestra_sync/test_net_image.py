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

import csv
import subprocess

from scipy.signal import convolve
from scipy.spatial import distance

# Constants
SIZE = 2084
PATCH_SIZE = 101
PATCH_GAP = 50
RADIUS = 10


nn = pickle.load(open("nn_stage2.pkl", "rb"))

img = plt.imread(image_file)
patch_probs = np.zeros(SIZE, SIZE)

patch = np.zeros((1, PATCH_SIZE, PATCH_SIZE, 3))
patch = patch.astype(np.float32)
patch = np.swapaxes(patch, 1, 3)

y1, y2 = PATCH_GAP + 1, SIZE - PATCH_GAP - 1
x1, x2 = PATCH_GAP + 1, SIZE - PATCH_GAP - 1

num = 0
with progressbar.ProgressBar(max_value=2084*2084) as bar:
    for i in range(x1, x2):
        for j in range(y1, y2):
    	#if not nuclei_map[i, j]:
    	#	num += 1
    	#	bar.update(num)
    	#	continue
            sx = i - PATCH_SIZE/2
            sy = j - PATCH_SIZE/2
            patch = np.swapaxes(patch, 1, 3)
            patch[0] = np.divide(img[sx:sx + PATCH_SIZE, sy:sy+PATCH_SIZE], 255.0)
            patch = np.swapaxes(patch, 1, 3)
            with suppress_stdout():
                prob = nn.predict_proba(patch)
            patch_probs[i, j] = prob[0, 1]
        	bar.update(num)
        	num += 1

npy.save(outfile, patch_probs)
print patch_probs
print patch_probs.size

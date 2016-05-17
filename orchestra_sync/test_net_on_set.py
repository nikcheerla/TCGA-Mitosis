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
import bisect




# Constants
SIZE = 2084
PATCH_SIZE = 101
PATCH_GAP = 50
RADIUS = 10

images = glob.glob("test/*.bmp")

for image in images:
    print subprocess.check_output(["bsub", "-n", "3", "-q", "mcore", "-W", "120:00", "-R", "rusage[mem=10000]", "-o", image[:-4] + "_prgmdata.out",
        "-e", image[:-4] + "_prgmdata.err", "python", "-u", "img_read.py", image])

radius = RADIUS
kernel = np.zeros((2*radius+1, 2*radius+1))
y,x = np.ogrid[-radius:radius+1, -radius:radius+1]
mask = x**2 + y**2 <= radius**2
kernel[mask] = 1

sys.exit(0);

centroid_coords = []
cnt = 0
for imgfile in glob.iglob("test/*.bmp"):
    print "\n" + imgfile,
    annotfile = imgfile[:-3] + "csv"
    csvReader = csv.reader(open(annotfile, 'rb'))
    tot = 0
    for row in csvReader:
        minx, miny, maxx, maxy = (SIZE, SIZE, 0, 0)
        random_coords = []
        for i in range(0, len(row) / 2):
            xv, yv = (int(row[2 * i]), int(row[2 * i + 1]))
            if xv > PATCH_SIZE / 2 + 1 and yv > PATCH_SIZE / 2 + 1 and xv < SIZE - PATCH_SIZE / 2 - 1 and yv < SIZE - PATCH_SIZE / 2 - 1:
                random_coords.append([yv, xv, cnt])
        centroid = np.array(random_coords).mean(axis=0).astype(int)
        centroid_coords.append(centroid)
        print centroid,

patch_probs = np.zeros((len(images), SIZE, SIZE))
i = 0
for image in images:
    image_data_file = image[:-4] + ".npy"
    probs = np.load(image_data_file)
    patch_probs[i] = convolve(probs, kernel)
    i += 1

centroid_coords_found = []
probs = []
while len(centroid_coords_found) < 1000:
    max_element = np.unravel_index(patch_probs.argmax(), patch_probs.shape)
    prob = patch_probs[max_element]
    img_num, i, j = max_element
    patch_probs[img_num, i - RADIUS:i + RADIUS + 1, j - RADIUS:j + RADIUS + 1] = np.zeros((2*RADIUS + 1, 2*RADIUS + 1))
    centroid_coords_found.append(max_element)
    probs.append(prob)


MARGIN = 30
def f1score(thresh):
    found = 0
    idx = bisect_left(probs, thresh)
    print idx
    centroid_coords_narrow = centroid_coords_found[0:idx]
    for coord in centroid_coords:
        for coord2 in centroid_coords_narrow:
            if distance.euclidean(coord, coord2) <= MARGIN:
                found += 1
    precision = float(found)/float(len(centroid_coords))
    recall = float(found)/float(len(centroid_coords_narrow))
    f1 = 2*precision*recall/(precision + recall)
    return f1

print f1score(0.25)
print f1score(0.5)
print f1score(0.75)

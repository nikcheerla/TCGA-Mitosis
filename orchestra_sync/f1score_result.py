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

count1 = 0
len1 = 0
len2 = 0

centroid_coords = []
cnt = 0
images = sorted(glob.glob("test/*.bmp"))


# Constants
SIZE = 2084
PATCH_SIZE = 101
PATCH_GAP = 50
RADIUS = 10

cnt = 0

for imgfile in images:
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
        try:
            print centroid[0:3],
            centroid_coords.append(centroid)
        except:
            print random_coords
    cnt += 1

radius = 4
kernel = np.zeros((2*radius+1, 2*radius+1))
y,x = np.ogrid[-radius:radius+1, -radius:radius+1]
mask = x**2 + y**2 <= radius**2
kernel[mask] = 1
cnt = 0

points = []
for thresh in np.linspace(0.2,0.95,25,endpoint=False):

    count1 = 0
    len1 = 0
    len2 = 0
    cnt = 0

    for image in images:
        image_data_file = image[:-4] + ".out.npy"
        patch_probs = np.load(image_data_file)
        centroid_coords = np.array(centroid_coords)
        centroid_coords_filtered = centroid_coords[centroid_coords [:, 2] == cnt]
        #print centroid_coords_filtered
        count = 0
        for x, y, img_num in centroid_coords_filtered:
            foundMitosis = False
            for i in range(-30, 30):
                if foundMitosis:
                    break
                for j in range(-30, 30):
                    if foundMitosis:
                        break
                    if patch_probs[x + i, y + j] > thresh:
                        count += 1
                        foundMitosis = True
        #print "Count " + str(count)
        patch_probs_convolve = convolve(patch_probs, kernel)
        centroid_coords_found = []
        probs = []
        while len(probs) == 0 or probs[-1] > thresh:
            max_element = np.unravel_index(patch_probs_convolve.argmax(), patch_probs_convolve.shape)
            prob = patch_probs[max_element]
            i, j = max_element
            patch_probs_convolve[i - RADIUS:i + RADIUS + 1, j - RADIUS:j + RADIUS + 1] = np.zeros((2*RADIUS + 1, 2*RADIUS + 1))
            centroid_coords_found.append(max_element)
            probs.append(prob)
        count1 += count
        len1 += len(probs) - 1
        len2 += len(centroid_coords_filtered)
        print len2
        cnt += 1
    fpz = count1/float(len1)
    tpz = count1/float(len2)
    f1score = 2.0*fpz*tpz/(fpz + tpz)
    print "\n\nF1 Score: " + str(f1score) + " for threshhold " + str(thresh)
    points.append(f1score)

print points
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
RESULT_RADIUS = 20

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

#image_data_file = "results/P100/googlenet/stage01.old/%s.png.npy"%(images_root)
#image_data_file = "results/P050/googlenet/stage01/%s.png.npy"%(images_root)
#result_root = 'data/images/test'
result_root = "image_results"

points = []
for thresh in np.linspace(5.2, 7.5, 40,endpoint=False):
#for thresh in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,0.9,0.95]:
#for thresh in [0.8,0.9,0.95]:
#for thresh in [0.8]:
    TPs, FPs, FNs = 0, 0, 0
    len1, len2 = 0, 0
    for image_ID, image in enumerate(images):
        print image_ID, image

        images_root = image.split('/')[-1][:-4]
        image_data_file = "%s/%s.out.npy"%(result_root, images_root)

        patch_probs = np.load(image_data_file)
        centroid_coords = np.array(centroid_coords)
        centroid_coords_filtered = centroid_coords[centroid_coords [:, 2] == image_ID]

        # get a smooth heatmap
        patch_probs_convolve = convolve(patch_probs, kernel)

        # find all positive points
        centroid_coords_found, probs = [], []
        while len(probs) == 0 or probs[-1] > thresh:
            max_element = np.unravel_index(patch_probs_convolve.argmax(), patch_probs_convolve.shape)
            prob = patch_probs_convolve[max_element]
            y, x = max_element
            patch_probs_convolve[y - RADIUS:y + RADIUS + 1, x - RADIUS:x + RADIUS + 1] = np.zeros((2*RADIUS + 1, 2*RADIUS + 1))
            centroid_coords_found.append(max_element)
            probs.append(prob)
        centroid_coords_found = centroid_coords_found[:-1]
        probs = probs[:-1]
        print "\tDetected Points:%d True Points:%d Threshold:%.2f"%(len(centroid_coords_found), len(centroid_coords_filtered), thresh)

        # find all Truth Positive
        TP, FP = 0, 0
        marker = np.zeros((len(centroid_coords_filtered), 1))
        for detected_centroid_id, centroid in enumerate(centroid_coords_found):
            IS_A_MITOSIS = False
            dis_all = []
            for ground_truth_centroid_id, (y, x, img_num) in enumerate(centroid_coords_filtered):
                disv = distance.euclidean(centroid, [y, x])
                dis_all.append(disv)
                if disv <= RESULT_RADIUS:
                    if marker[ground_truth_centroid_id] == 0:
                        marker[ground_truth_centroid_id] = 1
                        IS_A_MITOSIS = True
                        TP += 1
                        print "\t\t (%d) Find a new mitosis: %.3f"%(detected_centroid_id, disv)
                    else:
                        IS_A_MITOSIS = True
                        print "\t\t (%d) Find an existisng mitosis: %.3f"%(detected_centroid_id, disv)
            if not IS_A_MITOSIS:
                print "\t\t (%d) Find a wrong mitosis:%.3f"%(detected_centroid_id, np.min(dis_all))
                FP += 1
        FN = len(centroid_coords_filtered) - TP
        print "\tTP=%d, FP=%d, FN=%d + other lengths:"%(TP, FP, FN), len(probs), len(centroid_coords_filtered)

        TPs += TP
        FPs += FP
        FNs += FN
        len1 += len(probs)
        len2 += len(centroid_coords_filtered)
    print TPs + FPs, TPs + FNs, len1, len2

    precision = TPs / float(TPs + FPs)
    recall = TPs / float(TPs + FNs)
    f1score = (2.0 * precision * recall) / (precision + recall)
    #fpz = TPs/float(len1)
    #tpz = TPs/float(len2)
    #f1score = 2.0*fpz*tpz/(fpz + tpz)

    print "\n\nF1 Score: " + str(f1score) + " for threshhold " + str(thresh)
    points.append(f1score)
print points

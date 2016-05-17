from flask import render_template, url_for, send_file
from app import app


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

SIZE = 2084
PATCH_SIZE = 101
PATCH_GAP = 50
RADIUS = 10

img = []
coords = []

def get_patches(coords, patchsize=PATCH_SIZE):
    patches = np.zeros((len(coords), patchsize, patchsize, 3))
    i = 0
    heap = []
    for i in range(0, len(coords)):
        dpair = coords[i]
        print dpair
        x = dpair[0]
        y = dpair[1]
        img_num = dpair[2]
        # print x, y
        #print (x - patchsize/2), (x + patchsize/2 + 1), (y - patchsize/2), (y + patchsize/2 + 1)
        patches[i] = img[img_num, (x - patchsize / 2):(x + patchsize / 2 + 1),
                         (y - patchsize / 2):(y + patchsize / 2 + 1), :]
        patches[i] = np.divide(patches[i], 255.0)
    return patches

# Constants

def get_images():
    global img, coords
    img = []
    coords = []
    cnt = 0
    for imgfile in glob.iglob("../train/*.bmp"):
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

@app.route('/')
@app.route('/index')
def index():
    get_images()
    user = {'nickname': 'Miguel'}  # fake user
    return render_template('index.html', title='Home', user=user)

@app.route('/mitosis')
def random_mitosis():
    patch = get_patches([random.choice(coords)])[0]
    print patch
    plt.imsave("app/static/patch.png", patch)
    print "Saved Patch (Hopefully)"
    return send_file("static/patch.png")

@app.route('/normal')
def random_normal():
    normal_coords = np.load("../orchestra_sync/intermediate_data.npy");
    patch = get_patches([random.choice(normal_coords)])[0]
    print patch
    plt.imsave("app/static/patch.png", patch)
    print "Saved Patch (Hopefully)"
    return send_file("static/patch.png")

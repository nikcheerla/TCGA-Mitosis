# simple program to detect mitosis in annotated images
# uses training set from training/ folder
# maps 2084 x 2084 image green array to 2084 x 2084 bitmap 
# representing nuclei with mitosis

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from urllib import urlretrieve
import cPickle as pickle
import os
import sys
import gzip

import numpy as np
import theano

import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum

from nolearn.lasagne import NeuralNet
from nolearn.lasagne import visualize

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from PIL import Image
import codecs

import csv
import xlrd

# useful functions
def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

# Constants
SIZE = 2084
PATCH_SIZE = 50
NUM = 25
VAL_NUM = 11
PATCH_NUM = 366
VAL_PATCH_NUM = 144
TRAIN = "train/A0";
TEST = "test/A0";
retrain = False

with open("convnetMitosisPatches.pkl", 'rb') as f:
    net1 = pickle.load(f);
    nn = net1

cnt = 0
cnt2 = 0
for n in range(1, 5):
	for m in range(0, 9):
		print cnt,
		if cnt >= NUM:
			break
		imageFile = TRAIN + `n` + "_0" + `m` + ".bmp"
		annotFile = TRAIN + `n` + "_0" + `m` + ".csv"
		try:
			img = Image.open(imageFile);
			img2 = Image.open(imageFile);
		except:
			continue

		img.show();
		pix = img.load();
		pix2 = img2.load();
		for i in range(0, (SIZE - PATCH_SIZE)/20):
			i= 20*i
			print i,
			arr = np.zeros((SIZE - PATCH_SIZE, 1, PATCH_SIZE, PATCH_SIZE))
			for j in range(0, SIZE - PATCH_SIZE):
				for di in range(0, PATCH_SIZE):
					for dj in range(0, PATCH_SIZE):
						arr[j, 0, di, dj] = float(pix[i + di, j + dj] [1])/255.0
			print "Preds", i
			pred = net1.predict(arr)
			print pred
			print "Sum", sum(pred)
			for j in range(11, SIZE - PATCH_SIZE - 11):
				if pred[j] == 1:
					for di in range(0, 40):
						for dj in range(-10, 10):
							try:
								pix2[i + di, j + dj] = (0, 0, 200);
							except:
								continue
			img2.show();
		break
	break
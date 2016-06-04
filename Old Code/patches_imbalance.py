
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
import glob
import sys
import gzip

import numpy as np
import theano

from keras.utils import np_utils

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from PIL import Image
import codecs

import csv
import xlrd




# Constants
SIZE = 2084
PATCH_SIZE = 50
NUM = 1
VAL_NUM = 11
PATCH_NUM = 2000*NUM
VAL_PATCH_NUM = 144
split = 0.7
TRAIN = "train/A0";
TEST = "test/A0";
retrain = True


# creating and storing training + testing data
# needs to be pickled
trainImg = np.zeros((PATCH_NUM, 3, PATCH_SIZE, PATCH_SIZE))
trainVal = np.zeros((PATCH_NUM))
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
			img.show();
			pix = img.load()
		except:
			continue
		#img.resize((SIZE, SIZE))


		#for i in range(0, SIZE):
		#	for j in range(0, SIZE):
		#		trainImg[cnt, 0, i, j] = float(pix[i, j][1])/255

		csvReader = csv.reader(codecs.open(annotFile, 'rU', 'utf-8'))
		tot = 0
		imgMask = np.zeros((SIZE, SIZE));
		for row in csvReader:
			minx, miny, maxx, maxy = (SIZE, SIZE, 0, 0)
			for i in range(0, len(row)/2):
				tot = tot + 1
				xv, yv = (int(row[2*i]), int(row[2*i+1]))
				imgMask[xv, yv] = 1
			print ("here");

			if minx + PATCH_SIZE >= 2084 or miny + PATCH_SIZE >= 2084:
				continue
			if maxx + PATCH_SIZE >= 2084 or maxy + PATCH_SIZE >= 2084:
				continue
			#if minx - PATCH_SIZE <= 0 or miny - PATCH_SIZE <= 0:
			#	continue
			#if maxx - PATCH_SIZE <= 0 or maxy - PATCH_SIZE <= 0:
			#	continue
			# shows image
			img2 = img.crop((minx, miny, minx + PATCH_SIZE, miny + PATCH_SIZE))
			print "Showing image"
			img2.show();
			

			#puts image into train data
		for i in range(0, SIZE/PATCH_SIZE - 1):
			i = i * PATCH_SIZE
			for j in range(0, SIZE/PATCH_SIZE - 1):
				j = j * PATCH_SIZE
				for di in range(0, PATCH_SIZE):
					for dj in range(0, PATCH_SIZE):
						trainImg[cnt2, 0, di, dj] = float(pix[i + di, j + dj][0])/255
						trainImg[cnt2, 1, di, dj] = float(pix[i + di, j + dj][1])/255
						trainImg[cnt2, 2, di, dj] = float(pix[i + di, j + dj][2])/255

						if imgMask[i + di, j + dj] == 1:
							trainVal[cnt2] = 1
				cnt2 = cnt2 + 1
				if cnt2 % 1000 == 0:
					print cnt2
		cnt = cnt + 1

print trainImg
print trainVal
print cnt2

trainImg = trainImg[0:cnt2]
trainVal = trainVal[0:cnt2]

trainVal = np_utils.to_categorical(trainVal, 2)
print "Train validation", trainVal
print sum(trainVal)

np.save("train_image_data.npy", trainImg)
np.save("train_val_data.npy", trainVal)


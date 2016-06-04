
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

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils

from PIL import Image
import codecs

import csv
import xlrd
import h5py


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
BATCH = 64
EPOCH = 35
split = 0.7
TRAIN = "train/A0";
TEST = "test/A0";
retrain = True


# Network Constants

nb_filters = 250
nb_conv = 6
nb_pool = 2

img_rows, img_cols = PATCH_SIZE, PATCH_SIZE



# Neural network CONV model: will be used later
# to fit data
modelCONV9 = Sequential()
modelCONV9.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                        border_mode='valid',
                        input_shape=(3, img_rows, img_cols)))
modelCONV9.add(Activation('relu'))
modelCONV9.add(Convolution2D(nb_filters/2, nb_conv - 1, nb_conv - 1))
modelCONV9.add(Activation('relu'))
modelCONV9.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
modelCONV9.add(Dropout(0.5))
modelCONV9.add(Flatten())

modelCONV9.add(Dense(128))
modelCONV9.add(Activation('relu'))
modelCONV9.add(Dropout(0.4))
modelCONV9.add(Dense(2))
modelCONV9.add(Activation('softmax'))

modelCONV9.compile(loss='categorical_crossentropy', optimizer='adadelta')

print "Convolution model: ", modelCONV9




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
			pix = img.load()
		except:
			continue
		#img.resize((SIZE, SIZE))
		2


		#for i in range(0, SIZE):
		#	for j in range(0, SIZE):
		#		trainImg[cnt, 0, i, j] = float(pix[i, j][1])/255

		csvReader = csv.reader(codecs.open(annotFile, 'rU', 'utf-8'))
		tot = 0
		
		for row in csvReader:
			minx, miny, maxx, maxy = (SIZE, SIZE, 0, 0)
			for i in range(0, len(row)/2):
				tot = tot + 1
				xv, yv = (int(row[2*i]), int(row[2*i+1]))
				minx = min(minx, xv)
				maxx = max(maxx, xv)
				miny = min(miny, yv)
				maxy = max(maxy, yv)

			if minx + PATCH_SIZE >= 2084 or miny + PATCH_SIZE >= 2084:
				continue
			if maxx + PATCH_SIZE >= 2084 or maxy + PATCH_SIZE >= 2084:
				continue
			#if minx - PATCH_SIZE <= 0 or miny - PATCH_SIZE <= 0:
			#	continue
			#if maxx - PATCH_SIZE <= 0 or maxy - PATCH_SIZE <= 0:
			#	continue
			# shows image
			#img2 = img.crop((minx, miny, minx + PATCH_SIZE, miny + PATCH_SIZE))
			#img2.show();
			

			#puts image into train data
			for i in range(0, PATCH_SIZE):
				for j in range(0, PATCH_SIZE):
					trainImg[cnt2, 0, i, j] = float(pix[minx + i, miny + j][0])/255
					trainImg[cnt2, 1, i, j] = float(pix[minx + i, miny + j][1])/255
					trainImg[cnt2, 2, i, j] = float(pix[minx + i, miny + j][2])/255
			#print trainImg[cnt2, 0]
			#plt.imshow(trainImg[cnt2, 0])
			trainVal[cnt2] = 1;
			cnt2 = cnt2 + 1

			#puts normal image into train data
			for i in range(0, PATCH_SIZE):
				for j in range(0, PATCH_SIZE):
					trainImg[cnt2, 0, i, j] = float(pix[maxx + i, maxy + j][0])/255
					trainImg[cnt2, 1, i, j] = float(pix[maxx + i, maxy + j][1])/255
					trainImg[cnt2, 2, i, j] = float(pix[maxx + i, maxy + j][2])/255
			trainVal[cnt2] = 0;
			cnt2 = cnt2 + 1

			#puts flipped image into train data
			#for i in range(0, PATCH_SIZE):
			#	for j in range(0, PATCH_SIZE):
			#		trainImg[cnt2, 0, i, j] = float(pix[maxx - i, maxy - j][1])/255
			#trainVal[cnt2] = 1;
			#cnt2 = cnt2 + 1

			#puts flipped normal into train data
			#for i in range(0, PATCH_SIZE):
			#	for j in range(0, PATCH_SIZE):
			#		trainImg[cnt2, 0, i, j] = float(pix[minx - i, miny - j][1])/255
			#trainVal[cnt2] = 0;
			#cnt2 = cnt2 + 1
		cnt = cnt + 1

if retrain:
	trainVal = np_utils.to_categorical(trainVal)
	modelCONV9.fit(trainImg, trainVal, batch_size=BATCH, nb_epoch=EPOCH,
          		show_accuracy=True, verbose=1)
	#trainImg = trainImg.astype(np.uint8)

	#plt.imshow(trainVal[0, 0], cmap=cm.binary)
	json_string = modelCONV7.to_json()
	open('modelCONV7_architecture.json', 'w').write(json_string)
	modelCONV7.save_weights('modelCONV7_skew_weights.h5')
	trainVal = trainVal.astype(np.uint8)

	print cnt2
	print trainImg
	print trainVal

if not retrain:
	with open("convnetMitosisPatches.pkl", 'rb') as f:
	    net1 = pickle.load(f);
	    nn = net1



#preds = net1.predict(trainImg[701:PATCH_NUM])
#print trainVal[701:PATCH_NUM]
#print preds


#cm = confusion_matrix(trainVal[701:PATCH_NUM], preds)
#plt.matshow(cm)
#plt.title('Confusion matrix')
#plt.colorbar()
#plt.ylabel('True label')
#plt.xlabel('Predicted label')
#plt.show()
























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

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator

from nolearn.lasagne import NeuralNet
from nolearn.lasagne import visualize

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle

from PIL import Image
import codecs

import csv
import xlrd

# Constants
SIZE = 2084
PATCH_SIZE = 50
NUM = 25
VAL_NUM = 11
PATCH_NUM = 2000*NUM
VAL_PATCH_NUM = 144
BATCH = 8
EPOCH = 20
split = 0.7
TRAIN = "train/A0";
TEST = "test/A0";
retrain = True




# Network Constants

nb_filters = 256
nb_conv = 4
nb_pool = 2

img_rows, img_cols = PATCH_SIZE, PATCH_SIZE



# Neural network CONV model: will be used later
# to fit data
modelCONV9 = Sequential()
modelCONV9.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                        border_mode='valid',
                        input_shape=(3, img_rows, img_cols)))
modelCONV9.add(Activation('relu'))
modelCONV9.add(Convolution2D(nb_filters/2, nb_conv + 1, nb_conv + 1))
modelCONV9.add(Activation('relu'))
modelCONV9.add(Flatten())
modelCONV9.add(Dropout(0.5))
modelCONV9.add(Dense(64))
modelCONV9.add(Activation('relu'))
modelCONV9.add(Dropout(0.5))
modelCONV9.add(Dense(2))
modelCONV9.add(Activation('softmax'))


# Data augmentation model

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    zca_whitening=True,
    rotation_range=45,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.1,
    horizontal_flip=True,
    vertical_flip=True)



modelCONV9.compile(loss='mean_squared_error', optimizer='adadelta')

print "Convolution model: ", modelCONV9


print "Loading numpy data:"

trainImg = np.load("train_image_data.npy");
trainVal = np.load("train_val_data.npy");

PATCH_NUM = len(trainVal)



subset = np.zeros(PATCH_NUM) 
subset = subset.astype(int)
slen = 0


for i in range(0, PATCH_NUM):
	if trainVal[i][0] == 0:
		subset[slen] = i
		slen = slen + 1
print subset[0:slen]

crit = [slen*2, slen*4, slen*8, slen*16, slen*30, slen*50]

for i in range(0, PATCH_NUM):
	if trainVal[i][0] == 1:
		subset[slen] = i
		slen = slen + 1
		if slen in crit:

			print subset[1:slen]
			print "Accuracy Benchmark:", 421.0/slen

			trainImg2 = trainImg[subset[0:slen], ]
			trainVal2 = trainVal[subset[0:slen]]

			trainImg2, trainVal2 = shuffle(trainImg2, trainVal2, random_state=5)

			TRAIN_NUM = int(slen*0.7)
			VAL_NUM = slen

			print "Training validation stats:"
			print len(trainVal2)
			print sum(trainVal2)

			print "Fitting augmentation model:"
			datagen.fit(trainImg2);
			print "Training with generator"
			modelCONV9.fit_generator(datagen.flow(trainImg2[0:TRAIN_NUM], trainVal2[0:TRAIN_NUM], batch_size=BATCH), 
				samples_per_epoch=3*len(trainImg2), nb_epoch = EPOCH, show_accuracy=True, verbose=1, validation_data=(trainImg2[TRAIN_NUM:VAL_NUM], trainVal2[TRAIN_NUM:VAL_NUM]))
			EPOCH = int(EPOCH * 0.6)

# Train the network
#modelCONV9.fit(trainImg[0:TRAIN_NUM], trainVal[0:TRAIN_NUM], batch_size=BATCH, nb_epoch=EPOCH,
          #show_accuracy=True, verbose=1, validation_data=(trainImg[TRAIN_NUM:VAL_NUM], trainVal[TRAIN_NUM:VAL_NUM]))


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



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

			print "Creating Lasagne net:"
			net1 = NeuralNet(
			    layers=[('input', layers.InputLayer),
			            ('conv2d1', layers.Conv2DLayer),
			            ('maxpool1', layers.MaxPool2DLayer),
			            ('conv2d2', layers.Conv2DLayer),
			            ('maxpool2', layers.MaxPool2DLayer),
			            ('dropout1', layers.DropoutLayer),
			            ('dense', layers.DenseLayer),
			            ('dropout2', layers.DropoutLayer),
			            ('output', layers.DenseLayer),
			            ],
			    # input layer
			    input_shape=(None, 3, PATCH_SIZE, PATCH_SIZE),
			    # layer conv2d1
			    conv2d1_num_filters=100,
			    conv2d1_filter_size=(9, 9),
			    conv2d1_nonlinearity=lasagne.nonlinearities.rectify,
			    conv2d1_W=lasagne.init.GlorotUniform(),  
			    # layer maxpool1
			    maxpool1_pool_size=(4, 4),    
			    # layer conv2d2
			    conv2d2_num_filters=50,
			    conv2d2_filter_size=(6, 6),
			    conv2d2_nonlinearity=lasagne.nonlinearities.rectify,
			    # layer maxpool2
			    maxpool2_pool_size=(3, 3),
			    # dropout1
			    dropout1_p=0.3,    
			    # dense
			    dense_num_units=300,
			    dense_nonlinearity=lasagne.nonlinearities.rectify,    
			    # dropout2
			    dropout2_p=0.3, 
			    # output
			    output_nonlinearity=lasagne.nonlinearities.softmax,
			    output_num_units=2,
			    # optimization method params
			    update=nesterov_momentum,
			    update_learning_rate=0.004,
			    max_epochs=50,
			    verbose=1,
			    )
			
			print "Fitting Lasagne net:", net1
			# Train the network
			nn = net1.fit(trainImg2, trainVal2)


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


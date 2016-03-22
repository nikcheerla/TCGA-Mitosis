
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
SIZE = 20
NUM = 25
TRAIN = "train/A0";

# creating and storing training + testing data
# needs to be pickled
trainImg = np.zeros((NUM, 1, SIZE, SIZE))
trainVal = np.zeros((NUM))
cnt = 0

for n in range(1, 5):
	for m in range(0, 9):
		print cnt,
		if cnt >= NUM:
			break
		imageFile = TRAIN + `n` + "_0" + `m` + ".jpg"
		annotFile = TRAIN + `n` + "_0" + `m` + ".csv"
		try:
			img = Image.open(imageFile);
		except:
			continue
		img.resize((SIZE, SIZE))
		pix = img.load()

		for i in range(0, SIZE):
			for j in range(0, SIZE):
				trainImg[cnt, 0, i, j] = float(pix[i, j][1])/255

		csvReader = csv.reader(codecs.open(annotFile, 'rU', 'utf-8'))
		tot = 0
		for row in csvReader:
			for i in range(0, len(row)/2):
				tot = tot + 1
				#xv, yv = (int(row[2*i]), int(row[2*i+1]))
				#trainVal[cnt, 0, xv*SIZE + yv] = 1

		trainVal[cnt] = float(tot)/(2084*2084)
		trainVal[cnt] = trainImg[cnt, 0, 0, 0]
		cnt = cnt + 1

#trainImg = trainImg.astype(np.uint8)

#plt.imshow(trainVal[0, 0], cmap=cm.binary)


print trainImg
print trainVal

save_object(trainImg, "trainImg.pkl");
save_object(trainVal, "trainVal.pkl");
sys.exit();

# neural (later convolutional) network 
print("making net")

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
    input_shape=(None, 1, SIZE, SIZE),
    # layer conv2d1
    conv2d1_num_filters=32,
    conv2d1_filter_size=(5, 5),
    conv2d1_nonlinearity=lasagne.nonlinearities.rectify,
    conv2d1_W=lasagne.init.GlorotUniform(),  
    # layer maxpool1
    maxpool1_pool_size=(2, 2),    
    # layer conv2d2
    conv2d2_num_filters=32,
    conv2d2_filter_size=(5, 5),
    conv2d2_nonlinearity=lasagne.nonlinearities.rectify,
    # layer maxpool2
    maxpool2_pool_size=(2, 2),
    # dropout1
    dropout1_p=0.5,    
    # dense
    dense_num_units=256,
    dense_nonlinearity=lasagne.nonlinearities.rectify,    
    # dropout2
    dropout2_p=0.5,   
    # output
    output_nonlinearity=lasagne.nonlinearities.softmax,
    output_num_units=1,
    # optimization method params
    update_learning_rate=0.9,
    regression = True,
    max_epochs=20,
    verbose=2,
    )

print ("fitting net")
#net1.fit(trainImg, trainVal)

#preds = net1.predict(trainImg);
#print preds
print trainVal


net1 = NeuralNet(
    layers=[('input', layers.InputLayer),
    		('dense', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
    # input layer
    input_shape=(None, 1, 5),
    # dense
    dense_num_units=128,
    dense_nonlinearity=lasagne.nonlinearities.rectify,
    # output
    output_nonlinearity=lasagne.nonlinearities.softmax,
    output_num_units=1,
    # optimization method params
    update_learning_rate=0.9,
    regression = True,
    max_epochs=10,
    verbose=1,
    )

trainImg2 = np.zeros((NUM, 1, 5))
for i in range(0, NUM):
	trainImg2[i, 0, 0] = trainVal[i]
	trainImg2[i, 0, 1] = trainVal[i]*0.5
	trainImg2[i, 0, 2] = trainVal[i]*0.8
	trainImg2[i, 0, 3] = trainVal[i]*0.7
	trainImg2[i, 0, 4] = trainVal[i]*0.4

net1.fit(trainImg2, trainVal)

preds = net1.predict(trainImg2);
print preds

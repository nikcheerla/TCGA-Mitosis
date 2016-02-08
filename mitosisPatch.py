
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
PATCH_NUM = 366
TRAIN = "train/A0";

# creating and storing training + testing data
# needs to be pickled
trainImg = np.zeros((PATCH_NUM, 1, PATCH_SIZE, PATCH_SIZE))
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
		except:
			continue
		#img.resize((SIZE, SIZE))
		pix = img.load()


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
			# shows image
			#img2 = img.crop((minx, miny, minx + PATCH_SIZE, miny + PATCH_SIZE))
			#img2.show();
			

			#puts image into train data
			for i in range(0, PATCH_SIZE):
				for j in range(0, PATCH_SIZE):
					trainImg[cnt2, 0, i, j] = float(pix[minx + i, miny + j][1])/255
			#print trainImg[cnt2, 0]
			#plt.imshow(trainImg[cnt2, 0])
			trainVal[cnt2] = 1;
			cnt2 = cnt2 + 1

			

			#puts normal image into train data
			for i in range(0, PATCH_SIZE):
				for j in range(0, PATCH_SIZE):
					trainImg[cnt2, 0, i, j] = float(pix[maxx + i, maxy + j][1])/255
			trainVal[cnt2] = 0;
			cnt2 = cnt2 + 1
		cnt = cnt + 1

#trainImg = trainImg.astype(np.uint8)

#plt.imshow(trainVal[0, 0], cmap=cm.binary)
trainVal = trainVal.astype(np.uint8)

print cnt2
print trainImg
print trainVal


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
    input_shape=(None, 1, PATCH_SIZE, PATCH_SIZE),
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

# Train the network
nn = net1.fit(trainImg, trainVal)

save_object(nn, 'nnetMNIST.pkl')


preds = net1.predict(trainImg)
print trainVal
print preds

#cm = confusion_matrix(trainVal, preds)
#plt.matshow(cm)
#plt.title('Confusion matrix')
#plt.colorbar()
#plt.ylabel('True label')
#plt.xlabel('Predicted label')
#plt.show()


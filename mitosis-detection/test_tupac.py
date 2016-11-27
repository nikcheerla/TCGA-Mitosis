import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import cPickle as pickle
import os
import glob
import sys
import time



import numpy as np
import random
from heapq import heappush, heappop, heappushpop, nlargest, heapify

import csv
import subprocess

from scipy.signal import convolve
from scipy.spatial import distance
import bisect

import spams
from staining.stainingController import Controller
from staining.method.macenko import macenko



# Constants
SIZE = 2000
PATCH_SIZE = 101
PATCH_GAP = 50
RADIUS = 10

image_dir = sys.argv[1] + "*.jpg"
if len(sys.argv) == 5 and sys.argv[4] == 'icpr':
    image_dir = sys.argv[1] + "*.bmp"
print (image_dir)

images = sorted(glob.glob(image_dir))
print (images)

for image in images:
	print "Image: " + image
	if len(sys.argv) == 5 and sys.argv[4] == 'icpr':
		print ("ICPR")
		print subprocess.check_output(["bsub", "-n", "3", "-q", "mcore", "-W", "120:00", "-R", "rusage[mem=10000]", "-o", image[:-4] + "_prgmdata.out",
			"-e", image[:-4] + "_prgmdata.err", "THEANO_FLAGS=gcc.cxxflags='-march=corei7'", "python", "-u", "test_image_icpr.py", image, sys.argv[2], sys.argv[3]])
	else:
		print subprocess.check_output(["bsub", "-n", "3", "-q", "mcore", "-W", "120:00", "-R", "rusage[mem=10000]", "-o", image[:-4] + "_prgmdata.out",
			"-e", image[:-4] + "_prgmdata.err", "THEANO_FLAGS=gcc.cxxflags='-march=corei7'", "python", "-u", "test_image.py", image, sys.argv[2]])
	time.sleep(20)

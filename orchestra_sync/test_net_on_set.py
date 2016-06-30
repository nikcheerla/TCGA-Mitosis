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




# Constants
SIZE = 2084
PATCH_SIZE = 101
PATCH_GAP = 50
RADIUS = 10

images = glob.glob("test/*.bmp")

for image in images:
    print "Image: " + image
    print subprocess.check_output(["bsub", "-n", "3", "-q", "mcore", "-W", "120:00", "-R", "rusage[mem=10000]", "-o", image[:-4] + "_prgmdata.out",
        "-e", image[:-4] + "_prgmdata.err", "THEANO_FLAGS=gcc.cxxflags='-march=corei7'", "python", "-u", "test_net_image.py", image])
    time.sleep(40)

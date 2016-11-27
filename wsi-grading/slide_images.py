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

image_dir = "../../../groups/becklab/EC2/impath_datastore/tcga850/svs/*.svs"

filenames = ["TCGA-A1-A0SK-01Z-00-DX1.A44D70FA-4D96-43F4-9DD7-A61535786297.svs",
    "TCGA-A1-A0SP-01Z-00-DX1.20D689C6-EFA5-4694-BE76-24475A89ACC0.svs",
    "TCGA-A2-A04T-01Z-00-DX1.71444266-BD56-4183-9603-C7AC20C9DA1E.svs",
    "TCGA-A1-A0SH-01Z-00-DX1.90E71B08-E1D9-4FC2-85AC-062E56DDF17C.svs",
    "TCGA-A2-A0CP-01Z-00-DX1.ECFD263C-BB17-4ADA-8F2C-654C2AA4C45F.svs",
    "TCGA-A2-A0CS-01Z-00-DX1.3986B545-63E8-4727-BCC1-701DE947D1FB.svs"]

print (len(image_dir))

images = sorted(glob.glob(image_dir))
print (len(image_dir))



csv_reader = csv.reader(open("data/TUPAC_conversion.csv", 'rU'))
conversion = {}
for row in csv_reader:
    conversion[row[0].strip()] = row[1].strip() 

for image_file in images:
    image = image_file
    csv_file = None
    print ("Image: " + image)
    patient_name = image[-64:][:-4]
    print ("Debug loc: ", "debug/img_debugs/" + patient_name + "_prgmdata.err")

    if patient_name in conversion:
        print ("Patient " + patient_name + " in TUPAC database!")
        patient_id = conversion[patient_name]
        csv_potential_file = "data/ROIS/" + patient_id + "-ROI.csv"
        if os.path.exists(csv_potential_file):
            csv_file = csv_potential_file
            print ("Patient " + patient_name + " ANNOTATED with ROIS!")
            print (csv_file)
        else:
            print ("Patient " + patient_name + " not annotated with ROIS!")
    else:
        print ("Patient " + patient_name + " NOT IN TUPAC!")

    if len(sys.argv) == 3 and sys.argv[2] == 'diff':
        score_file = "data/scores/patient" + patient_name + ".score.npy"
        print (score_file)
        if os.path.exists(score_file):
            print ('Score already found!')
            continue
    if csv_file is not None:
        print subprocess.check_output(["bsub", "-n", "3", "-q", "mcore", "-W", "120:00", "-R", "rusage[mem=10000]", "-o", "debug/img_debugs/" + image[-64:][:-4] + "_prgmdata.out",
        "-e", "debug/img_debugs/" + image[-64:][:-4] + "_prgmdata.err", "THEANO_FLAGS=gcc.cxxflags='-march=corei7'", "python", "-u", "score_full2.py", image, sys.argv[1], csv_file])
    else:
        print subprocess.check_output(["bsub", "-n", "3", "-q", "mcore", "-W", "120:00", "-R", "rusage[mem=10000]", "-o", "debug/img_debugs/" + image[-64:][:-4] + "_prgmdata.out",
            "-e", "debug/img_debugs/" + image[-64:][:-4] + "_prgmdata.err", "THEANO_FLAGS=gcc.cxxflags='-march=corei7'", "python", "-u", "score_full2.py", image, sys.argv[1]])
    time.sleep(5)

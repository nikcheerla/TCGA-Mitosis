from __future__ import print_function

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import cPickle as pickle
import os, glob, sys, time, shutil

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

img = []
img_cur = []
img_aux = []
coords = []

total_coords = []

num_images = 0
img_cnt = 0
img_names = []
csv_file_loc = "../mitoses_aux_ground/"
img_file_loc = "../mitoses_image_data/"

count = 0
for patient_num in range(1, 73):
    if patient_num % 30 == 0:
        continue
    patient_dir_csv = csv_file_loc + "{0:0=2d}".format(patient_num) + "/"
    patient_dir_img = img_file_loc + "{0:0=2d}".format(patient_num) + "/"
    print (patient_dir_csv, patient_dir_img)

    num_image_files = len(sorted(glob.glob(patient_dir_img + "*.tif")))
    num_csv_files = len(sorted(glob.glob(patient_dir_csv + "*.csv")))

    print (num_image_files, num_csv_files)

    for image_file_num in range(1, num_image_files + 1):

        full_path_csv = patient_dir_csv + "{0:0=2d}".format(image_file_num) + ".csv"
        full_path_img = patient_dir_img + "{0:0=2d}".format(image_file_num) + ".tif"

        new_file_csv = "data/train/patient" + "{0:0=2d}".format(patient_num) + "case" + full_path_img[-6:-4] + ".csv"
        new_file_img = "data/train/patient" + "{0:0=2d}".format(patient_num) + "case" + full_path_img[-6:-4] + ".jpg"

        print (full_path_img, full_path_csv)

        shutil.copyfile(full_path_img, new_file_img)
        if os.path.exists(full_path_csv):
            shutil.copyfile(full_path_csv, new_file_csv)

        img_cnt += 1


print ("\nTotal image count: ", img_cnt)
print ("\n")
        


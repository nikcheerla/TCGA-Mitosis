import openslide
print ("oppa")

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from PIL import Image
import cPickle as pickle
import os, glob, sys, gzip, random

import numpy as np
import csv
import theano

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from scipy.signal import convolve

from nuclei_detect import nuclei_detect_pipeline

sys.setrecursionlimit(10000)


#Constants

THRESHOLD = 25 #best threshold for mitoses



#RGBA --> RGB conversion
def pure_pil_alpha_to_color(image, color=(255, 255, 255)):
    """Alpha composite an RGBA Image with a specified color.

    Simpler, faster version than the solutions above.

    Source: http://stackoverflow.com/a/9459208/284318

    Keyword Arguments:
    image -- PIL RGBA Image object
    color -- Tuple r, g, b (default 255, 255, 255)

    """
    image.load()  # needed for split()
    background = Image.new('RGB', image.size, color)
    background.paste(image, mask=image.split()[3])  # 3 is the alpha channel
    return background




#gets ROIs from WSI image fils
def get_rois(image_file, size=1000, num=10, rois_file=None):
    print image_file

    img = openslide.open_slide(image_file)

    XMAX = img.dimensions[0]
    YMAX = img.dimensions[1]

    images = []
    mean_pixels = []
    IMAGE_SIZE = size

    if rois_file is not None and os.path.exists(rois_file):
        roi_regions = np.loadtxt(open(rois_file, "rU"),delimiter=",").astype(int)
        print ("Detected ROI regions: ", roi_regions)
        images = []
        for xv, yv, width, height in roi_regions:
            xc = xv + int(width/2) - int(IMAGE_SIZE/2)
            yc = yv + int(height/2) - int(IMAGE_SIZE/2)

            closeup = img.read_region((xc, yc), 0, (IMAGE_SIZE, IMAGE_SIZE))
            closeup = np.array(pure_pil_alpha_to_color(closeup))

            images.append(closeup)
        return np.array(images)

    for i in range(0, num*10):
        xv = random.randint(0, XMAX - IMAGE_SIZE)
        yv = random.randint(0, YMAX - IMAGE_SIZE)

        closeup = img.read_region((xv, yv), 0, (IMAGE_SIZE, IMAGE_SIZE))
        closeup = np.array(pure_pil_alpha_to_color(closeup))

        images.append(closeup)
        nimg, nuclei_bmp = nuclei_detect_pipeline(closeup)
        mean_pixels.append(nuclei_stats(nuclei_bmp)[0])

    sort_idx = np.argsort(mean_pixels)[::-1]

    #histogram distribution, mosts compressed (i.e peak) window of length corresponding to desired ROIS

    best_diff = 100000000.0
    best_window = 0
    for i in range(len(sort_idx//2, len(sort_idx) - num):
        window_start = i
        window_end = i + num
        diff = mean_pixels[sort_idx[window_end]] - mean_pixels[sort_idx[window_start]]
        if diff < best_diff:
            best_diff = diff
            best_window = window_start

    images_filtered = np.array(images)[sort_idx][best_window:best_window + num]

    return images_filtered



#gets mitosis and nuclei features
def extract_features(nuclei_map, patch_probs):

    radius = 10
    kernel = np.zeros((2*radius+1, 2*radius+1))
    y,x = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = x**2 + y**2 <= radius**2
    kernel[mask] = 1

    patch_probs = convolve(patch_probs, kernel, mode='same')

    mitosis_sum = np.sum(patch_probs)
    nuclei_sum = np.sum(nuclei_map)

    mitosis_map = patch_probs > THRESHOLD

    grid_markoff = np.zeros(shape=mitosis_map.shape)

    num_mitosis = 0
    avg_mitosis = 0
    for x in range(0, mitosis_map.shape[0]):
        for y in range(0, mitosis_map.shape[1]):
            visited = []
            floodfill(mitosis_map, x, y, visited, grid_markoff)
            if len(visited) > 0:
                print ("Found mitosis!", x, y)
                num_mitosis += 1
                avg_mitosis += len(visited)

    if num_mitosis == 0:
        avg_mitosis = 0
    else:
        avg_mitosis /= num_mitosis

    grid_markoff = np.zeros(shape=nuclei_map.shape)

    num_nuclei = 0
    avg_nuclei = 0
    for x in range(0, nuclei_map.shape[0]):
        for y in range(0, nuclei_map.shape[1]):
            visited = []
            floodfill(nuclei_map, x, y, visited, grid_markoff)
            if len(visited) > 0:
                print ("Found nuclei!", x, y)
                num_nuclei += 1
                avg_nuclei += len(visited)

    if num_nuclei == 0:
        avg_nuclei = 0
    else:
        avg_nuclei /= num_nuclei

    return [mitosis_sum, nuclei_sum, num_mitosis, avg_mitosis, num_nuclei, avg_nuclei]



#floodfill on image
def floodfill(bin_map, x, y, visited, grid_markoff):
    if x >= grid_markoff.shape[0] or y >= grid_markoff.shape[1] or x < 0 or y < 0:
        return
    if grid_markoff[x, y]:
        return
    if not bin_map[x, y]:
        return

    grid_markoff[x, y] = True
    visited.append((x, y))

    floodfill(bin_map, x+1, y, visited, grid_markoff)
    floodfill(bin_map, x, y+1, visited, grid_markoff)
    floodfill(bin_map, x-1, y, visited, grid_markoff)
    floodfill(bin_map, x, y-1, visited, grid_markoff)


#gets nuclei number, avg nuclei statistics
def nuclei_stats(nuclei_map):
    grid_markoff = np.zeros(shape=nuclei_map.shape)
    
    num_nuclei = 0
    avg_nuclei = 0
    for x in range(0, nuclei_map.shape[0]):
        for y in range(0, nuclei_map.shape[1]):
            visited = []
            floodfill(nuclei_map, x, y, visited, grid_markoff)
            if len(visited) > 0:
                #print ("Found nuclei!", x, y)
                num_nuclei += 1
                avg_nuclei += len(visited)
    if num_nuclei > 0:
        avg_nuclei /= num_nuclei
    return (num_nuclei, avg_nuclei)




#test statistics on example image file
if __name__ == "__main__":
    patch_probs = np.load("experiment/img13.out.npy")
    img = plt.imread("experiment/img13.jpg")

    nimg, nuclei_map = nuclei_detect_pipeline(img)

    features = extract_features(nuclei_map, patch_probs)
    print ("Features: ", features)

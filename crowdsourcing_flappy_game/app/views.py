from flask import render_template, url_for, send_file
from app import app
from functools import wraps, update_wrapper
from datetime import datetime

import cPickle as pickle
import os
import glob
import sys

import matplotlib
if 'DYNO' not in os.environ:
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm




import numpy as np
import random
from heapq import heappush, heappop, heappushpop, nlargest, heapify

from numpy.linalg import norm

import csv

from flask_table import Table, Col



### Important Methods ###

from datetime import timedelta
from flask import make_response, request, current_app
from functools import update_wrapper

def nocache(view):
    @wraps(view)
    def no_cache(*args, **kwargs):
        response = make_response(view(*args, **kwargs))
        response.headers['Last-Modified'] = datetime.now()
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '-1'
        return response
        
    return update_wrapper(no_cache, view)


def crossdomain(origin=None, methods=None, headers=None,
                max_age=21600, attach_to_all=True,
                automatic_options=True):
    if methods is not None:
        methods = ', '.join(sorted(x.upper() for x in methods))
    if headers is not None and not isinstance(headers, basestring):
        headers = ', '.join(x.upper() for x in headers)
    if not isinstance(origin, basestring):
        origin = ', '.join(origin)
    if isinstance(max_age, timedelta):
        max_age = max_age.total_seconds()

    def get_methods():
        if methods is not None:
            return methods

        options_resp = current_app.make_default_options_response()
        return options_resp.headers['allow']

    def decorator(f):
        def wrapped_function(*args, **kwargs):
            if automatic_options and request.method == 'OPTIONS':
                resp = current_app.make_default_options_response()
            else:
                resp = make_response(f(*args, **kwargs))
            if not attach_to_all and request.method != 'OPTIONS':
                return resp

            h = resp.headers

            h['Access-Control-Allow-Origin'] = origin
            h['Access-Control-Allow-Methods'] = get_methods()
            h['Access-Control-Max-Age'] = str(max_age)
            if headers is not None:
                h['Access-Control-Allow-Headers'] = headers
            return resp

        f.provide_automatic_options = False
        return update_wrapper(wrapped_function, f)
    return decorator
















SIZE = 2084
PATCH_SIZE = 101
PATCH_GAP = 50
RADIUS = 10

coords = []
filenames = glob.glob("app/static/train/*.jpg")

def get_patches(coords, patchsize=PATCH_SIZE):
    patches = np.zeros((len(coords), patchsize, patchsize, 3))
    i = 0
    heap = []
    for i in range(0, len(coords)):
        dpair = coords[i]
        print dpair
        x = dpair[0]
        y = dpair[1]
        img_num = dpair[2]

        image_file = filenames[img_num]
        print("Loading Image")
        img = plt.imread(image_file)
        # print x, y
        #print (x - patchsize/2), (x + patchsize/2 + 1), (y - patchsize/2), (y + patchsize/2 + 1)
        patches[i] = img[(x - patchsize / 2):(x + patchsize / 2 + 1),
                         (y - patchsize / 2):(y + patchsize / 2 + 1), :]
        patches[i] = np.divide(patches[i], 255.0)
    return patches

# Constants

def get_images():
    global img, coords
    coords = []
    cnt = 0
    for imgfile in glob.iglob("app/static/train/*.jpg"):
        print "\n" + imgfile,
        annotfile = imgfile[:-3] + "csv"
        csvReader = csv.reader(open(annotfile, 'rb'))
        tot = 0
        imgMask = np.zeros((SIZE, SIZE))
        for row in csvReader:
            minx, miny, maxx, maxy = (SIZE, SIZE, 0, 0)
            random_coords = []
            for i in range(0, len(row) / 2):
                xv, yv = (int(row[2 * i]), int(row[2 * i + 1]))
                if xv > PATCH_SIZE / 2 + 1 and yv > PATCH_SIZE / 2 + 1 and xv < SIZE - PATCH_SIZE / 2 - 1 and yv < SIZE - PATCH_SIZE / 2 - 1:
                    random_coords.append([yv, xv, cnt])

            centroid = np.array(random_coords).mean(axis=0).astype(int)
            print centroid,
            for i in range(0, len(row) / 2):
                xv, yv = (int(row[2 * i]), int(row[2 * i + 1]))
                if norm(np.array([yv, xv, cnt]) - centroid) <= RADIUS:
                    if xv > PATCH_SIZE / 2 + 1 and yv > PATCH_SIZE / 2 + 1 and xv < SIZE - PATCH_SIZE / 2 - 1 and yv < SIZE - PATCH_SIZE / 2 - 1:
                        coords.append((yv, xv, cnt))
                        tot = tot + 1
        cnt += 1

    print len(coords)


get_images()
try:
    results = pickle.load(open( "results.pkl", "rb" ))
except:
    results = []
print "Results = " + str(results)

@app.route('/')
@app.route('/index')
@nocache
@crossdomain(origin='*')
def index():
    return render_template('index.html', title='Home')



@app.route('/mitosis')
@crossdomain(origin='*')
@nocache
def random_mitosis():
    x, y, img_num = random.choice(coords)
    x += random.randint(-10, 10)
    y += random.randint(-10, 10)
    patch = get_patches([(x , y, img_num)])[0]
    plt.imsave("app/static/patch.png", patch)
    print "Saved Patch (Hopefully)"
    return send_file("static/patch.png")

@app.route('/normal')
@crossdomain(origin='*')
@nocache
def random_normal():
    normal_coords = np.load("app/static/intermediate_data.npy")
    x, y, img_num = random.choice(normal_coords)
    x += random.randint(-10, 10)
    y += random.randint(-10, 10)
    patch = get_patches([(x , y, img_num)])[0]
    plt.imsave("app/static/patch.png", patch)
    print "Saved Normal Patch (Hopefully)"
    return send_file("static/patch.png")

idq = 1
@app.route('/uncategorized')
@crossdomain(origin='*')
@nocache
def random_uncat():
    global idq
    try:
        idq = pickle.load(open( "idq.pkl", "rb" ))
        print "Loaded id " + str(idq)
    except:
        idq = 1
    num = random.randint(1, 6)
    idq = num
    pickle.dump( idq, open( "idq.pkl", "wb" ) )
    return send_file("static/uncat/uncat" + str(num) + ".png")


@app.route('/crossdomain.xml')
@crossdomain(origin='*')
@nocache
def crossdmn():
    return send_file("../crossdomain.xml")


@app.route('/categorize', methods = ['POST'])
def categorize():
    global results
    try:
        results = pickle.load(open( "results.pkl", "rb" ))
        print "Loaded results " + str(results)
    except:
        results = []
    accuracy = int(request.form['accuracy'])
    label = int(request.form['label'])
    category = str(request.form['category'])
    results.append((accuracy, label, category))
    pickle.dump( results, open( "results.pkl", "wb" ) )
    print "Results = " + str(results)
    return "Good"

@app.route('/idquery')
@crossdomain(origin='*')
@nocache
def idquery():
    global idq
    try:
        idq = pickle.load(open( "idq.pkl", "rb" ))
        print "Loaded id " + str(idq)
    except:
        idq = 1
    return str(idq)





def gen_tables():
    global results
    try:
        results = pickle.load(open( "results.pkl", "rb" ))
        print "Loaded results " + str(results)
    except:
        results = []
    tables = {}
    for num in range(1, 7):
        items = [];

        # Declare your table
        class ItemTable(Table):
            player_num = Col('Player ID')
            num_correct = Col('Number Correct')
            classification = Col('Classification')
        # load items 
        fileUrl = "uncat/uncat" + str(num) + ".png"

        numTot = 0
        numMit = 0
        correct = 0
        for i in range(0, len(results)):
            print (i, results[i])
            if results[i][1] == num:
                items.append(dict(player_num = i, num_correct=results[i][0], classification = results[i][2]))
                numTot += 1
                correct += results[i][0]
                if results[i][2] == 'mitosis':
                    numMit += 1
        if numTot > 0:
            items.append(dict(player_num = "Average", num_correct= "{0:.2f}".format(correct/float(numTot)), classification = "{0:.1f}".format(numMit*100.0/numTot) + "% Mitosis"))
        
        # Populate the table
        table = ItemTable(items)
        tables[fileUrl] = table
    print tables
    return tables

@app.route('/data')
@crossdomain(origin='*')
@nocache
def render_data():
    tables = gen_tables()
    return render_template('data.html', tables=tables)

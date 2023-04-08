import os
import flask
import numpy as np
import argparse
import json
import csv

from flask import Flask
from flask_cors import CORS
import math

# create Flask app
app = Flask(__name__)
CORS(app)

# --- these will be populated in the main --- #

# list of attribute names of size m
attribute_names=None

# a 2D numpy array containing binary attributes - it is of size n x m, for n paintings and m attributes
painting_attributes=None

# a list of epsiode names of size n
episode_names=None

# a list of painting image URLs of size n
painting_image_urls=None

'''
This will return an array of strings containing the episode names -> these should be displayed upon hovering over circles.
'''
@app.route('/get_episode_names/', methods=['GET'])
def get_episode_names():
    return flask.jsonify(episode_names)
#

'''
This will return an array of URLs containing the paths of images for the paintings
'''
@app.route('/get_painting_urls', methods=['GET'])
def get_painting_urls():
    return flask.jsonify(painting_image_urls)
#

'''
TODO: implement PCA, this should return data in the same format as you saw in the first part of the assignment:
    * the 2D projection
    * x loadings, consisting of pairs of attribute name and value
    * y loadings, consisting of pairs of attribute name and value
'''
@app.route('/initial_pca', methods=['GET'])
def initial_pca():
    pass
#

'''
TODO: implement ccPCA here. This should return data in _the same format_ as initial_pca above.
It will take in a list of data items, corresponding to the set of items selected in the visualization. This can be acquired from `flask.request.json`. This should be a list of data item indices - the **target set**.
The alpha value, from the paper, should be set to 1.1 to start, though you are free to adjust this parameter.
'''
@app.route('/ccpca', methods=['GET','POST'])
def ccpca():
    pass
#

'''
TODO: run kmeans on painting_attributes, returning data in the same format as in the first part of the assignment. Namely, an array of objects containing the following properties:
    * label - the cluster label
    * id: the data item's id, simply its index
    * attribute: the attribute name
    * value: the binary attribute's value
'''
@app.route('/kmeans', methods=['GET'])
def kmeans():
    pass
#

if __name__=='__main__':
    painting_image_urls = json.load(open('painting_image_urls.json','r'))
    attribute_names = json.load(open('attribute_names.json','r'))
    episode_names = json.load(open('episode_names.json','r'))
    painting_attributes = np.load('painting_attributes.npy')

    app.run()
#

import os
import flask
import numpy as np
import time
from PIL import Image

from tqdm import tqdm

import torch

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from scipy.sparse import csgraph
from sklearn.neighbors import radius_neighbors_graph, kneighbors_graph

from collections import Counter

from flask import Flask, request, redirect, url_for
from flask_cors import CORS
from pathlib import Path
import math
import joblib

from matplotlib import pyplot as plt

from scipy import ndimage
from umap import UMAP

from file_functions import verifyDir, get_current_path

absolute_current_path = get_current_path()

# create Flask app
verifyDir(absolute_current_path+'/static')
app = Flask(__name__, static_folder=absolute_current_path+'/static')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
CORS(app)

# load all of the data generated from preprocessing
data_dict = joblib.load(f"{absolute_current_path}/static/data.joblib")

#similarity matrix
iou_23 = data_dict["iou_23"]
iou_34 = data_dict["iou_34"]
print(iou_34.shape)

S_2 = data_dict["S_2"]
S_3 = data_dict["S_3"]
S_4 = data_dict["S_4"]
    #print("S_2", S_2.shape, S_2)
    #print("S_3", S_3.shape, S_3)
    #print("S_4", S_4.shape, S_4)

#act2_thres = data_dict["act2_thres"]
#act3_thres = data_dict["act3_thres"]
#act3_up_thres = data_dict["act3_up_thres"]
#act4_thres = data_dict["act4_thres"]

# number of clusters - feel free to adjust
n_clusters = 9

pool_size = (2, 2)

# these variables will contain the clustering of channels for the different layers
a2_clustering,a3_clustering,a4_clustering = None, None, None
S_2_cluster, S_3_cluster, S_4_cluster = None, None, None

'''
Do not cache images on browser, see: https://stackoverflow.com/questions/34066804/disabling-caching-in-flask
'''
@app.after_request
def add_header(r):
    r.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    r.headers['Pragma'] = 'no-cache'
    r.headers['Expires'] = '0'
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r
#

'''
Implement spectral clustering, given an affinity matrix. You are required to implement this using standard matrix computation libraries, e.g. numpy, for computing a spectral embedding.
You may use k-means once you've obtained the spectral embedding.

NOTE: the affinity matrix should _not_ be symmetric! Nevertheless, eigenvectors will be real, up to numerical precision - so you should cast to real numbers (e.g. np.real).
'''
# Compute the Laplacian matrix
def compute_laplacian(affinity_matrix):
    degree_matrix = np.diag(np.sum(affinity_matrix, axis=1))
    laplacian = degree_matrix - affinity_matrix
    return laplacian

def spectral_clustering(affinity_mat, n_clusters=n_clusters):

    #A = radius_neighbors_graph(affinity_mat,0.4,mode='distance', metric='minkowski', p=2, metric_params=None, include_self=False)
    #A = kneighbors_graph(X_mn, 2, mode='connectivity', metric='minkowski', p=2, metric_params=None, include_self=False)
    #A = A.toarray()
    laplacian = csgraph.laplacian(affinity_mat, normed=False)
    #laplacian = compute_laplacian(affinity_mat)

    eigval, eigvec = np.linalg.eig(laplacian)
    eigvec_real = np.real(eigvec)
    sorted_indices = np.argsort(eigval)
    sorted_eigenvectors = eigvec_real[:, sorted_indices]
    selected_eigenvectors = sorted_eigenvectors[:, :n_clusters]

    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    clusters_labels = kmeans.fit_predict(selected_eigenvectors)

    return clusters_labels

'''
Cluster the channels within each layer.
This should take, as arguments, the two similarity matrices derived from the IoU scores.
Specifically, the first argument is the similarity matrix between channels at layer 2 and channels at layer 3.
The second argument is the similarity matrix between channels at layer 3 and channels at layer 4.

A generalization of spectral biclustering should be performed. More details given in the assignment notebook.
'''

def multiway_spectral_clustering(iou_23, iou_34, n_clusters=n_clusters):
    scaler = StandardScaler()
    S_2_standard = scaler.fit_transform(S_2)
    
    scaler = StandardScaler()
    S_3_standard = scaler.fit_transform(S_3)

    scaler = StandardScaler()
    S_4_standard = scaler.fit_transform(S_4)

    #print("S_2_standard", S_2_standard.shape, S_2_standard)
    #print("S_3_standard", S_3_standard.shape, S_3_standard)
    #print("S_4_standard", S_4_standard.shape, S_4_standard)

    S_2_cluster = spectral_clustering(S_2_standard)
    S_2_sklearn = SpectralClustering(n_clusters=n_clusters).fit(S_2_standard).labels_
    S_3_cluster = spectral_clustering(S_3_standard)
    S_3_sklearn = SpectralClustering(n_clusters=n_clusters).fit(S_2_standard).labels_
    S_4_cluster = spectral_clustering(S_4_standard)
    S_4_sklearn = SpectralClustering(n_clusters=n_clusters).fit(S_2_standard).labels_

    print("normal", S_2_cluster, "sklearn", S_2_sklearn)
    print("normal", S_3_cluster, "sklearn", S_3_sklearn)
    print("normal", S_4_cluster, "sklearn", S_4_sklearn)

    return S_2_cluster, S_3_cluster, S_4_cluster

'''
TODO
Given a link selected from the visualization, namely the layer and clusters at the individual layers, this route should compute the mean correlation from all channels in the source layer and all channels in the target layer, for each sample.
'''
def channel_correlation(layer1_activations, layer2_activations):
    # Reshape activations to (n, 512, 8*8)
    layer1_reshaped = np.reshape(layer1_activations, (layer1_activations.shape[0], 512, -1))
    layer2_reshaped = np.reshape(layer2_activations, (layer2_activations.shape[0], 512, -1))
    
    # Compute channel-wise mean
    layer1_mean = np.mean(layer1_reshaped, axis=2, keepdims=True)
    layer2_mean = np.mean(layer2_reshaped, axis=2, keepdims=True)
    
    # Compute channel-wise standard deviation
    layer1_std = np.std(layer1_reshaped, axis=2, keepdims=True)
    layer2_std = np.std(layer2_reshaped, axis=2, keepdims=True)
    
    # Compute correlation between channels
    correlation = np.mean(((layer1_reshaped - layer1_mean) / layer1_std) * ((layer2_reshaped - layer2_mean) / layer2_std),
                          axis=2)
    
    return correlation

@app.route('/link_score', methods=['GET','POST'])
def link_score():

    if request.method == 'POST':
        la_1 =  request.get_json()["layer_activations_1"]
        la_2 =  request.get_json()["layer_activations_2"]
        
        correlation = channel_correlation(la_1, la_2)
#

'''
Given a layer (of your choosing), perform max-pooling over space, giving a vector of activations over channels for each sample. Perform UMAP to compute a 2D projection.
'''
def get_layer_clustering(layer_num):
    if layer_num==2:
        return a2_clustering
    elif layer_num==3:
        return a3_clustering
    elif layer_num==4:
        return a4_clustering
    else:
        return a2_clustering

def get_layer_activation(layer_num):
    return data_dict[f"act{layer_num}"]

def get_2D_projection(layer_num):
    layer_activations = get_layer_activation(layer_num)
    # Max pooling
    max_pooled_activations = np.max(layer_activations, axis=(2, 3))
    
    # Standardize the pooled activations
    scaler = StandardScaler()
    scaled_activations = scaler.fit_transform(max_pooled_activations)
    
    # Perform PCA for dimensionality reduction
    #pca = PCA(n_components=50)  # Adjust the number of components as needed
    #reduced_activations = pca.fit_transform(scaled_activations)
    
    # Apply UMAP for 2D projection
    umap = UMAP(n_components=2)
    embedding = umap.fit_transform(scaled_activations)
    print("results embedding:", embedding.shape)
    projection = embedding.tolist()
    
    return projection

@app.route('/channel_dr', methods=['GET','POST'])
def channel_dr():

    layer_activations = 2
    
    if request.method == 'POST':
        try:
            layer_activations =  request.get_json()["layer_activations"]
            print("Layer selected", layer_activations)
        except Exception as e:
            print("ERROR", e)

    projection = get_layer_clustering(layer_activations)
    
    #umap_projection = [ {"pc0": a, "pc1": b} for a,b in projection ] 
        
    return flask.jsonify({"projection": projection})
#

'''
TODO

Compute correlation strength over selected instances, those brushed by the user.
'''
@app.route('/selected_correlation', methods=['GET','POST'])
def selected_correlation():
    pass
#

'''
TODO

Compute correlation strength over all instances.
'''
@app.route('/activation_correlation_clustering', methods=['GET'])
def activation_correlation_clustering():
    pass

'''
Get generated images.
'''
@app.route('/get_image_path', methods=['GET'])
def get_image_path():
    
    return flask.jsonify(data_dict["images"])

@app.route('/get_image_url', methods=['GET'])
def get_image_url():

    img_url = [  f"http://localhost:8080/static/{img.split('/')[-1]}" for img in  data_dict["images"] ]
    
    return flask.jsonify(img_url)

@app.route('/image/<img_name>', methods=['GET'])
def get_image(img_name):

    return redirect(url_for("static", filename=img_name))

'''
Get clusters !!! Entermate:
'''
@app.route('/get_clusters', methods=['GET'])
def get_clusters():
    cluster_2_count = dict(Counter(S_2_cluster))
    cluster_3_count = dict(Counter(S_3_cluster))
    cluster_4_count = dict(Counter(S_4_cluster))

    return_dict = {
            "layer_2": cluster_2_count,
            "layer_3": cluster_3_count,
            "layer_4": cluster_4_count,
    }

    return flask.jsonify(return_dict)

'''
In the main, before running the server, run clustering, store results in variables a2_clustering, a3_clustering, a4_clustering
'''
if __name__=='__main__':

    print("Clustering 2 ...")
    a2_clustering = get_2D_projection(2)
    print("Clustering 3 ...")
    a3_clustering = get_2D_projection(3)
    print("Clustering 4 ...")
    a4_clustering = get_2D_projection(4)

    print("Clustering channels ...")
    S_2_cluster, S_3_cluster, S_4_cluster = multiway_spectral_clustering(iou_23, iou_34)

    print("Initializing server ...")
    app.run(port=8080)


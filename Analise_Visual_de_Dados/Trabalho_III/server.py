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
from skimage.measure import block_reduce

from collections import Counter, OrderedDict

from flask import Flask, request, redirect, url_for
from flask_cors import CORS
from pathlib import Path
import copy
import math
import joblib
import itertools

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

S_2 = data_dict["S_2"]
S_3 = data_dict["S_3"]
S_4 = data_dict["S_4"]

img_url_data = [  f"http://localhost:8080/static/{img.split('/')[-1]}" for img in  data_dict["images"] ]

'''
act2_thres = data_dict["act2_thres"]
act3_thres = data_dict["act3_thres"]
act3_up_thres = data_dict["act3_up_thres"]
act4_thres = data_dict["act4_thres"]

print(np.unique(act2_thres))
print(np.unique(act3_thres))
print(np.unique(act3_up_thres))
print(np.unique(act4_thres))
'''

# number of clusters - feel free to adjust
n_clusters = 9

pool_size = (2, 2)

# these variables will contain the clustering of channels for the different layers
a2_clustering,a3_clustering,a4_clustering = None, None, None
S_2_cluster, S_3_cluster, S_4_cluster = None, None, None
cluster_2_count, cluster_3_count, cluster_4_count = None, None, None
all_correlation_23, all_correlation_34 = None, None

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
    print("cluster 2", S_2_cluster.shape)
    print("scratch", S_2_cluster)
    #S_2_sklearn = SpectralClustering(n_clusters=n_clusters).fit_predict(S_2_standard)
    #print("sklearn", S_2_sklearn)

    S_3_cluster = spectral_clustering(S_3_standard)
    print("cluster 3", S_3_cluster.shape)
    print("scratch", S_3_cluster)
    #S_3_sklearn = SpectralClustering(n_clusters=n_clusters).fit_predict(S_3_standard)
    #print("sklearn", S_3_sklearn)
    
    S_4_cluster = spectral_clustering(S_4_standard)
    print("cluster 4", S_4_cluster.shape)
    print("scratch", S_4_cluster)
    #S_4_sklearn = SpectralClustering(n_clusters=n_clusters).fit_predict(S_4_standard)
    #print("sklearn", S_4_sklearn)

    return S_2_cluster, S_3_cluster, S_4_cluster

'''
TODO
Given a link selected from the visualization, namely the layer and clusters at the individual layers, this route should compute the mean correlation from all channels in the source layer and all channels in the target layer, for each sample.
'''
def sort_dict(dict_):
    sorted_keys = sorted(dict_.keys())
    
    new_dict = {}
    for key in sorted_keys:
        new_dict[key] = dict_[key]
    
    return new_dict

def link_channel_correlation(iou_ab, channel_a, channel_b, cluster_a, cluster_b, layer_name=""):

    n_samples = iou_ab.shape[0]
    
    cluster_a_count = sort_dict(dict(Counter([str(num) for num in cluster_a])))
    cluster_a_label = list(cluster_a_count.keys())
    
    cluster_b_count = sort_dict(dict(Counter([str(num) for num in cluster_b])))
    cluster_b_label = list(cluster_b_count.keys())
    
    indexes_a = list(np.where(cluster_a == int(channel_a))[0])
    indexes_b = list(np.where(cluster_b == int(channel_b))[0])
    
    Z = n_samples*cluster_a_count[channel_a]*cluster_b_count[channel_b]
    
    indexes_samples = np.array(tuple(itertools.product(indexes_a, indexes_b)))
    samples_used = iou_ab[:, indexes_samples[:, 0], indexes_samples[:, 1]]
    samples_sum = samples_used.sum(axis=1)*1.0/Z

    indexes_ab = np.array(tuple(itertools.product(list(range(n_samples)), indexes_a, indexes_b)))    
    total_channels = iou_ab[indexes_ab[:, 0], indexes_ab[:, 1], indexes_ab[:, 2]]
    total_sum = total_channels.sum()*1.0/Z

    new_data = {}
    new_data["layer_name"] = layer_name
    new_data["source"] = int(channel_a)
    new_data["source_count"] = cluster_a_count[channel_a]
    new_data["target"] = int(channel_b)
    new_data["target_count"] = cluster_b_count[channel_b]
    new_data["correlation"] = total_sum
    new_data["samples_total_sum"] = total_sum*Z
    new_data["samples_used"] = n_samples
    new_data["samples_correlation"] = samples_sum.tolist()
    new_data["Z"] = Z
    
    return new_data

@app.route('/link_score', methods=['GET','POST'])
def link_score():
    selected_ids = None
    channel_a = "0"
    channel_b = "0"
    if request.method == 'POST':
        try:
            selected_ids =  request.get_json()["selected_ids"]
            channel_a =  request.get_json()["channel_a"]
            channel_b =  request.get_json()["channel_b"]
            layer_name = request.get_json()["layer_name"]
            print("Selected samples", selected_ids, channel_a, channel_b, layer_name)
        except Exception as e:
            print("ERROR", e)
    
    if selected_ids is not None:
        iou_ab = iou_matrix[selected_ids].copy()
    else:
        iou_ab = iou_matrix.copy()
        selected_ids = list(range(iou_ab.shape[0]))
        
    new_data = link_channel_correlation(iou_ab, channel_a, channel_b, S_2_cluster, S_3_cluster, layer_name=layer_name)
    new_data["samples"] = selected_ids
    
    return flask.jsonify(new_data)
    
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

def apply_max_pooling(images, pool_size=(2, 2)):
    pool_images = []
    for img in images:
        #pooled_img = np.array([maximum_filter(ch, size=pool_size) for ch in img])
        #pooled_img = maximum_filter(img, size=pool_size)
        pooled_img = np.array([block_reduce(channel, pool_size, np.max) for channel in img])
        #pooled_img = np.max(img, axis=(1, 2))
        pool_images.append(pooled_img)
    return np.array(pool_images)

def get_2D_projection(layer_num):
    layer_activations = get_layer_activation(layer_num)
    print("Layer activation:", layer_activations.shape)
    # Max pooling
    #max_pooled_activations = np.max(layer_activations, axis=(2, 3))
    max_pooled_activations = apply_max_pooling(layer_activations, pool_size)
    flattened_images = max_pooled_activations.reshape(max_pooled_activations.shape[0], -1)
    print("after Pooling Layer activation:", max_pooled_activations.shape, flattened_images.shape)
    
    # Standardize the pooled activations
    scaler = StandardScaler()
    scaled_activations = scaler.fit_transform(flattened_images)
    
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
Compute correlation strength over selected instances, those brushed by the user.
'''
def channel_correlation(iou_matrix, cluster_a, cluster_b, layer_name="", selected_ids=None):
    if selected_ids is not None:
        iou_ab = iou_matrix[selected_ids].copy()
    else:
        iou_ab = iou_matrix.copy()
        selected_ids = list(range(iou_ab.shape[0]))

    cluster_a_count = sort_dict(dict(Counter([str(num) for num in cluster_a])))
    cluster_a_label = list(cluster_a_count.keys())
    
    cluster_b_count = sort_dict(dict(Counter([str(num) for num in cluster_b])))
    cluster_b_label = list(cluster_b_count.keys())

    result_data = []
    
    for channel_a in cluster_a_label:
        for channel_b in cluster_b_label:
            
            new_data = link_channel_correlation(iou_ab, channel_a, channel_b, cluster_a, cluster_b, layer_name=layer_name)
            
            new_data["samples"] = selected_ids
            
            result_data.append(new_data)
    
    return result_data

@app.route('/selected_correlation', methods=['GET','POST'])
def selected_correlation():
    selected_ids = None
    if request.method == 'POST':
        try:
            selected_ids =  request.get_json()["selected_ids"]
            print("Selected samples", selected_ids)
        except Exception as e:
            print("ERROR", e)
    
    if selected_ids is None:
        selected_correlation_23 = copy.deepcopy(all_correlation_23)
        selected_correlation_34 = copy.deepcopy(all_correlation_34)
    else:
        selected_correlation_23 = channel_correlation(iou_23, S_2_cluster, S_3_cluster, selected_ids=selected_ids, layer_name="layers_23")
        
        selected_correlation_34 = channel_correlation(iou_34, S_3_cluster, S_4_cluster, selected_ids=selected_ids, layer_name="layers_34")
    
    selected_correlation_23.extend(selected_correlation_34)
    
    return flask.jsonify(selected_correlation_23)

'''
Compute correlation strength over all instances.
'''
@app.route('/activation_correlation_clustering', methods=['GET'])
def activation_correlation_clustering():

    all_correlation_234 = copy.deepcopy(all_correlation_23)
    all_correlation_234.extend(copy.deepcopy(all_correlation_34))
    
    return flask.jsonify(all_correlation_234)

'''
Get generated images.
'''
@app.route('/get_image_path', methods=['GET'])
def get_image_path():
    
    return flask.jsonify(data_dict["images"])

@app.route('/get_image_url', methods=['GET', 'POST'])
def get_image_url():

    selected_ids = None
    if request.method == 'POST':
        try:
            selected_ids =  request.get_json()["selected_ids"]
            print("Selected samples", selected_ids)
        except Exception as e:
            print("ERROR", e)
    
    if selected_ids is not None:
        img_url = [img_url_data[i] for i in selected_ids]
    else:
        img_url = img_url_data.copy() 
    
    return flask.jsonify(img_url)

@app.route('/image/<img_name>', methods=['GET'])
def get_image(img_name):

    return redirect(url_for("static", filename=img_name))

'''
Get clusters !!! Entermate:
'''
@app.route('/get_clusters', methods=['GET'])
def get_clusters():

    return_dict = [
            {"layer_2": cluster_2_count},
            {"layer_3": cluster_3_count},
            {"layer_4": cluster_4_count},
    ]
    print(return_dict)

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
    
    print("Calculating clusters frequencies ...")
    cluster_2_count = sort_dict(dict(Counter([str(num) for num in S_2_cluster])))
    print(cluster_2_count)
    cluster_3_count = sort_dict(dict(Counter([str(num) for num in S_3_cluster])))
    print(cluster_3_count)
    cluster_4_count = sort_dict(dict(Counter([str(num) for num in S_4_cluster])))
    print(cluster_4_count)

    print("Processing channel correlations ...")
    all_correlation_23 = channel_correlation(iou_23, S_2_cluster, S_3_cluster, layer_name="layers_23")
    all_correlation_34 = channel_correlation(iou_34, S_3_cluster, S_4_cluster, layer_name="layers_34")

    print("Initializing server ...")
    app.run(port=8080, use_reloader=False, threaded=True)


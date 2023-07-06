import os
import flask
import numpy as np
import time
from PIL import Image

from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.sparse import csgraph
from sklearn.neighbors import radius_neighbors_graph, kneighbors_graph
from skimage.measure import block_reduce
from scipy.ndimage import maximum_filter

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

from py.utils import verifyDir, get_current_path
from pathlib import Path

PORT=9090
CITY="Rio de Janeiro"

absolute_current_path = Path().resolve()
# create Flask app
verifyDir(f"{absolute_current_path}/outputs/static/")
app = Flask(__name__, static_folder=f"{absolute_current_path}/outputs/static/")
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
CORS(app)

# load all of the data generated from preprocessing
data_dict = joblib.load(f"{absolute_current_path}/outputs/static/data.joblib")
data_dict["images"] = []
images = np.array(data_dict["images"]).copy()
features = np.array(data_dict["features"]).copy()
cities = data_dict["city"].copy()
latitude = data_dict["latitude"].copy()
longitude = data_dict["longitude"].copy()
labels = data_dict["label"].copy()
scores = data_dict["safety"].copy()

img_url_data = [  f"http://localhost:{PORT}/static/images/{img.split('/')[-2]}/{img.split('/')[-1]}" for img in  data_dict["path"] ]

# these variables will contain the clustering of channels for the different layers
clustering_dict = {}

@app.after_request
def add_header(r):
    r.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    r.headers['Pragma'] = 'no-cache'
    r.headers['Expires'] = '0'
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

# IMAGES
@app.route('/get_image_information/', methods=['GET', 'POST'])
def get_image_information():

    selected_ids = None
    if request.method == 'POST':
        try:
            selected_ids =  request.get_json()["selected_ids"]
            print("Selected samples", selected_ids)
        except Exception as e:
            print("ERROR", e)

    img_info = [ {"id": index,"city": cit, "lat": lat, "long": long_, "label": lab, "score": scor} for index, (cit, lat, long_, lab, scor) in enumerate(zip(cities, latitude, longitude, labels, scores)) ]
    
    return flask.jsonify(img_info)

@app.route('/get_image_label', methods=['GET', 'POST'])
def get_image_label():

    selected_ids = None
    if request.method == 'POST':
        try:
            selected_ids =  request.get_json()["selected_ids"]
            print("Selected samples", selected_ids)
        except Exception as e:
            print("ERROR", e)
    
    if selected_ids is not None:
        label_list = [i for i in selected_ids]
        img_label = labels[label_list]
    else:
        img_label = labels.copy() 
    
    return flask.jsonify(img_label)

@app.route('/get_image_score', methods=['GET', 'POST'])
def get_image_score():

    selected_ids = None
    if request.method == 'POST':
        try:
            selected_ids =  request.get_json()["selected_ids"]
            print("Selected samples", selected_ids)
        except Exception as e:
            print("ERROR", e)
    
    if selected_ids is not None:
        label_list = [i for i in selected_ids]
        img_score = scores[label_list]
    else:
        img_score = scores.copy() 
    
    return flask.jsonify(img_score)

@app.route('/get_image_path', methods=['GET'])
def get_image_path():
    
    return flask.jsonify(data_dict["path"])

@app.route('/get_image_url/<current_city>/', methods=['GET', 'POST'])
def get_image_url(current_city):

    selected_ids = None

    if request.method == 'GET':
        if current_city!="all" and current_city is not None:
            img_url = [img for img in img_url_data if current_city in img].copy()
        else:
            img_url = img_url_data.copy()

    if request.method == 'POST':
        try:
            selected_ids =  request.get_json()["selected_ids"]
            print("Selected samples", selected_ids)
            if selected_ids is not None:
                img_url = [img_url_data[i] for i in selected_ids]
            else:
                if current_city!="all" and current_city is not None:
                    img_url = [img for img in img_url_data if current_city in img].copy()
                else:
                    img_url = img_url_data.copy()
        except Exception as e:
            print("ERROR", e)

    return flask.jsonify(img_url)

@app.route('/image/<img_name>/', methods=['GET'])
@app.route('/image/<city>/<img_name>/', methods=['GET'])
def get_image(img_name, city="Rio De Janeiro"):

    name_file = "images/painting.jpg"
    try:
        current_image = [_name for _name in data_dict["path"] if img_name in _name][0].split("/")[-1]
        name_file = f"images/{city}/{current_image}"
        print("IMAGE:", name_file)
    except Exception as e:
        print("ERROR", e)

    return redirect(url_for("static", filename=name_file))

def apply_max_pooling(images, pool_size=(4, 4)):
    pool_images = []
    for img in images:
        #pooled_img = np.array([maximum_filter(ch, size=pool_size) for ch in img])
        #pooled_img = maximum_filter(img, size=pool_size)
        pooled_img = np.array([block_reduce(channel, pool_size, np.max) for channel in img])
        #pooled_img = np.max(img, axis=(1, 2))
        pool_images.append(pooled_img)
    return np.array(pool_images)

def get_umap_2D_projection(images, type_data="features"):

    print("Input", images.shape)
    if type_data=="features":
        flattened_images = images.copy()
        flattened_images = (flattened_images - np.mean(flattened_images, axis=0)) / np.std(flattened_images, axis=0)
    else:
        #images_pooled = block_reduce(images, pool_size, np.max)
        images_pooled = apply_max_pooling(images)
        flattened_images = images_pooled.reshape(images_pooled.shape[0], -1)
        flattened_images = flattened_images / 255.0
    
    print("Flatten", flattened_images.shape)
    # Standardize the pooled activations
    #scaler = StandardScaler()
    #scaled_activations = scaler.fit_transform(flattened_images)
        
    # Apply UMAP for 2D projection
    umap = UMAP(n_components=2)
    embedding = umap.fit_transform(flattened_images)
    print("results embedding:", embedding.shape)
    projection = embedding.tolist()
    
    return projection

def get_tsne_2D_projection(images, type_data="features"):

    print("Input", images.shape)
    if type_data=="features":
        flattened_images = images.copy()
        flattened_images = (flattened_images - np.mean(flattened_images, axis=0)) / np.std(flattened_images, axis=0)
    else:
        flattened_images = images.reshape(images.shape[0], -1)
        flattened_images = flattened_images / 255.0

    print("Flatten", flattened_images.shape)
    # Normalize the flattened images to have zero mean and unit variance (optional but recommended)

    # Apply t-SNE to the normalized flattened images
    tsne = TSNE(n_components=2)  # You can adjust the number of components as needed
    tsne_images = tsne.fit_transform(flattened_images)
    print("results embedding:", tsne_images.shape)
    projection = tsne_images.tolist()
    
    return projection

def get_pca_2D_projection(images, type_data="features"):

    print("Input", images.shape)
    if type_data=="features":
        flattened_images = images.copy()
        flattened_images = (flattened_images - np.mean(flattened_images, axis=0)) / np.std(flattened_images, axis=0)
    else:
        flattened_images = images.reshape(images.shape[0], -1)
        flattened_images = flattened_images / 255.0
    
    print("Flatten", flattened_images.shape)
    pca = PCA(n_components=2)  # Specify the desired number of components
    pca_images = pca.fit_transform(flattened_images)
    
    # Access the principal components
    principal_components = pca.components_

    # Access the explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_
    print("results embedding:", pca_images.shape)
    projection = pca_images.tolist()
    return projection

@app.route('/clustering/<type_cluster>/', methods=['GET','POST'])
def get_clusters(type_cluster="umap", current_city="Boston"):

    current_feature = "features"
    current_city = None

    if request.method == 'POST':
        try:
            current_city =  request.get_json()["city"]
            current_feature =  request.get_json()["features"]
        except Exception as e:
            print("ERROR", e)
        
    if current_city!="all" and current_city is not None:
        indexes = [index for index, img in enumerate(img_url_data) if current_city in img].copy()
    else:
        indexes = list(range(len(img_url_data))).copy()

    current_projection = clustering_dict[f"{current_feature}_{type_cluster}_clustering"].copy()
    projection = [ proj for index, proj in enumerate(current_projection) if index in indexes ]

    return flask.jsonify({"projection": projection})

'''
In the main, before running the server, run clustering, store results in variables a2_clustering, a3_clustering, a4_clustering
'''
if __name__=='__main__':

    print("Clustering UMAp ...")
    #clustering_dict["images_umap_clustering"] = get_umap_2D_projection(images, type_data="images")
    clustering_dict["features_umap_clustering"] = get_umap_2D_projection(features, type_data="features")

    print("Clustering PCA ...")
    #clustering_dict["images_pca_clustering"] = get_pca_2D_projection(images, type_data="images")
    clustering_dict["features_pca_clustering"] = get_pca_2D_projection(features, type_data="features")

    print("Clustering t-SNE ...")
    #clustering_dict["images_tsne_clustering"] = get_tsne_2D_projection(images, type_data="images")
    clustering_dict["features_tsne_clustering"] = get_pca_2D_projection(features, type_data="features")
    
    print("Initializing server ...")
    app.run(port=PORT, use_reloader=False, threaded=True)
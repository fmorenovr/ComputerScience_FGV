import os
import flask
import numpy as np
import argparse
import json
import csv

from flask import Flask, request, redirect, url_for
from flask_cors import CORS
from pathlib import Path
import math

from sklearn.cluster import KMeans

def get_current_path():
    return str(Path(__file__).parent.resolve())

absolute_current_path = get_current_path()

# create Flask app
app = Flask(__name__, static_folder=absolute_current_path+'/static')
CORS(app)

#CORS(app, supports_credentials=True, resources={r"/*": {"origins": "*"}})

# --- these will be populated in the main --- #
def ccPCA(data, targets , n_components=2, alpha=0.5):

    if len(targets)==0:
        X_reduced, loadings = PCA(data, n_components=2)
        return X_reduced, loadings
    
    #Step-0: create targets 
    X_R = np.delete(data.copy(), targets, axis=0)
    R = data[targets]
    
    #Step-1 A: concat
    X = np.concatenate((X_R, R), axis=0)
    print("X and R Matrix:", X.shape, X_R.shape, R.shape)

    #Step-1 B: Standarize
    x_mean = np.mean(X, axis = 0)
    x_mean_square = np.mean(X**2, axis = 0)
    X_meaned = ( X - x_mean ) / np.sqrt(x_mean_square)

    r_mean = np.mean(R, axis = 0)
    r_mean_square = np.mean(R**2, axis = 0)
    R_meaned = ( R - r_mean ) / np.sqrt(r_mean_square)
    
    #Step-1 C: Evaluate if is finite
    x_finite = np.isfinite(X_meaned)
    X_meaned[x_finite!=True] = 0.
    
    r_finite = np.isfinite(R_meaned)
    R_meaned[r_finite!=True] = 0.
    print("X-meaned and R-meaned Matrix:", X_meaned.shape, R_meaned.shape)

    #Step-2: Covarianza
    cov_mat_X = np.cov(X_meaned , rowvar = False)
    cov_mat_R = np.cov(R_meaned , rowvar = False)
    cov_mat = np.cov(cov_mat_X - alpha*cov_mat_R , rowvar = False)
    print("Covariance Matrix:", cov_mat.shape)

    #Step-3: eigen solver
    _, sigmas, __ = np.linalg.svd(X_meaned)
    
    eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)
     
    #Step-4
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:,sorted_index]
     
    #Step-5_A: components
    pca_components = sorted_eigenvectors[:,0:n_components]
    print("components:", pca_components.shape)
    #print("sratch components:", pca_components)
   
    #Step-5_B: Explained variances
    #explained_variances_ratios = [value / np.sum(sorted_eigenvalue) for value in sorted_eigenvalue]
    explained_variances = sorted_eigenvalue[0:n_components]

    #Step-5_C: Sigmas
    singular_values = sigmas[0:n_components]
     
    #Step-6: Projections
    X_reduced = np.dot(pca_components.transpose() , X_meaned.transpose() ).transpose()
    print("X-reduced Matrix:", X_reduced.shape)
    
    #Step-7: Loadings
    loadings = pca_components*singular_values

    return X_reduced, loadings

def PCA(X , n_components=2):
    
    print("X Matrix:", X.shape) 
    #Step-1
    X_meaned = X - np.mean(X , axis = 0)
    print("X-meaned Matrix:", X_meaned.shape)
     
    #Step-2
    cov_mat = np.cov(X_meaned , rowvar = False)
    print("Covariance Matrix:", cov_mat.shape)
     
    #Step-3
    _, sigmas, __ = np.linalg.svd(X_meaned)

    eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)
     
    #Step-4
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:,sorted_index]
     
    #Step-5_A: components
    pca_components = sorted_eigenvectors[:,0:n_components]
    #print("sratch components:", pca_components)
    print("components:", pca_components.shape)
   
    #Step-5_B: Explained variances
    #explained_variances_ratios = [value / np.sum(sorted_eigenvalue) for value in sorted_eigenvalue]
    explained_variances = sorted_eigenvalue[0:n_components]

    #Step-5_C: Sigmas
    singular_values = sigmas[0:n_components]
     
    #Step-6: Projections
    X_reduced = np.dot(pca_components.transpose() , X_meaned.transpose() ).transpose()
    print("X-reduced Matrix:", X_reduced.shape)
    
    #Step-7: Loadings
    loadings = pca_components*singular_values

    return X_reduced, loadings

# list of attribute names of size m
attribute_names=json.load(open('attribute_names.json','r'))

# a 2D numpy array containing binary attributes - it is of size n x m, for n paintings and m attributes
painting_attributes=np.load('painting_attributes.npy')

# a list of epsiode names of size n
episode_names=json.load(open('episode_names.json','r'))

# a list of painting image URLs of size n
painting_image_urls=json.load(open('painting_image_urls.json','r'))

#print(painting_attributes, painting_attributes.shape)

'''
This will return an array of strings containing the episode names -> these should be displayed upon hovering over circles.
'''
@app.route('/get_episode_names', methods=['GET'])
def get_episode_names():
    return flask.jsonify(episode_names)
#

'''
This will return an array of URLs containing the paths of images for the paintings
'''
@app.route('/get_painting_urls', methods=['GET'])
def get_painting_urls():
    img_urls = [ f"{absolute_current_path}/{img_path}" for img_path in  painting_image_urls]
    
    img_urls = [ img_path.replace("\\", "/") for img_path in  img_urls]

    img_urls = [  f"http://localhost:5000/static/{img.split('/')[-1]}" for img in  img_urls ]

    #return flask.jsonify({"img_urls": img_urls, "img": img_encoded})
    return flask.jsonify(img_urls)
#

@app.route('/image/<img_name>', methods=['GET'])
def get_image(img_name):

    return redirect(url_for("static", filename=img_name))

'''
TODO: implement PCA, this should return data in the same format as you saw in the first part of the assignment:
    * the 2D projection
    * x loadings, consisting of pairs of attribute name and value
    * y loadings, consisting of pairs of attribute name and value
'''
@app.route('/initial_pca', methods=['GET'])
def initial_pca():
    #pca = decomposition.PCA(n_components=2, svd_solver='full')
    #new_attributes = pca.fit_transform(painting_attributes)
    #loadings_ = pca.components_.T* pca.singular_values_
    #print("sklearn.PCA components", pca.components_.T)
    #print("sklearn PCA exxplain variance", pca.explained_variance_)
    #new_attributes[:,0] = -1*new_attributes[:,0]
    #print("singular_val", pca.singular_values_)
    #print("sklearn PCA loadings", loadings_)
    new_projection, loadings = PCA(painting_attributes, n_components=2)
    x_loadings = [ {"attribute": a, "loading": b} for a,b in zip(attribute_names, loadings[:,0].copy()) ] 
    y_loadings = [ {"attribute": a, "loading": b} for a,b in zip(attribute_names, loadings[:,1].copy()) ] 
    
    return flask.jsonify({"loading_x": x_loadings, "loading_y":y_loadings, "projection": new_projection.tolist()})
#

'''
TODO: implement ccPCA here. This should return data in _the same format_ as initial_pca above.
It will take in a list of data items, corresponding to the set of items selected in the visualization. This can be acquired from `flask.request.json`. This should be a list of data item indices - the **target set**.
The alpha value, from the paper, should be set to 1.1 to start, though you are free to adjust this parameter.
'''
@app.route('/ccpca', methods=['GET','POST'])
def ccpca():

    targets = [];
    custom_alpha = 0.5;

    if request.method == 'POST':
        try:
            targets =  request.get_json()["targets"]
            custom_alpha =  request.get_json()["alpha"]
            print("entro pe", targets, "alpha", custom_alpha)
        except Exception as e:
            print("ERROR", e)
        
    new_projection, loadings = ccPCA(painting_attributes, targets, alpha=custom_alpha, n_components=2)
    
    x_loadings = [ {"attribute": a, "loading": b} for a,b in zip(attribute_names, loadings[:,0].copy()) ] 
    y_loadings = [ {"attribute": a, "loading": b} for a,b in zip(attribute_names, loadings[:,1].copy()) ]

    return flask.jsonify({"loading_x": x_loadings, "loading_y": y_loadings, "projection": new_projection.tolist()})
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
    kmeans = KMeans(n_clusters=6, random_state=0, init="random").fit(painting_attributes)
    labels = kmeans.labels_
    #print(labels, labels.shape)
    
    kmeans_data = []
    
    for i in range(len(painting_attributes)):
        for j, att in enumerate(attribute_names):
            kmeans_data.append({"attribute": att, 
                                "id": i, 
                                "label": int(labels[i]), 
                                "value": int(painting_attributes[i][j])})
        
    return flask.jsonify(kmeans_data)
#

if __name__=='__main__':

    app.run(port=5000)
#

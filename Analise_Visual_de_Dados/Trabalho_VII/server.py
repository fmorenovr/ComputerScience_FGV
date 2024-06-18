import os
import flask
from flask import Flask, request, redirect, url_for
from flask_cors import CORS
from pathlib import Path
import numpy as np
import time
from PIL import Image
from tqdm import tqdm
import copy
import math
import joblib
import itertools
from matplotlib import pyplot as plt
from scipy import ndimage
from collections import Counter, OrderedDict

from numpy import argmax
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

from tensorflow.keras.datasets import mnist

from sklearn.linear_model import Ridge
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

X, y = make_regression(n_samples=500, n_features=10, noise=1, random_state=42)


from file_functions import verifyDir, get_current_path

model = load_model('model_99.142.h5')
layer_outputs = [layer.output for layer in model.layers if isinstance(layer, Conv2D)]
activation_model = Model(inputs=model.input, outputs=layer_outputs)

print(model.summary())

img_url_data = []
img_all_activations = []
x_train, x_test=None, None

absolute_current_path = get_current_path()

# create Flask app
verifyDir(f"{absolute_current_path}/static/images/")
verifyDir(f"{absolute_current_path}/static/layers/")
app = Flask(__name__, static_folder=absolute_current_path+'/static')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
CORS(app)
APP_PORT=8080


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

def load_dataset():
    (x_train, _), (x_test, _) = mnist.load_data()
    
    for i, img in enumerate(x_test[:100]):
      img_path = f"static/images/mnist_{i}.png"
      Image.fromarray(img).save(img_path)
      
      img_url_data.append( f"http://localhost:{APP_PORT}/static/images/mnist_{i}.png" )
      
    return x_train, x_test

def save_activations(activations, selected_id):
    img_activations = []
    for layer_num, activation in enumerate(activations):
        num_filters = activation.shape[-1]
        for filter_num in range(num_filters):
            img = activation[0, :, :, filter_num]

            # Normalize the image to the range [0, 255]
            img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
            img = img.astype(np.uint8)
            
            img_path = f"{absolute_current_path}/static/layers/mnist_{selected_id}_conv_layer_{layer_num}_filter_{filter_num}.png"
            img_activations.append(f"http://localhost:{APP_PORT}/static/layers/mnist_{selected_id}_conv_layer_{layer_num}_filter_{filter_num}.png")
            Image.fromarray((img * 255).astype(np.uint8)).save(img_path)
    
    return img_activations


@app.route('/activations/', methods=['GET','POST'])
def get_activation():
    selected_id = None
    if request.method == 'POST':
        try:
            selected_id =  request.get_json()["selected_ids"]
            print("Selected samples", selected_id)
        except Exception as e:
            print("ERROR", e)
    
    if selected_id is not None:
        img_test = f"{absolute_current_path}/static/images/mnist_{selected_id}.png"
        image = load_img(img_test, target_size=(28, 28), color_mode="grayscale")
        image_array = img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0)
        activations_ = activation_model.predict(image_array)
        img_url = save_activations(activations_, selected_id)
    else:
        img_url = img_all_activations
    
    return flask.jsonify({"layer_1": img_url[:32], "layer_2": img_url[32:96], "layer_3": img_url[96:]})

@app.route('/get_images_url/', methods=['GET', 'POST'])
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
In the main, before running the server, run clustering, store results in variables a2_clustering, a3_clustering, a4_clustering
'''
if __name__=='__main__':

    x_train, x_test = load_dataset()
    activations = activation_model.predict(x_test)
    
    for layer_num, activation in enumerate(activations):
        num_filters = activation.shape[-1]
        for filter_num in range(num_filters):
            img = activation[0, :, :, filter_num]

            # Normalize the image to the range [0, 255]
            img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
            img = img.astype(np.uint8)

            # Save the image
            img_path = f"{absolute_current_path}/static/layers/all_conv_layer_{layer_num}_filter_{filter_num}.png"
            img_all_activations.append(f"http://localhost:{APP_PORT}/static/layers/all_conv_layer_{layer_num}_filter_{filter_num}.png")
            Image.fromarray(img).save(img_path)
    
    
    print("Initializing server ...")
    app.run(port=APP_PORT, debug=True, use_reloader=True, threaded=True)


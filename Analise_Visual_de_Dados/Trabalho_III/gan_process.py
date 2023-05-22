import os
import numpy as np
import time
from PIL import Image
import joblib

from tqdm import tqdm

import torch

from models import MODEL_ZOO
from models import build_generator

from file_functions import verifyDir, get_current_path

absolute_current_path = get_current_path()

# if you have CUDA-enabled GPU, set this to True!
is_cuda = False

# StyleGAN tower
model_name = 'stylegan_tower256'
model_config = MODEL_ZOO[model_name].copy()
url = model_config.pop('url')  # URL to download model if needed.

# generator
generator = build_generator(**model_config)

# load the weights of the generator
checkpoint_path = os.path.join('checkpoints', model_name+'.pth')
checkpoint = torch.load(checkpoint_path, map_location='cpu')
if 'generator_smooth' in checkpoint:
    generator.load_state_dict(checkpoint['generator_smooth'])
else:
    generator.load_state_dict(checkpoint['generator'])
if is_cuda:
    generator = generator.cuda()
generator.eval()

'''
This draws a sample from the StyleGAN generator.
First we sample a random latent code.
Then we feed this through a generator neural network, producing a 3-channel RGB image, as well as activations at early layers.
In particular:
* `act2` are activations at layer 2 (0-indexing used here), giving us an 512x8x8 tensor of activations, e.g. 512 channels, 8x8 spatial resolution
* `act3` are activations at layer 3, giving us another 512x8x8 tensor of activations.
* `act3_up` is the result of bilinear upsampling of `act3` to a 512x16x16 tensor.
* `act4` are activations at layer 4, giving us a 512x16x16 tensor of activations.
'''
def sample_generator(n_samples=1):
    code = torch.randn(n_samples,generator.z_space_dim)
    print(code.size())
    if is_cuda:
        code = code.cuda()

    with torch.no_grad():
        # truncated normal distribution, no random noise in style layers!
        gen_out =  generator(code, trunc_psi=0.7,trunc_layers=8,randomize_noise=False)

        act2 = gen_out['act2'].detach()
        act3 = gen_out['act3'].detach()
        act3_up = torch.nn.functional.interpolate(act3,scale_factor=2,mode='bilinear',align_corners=True)
        act4 = gen_out['act4'].detach()

        image = gen_out['image'].detach()
    #

    return act2,act3,act3_up,act4,image
#

def tensor_to_numpy(tensor):
    return tensor.numpy()

'''
Postprocess images from the generator network - suitable to write to disk via PIL.
'''
def postprocess(images):
    scaled_images = (images+1)/2
    np_images = 255*scaled_images.numpy()
    np_images = np.clip(np_images + 0.5, 0, 255).astype(np.uint8)
    np_images = np_images.transpose(0, 2, 3, 1)
    return np_images
#

'''
Compute the Intersection-over-Union score between all pairs of channel activations for the provided tensors.
The tensors should be of the same spatial resolution. Further, the tensors should be comprised of values 0 or 1, derived from quantile-based thresholding.
This should return a tensor of shape (channel x channel)

NOTE: this can be done with a few lines of code using broadcasting! (no loops necessary)
'''
def iou(channel1, channel2):
    intersection = np.logical_and(channel1, channel2)
    union = np.logical_or(channel1, channel2)
    iou_score = np.sum(intersection, axis=(0, 1)) / np.sum(union, axis=(0, 1))
    return iou_score

'''
TODO

Given a tensor of activations (n_samples x channels x x-resolution x y-resolution), compute the per-channel top quantile (defined by perc), and then threshold activations
based on the quantile (perform this per channel)
'''
def threshold(tensor, k=4):
    # Calculate the per-channel top quantile
    quantiles = np.percentile(tensor, k, axis=(0, 2, 3))
    print(quantiles.shape)
    # Threshold activations based on the quantile
    thresholded_tensor = np.where(tensor > quantiles[:, np.newaxis, np.newaxis], 1, 0)

    return thresholded_tensor

#def threshold(acts,k=4):
#    channel_thresholds = np.percentile(acts, k, axis=(0, 1, 2))
#    return channel_thresholds

#    threshold = 1

#    acts[acts >= threshold] = 1
#    acts[acts < threshold] = 0

#    return acts
#

def generate_samples(n_samples=20):
    act2,act3,act3_up,act4,image = sample_generator(n_samples=n_samples)
    #print(act2.size())
    #print(act3.size())
    #print(act3_up.size())
    #print(act4.size())
    #print(image.size())

    image_np = postprocess(image)

    act2_np = tensor_to_numpy(act2)
    act3_np = tensor_to_numpy(act3)
    act3_up_np = tensor_to_numpy(act3_up)
    act4_np = tensor_to_numpy(act4)

    # Threshold activations
    act2_thres = threshold(act2_np)
    act3_thres = threshold(act3_np)
    act3_up_thres = threshold(act3_up_np)
    act4_thres = threshold(act4_np)
    
    print(act2_np.shape)
    print(act3_np.shape)
    print(act3_up_np.shape)
    print(act4_np.shape)
    print(image_np.shape)

    data_dict = {}

    for i, t in tqdm(enumerate(zip(act2_np, act3_np, act3_up_np, act4_np, image_np))):
        image_filename = f"{absolute_current_path}/static/sample_{i+1}.png"
        data_dict[f"sample_{i+1}"] = {}

        act2_np_, act3_np_, act3_up_np_, act4_np_, image_np_ = t
        
        data_dict[f"sample_{i+1}"]["atc2"] = act2_np_
        data_dict[f"sample_{i+1}"]["atc3"] = act3_np_
        data_dict[f"sample_{i+1}"]["act3_up"] = act3_up_np_
        data_dict[f"sample_{i+1}"]["atc4"] = act4_np_
        
        img = Image.fromarray(image_np_, "RGB")
        img.save(image_filename)
    
    joblib.dump(data_dict, f"{absolute_current_path}/static/data.joblib")
    #print(data_dict)
'''
TODO

Preprocessing:
    1. Generate a set of samples from the generator network (see sample_generator above).
    2. Threshold channel activations at each layer.
    3. Compute IoU scores, for each sample, between all pairs of channels from layer 2 to layer 3, and layer 3 to layer 4 -> should produce 2 tensors of shape (n_samples x channels x channels).
    4. Postprocess images and write the images to disk.
    5. Write out the threhsolded activations, and IoU score tensors, to disk.

Write everything to the 'static' directory.
'''
if __name__=='__main__':

    generate_samples(30)

    


#
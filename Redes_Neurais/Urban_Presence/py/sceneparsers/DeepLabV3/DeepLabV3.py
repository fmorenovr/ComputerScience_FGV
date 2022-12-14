import pixellib
from pixellib.semantic import semantic_segmentation, labelAde20k_to_color_image
import cv2
from PIL import Image
import numpy as np
from .functions import ade20k_map_color_mask

MODEL_DIR="models/deeplabv3_xception_ade20k/deeplabv3_xception65_ade20k.h5"

def DeepLabV3_Xception():
  segment_image = semantic_segmentation()
  segment_image.load_ade20k_model(f"{MODEL_DIR}")
  model = segment_image.model2
  return model

def ProcessSegmentation(model, image_path, output_image_name=None,overlay=False, process_frame = False, verbose = None):            
    trained_image_width=512
    mean_subtraction_value=127.5
    if process_frame == True:
      image = image_path
    else:  
      image = np.array(Image.open(image_path))     
    
    # resize to max dimension of images from training dataset
    w, h, n = image.shape

    if n > 3:
      image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    
    image_overlay = image.copy()

    ratio = float(trained_image_width) / np.max([w, h])
    resized_image = np.array(Image.fromarray(image.astype('uint8')).resize((int(ratio * h), int(ratio * w))))
    resized_image = (resized_image / mean_subtraction_value) -1


    # pad array to square image to match training images
    pad_x = int(trained_image_width - resized_image.shape[0])
    pad_y = int(trained_image_width - resized_image.shape[1])
    resized_image = np.pad(resized_image, ((0, pad_x), (0, pad_y), (0, 0)), mode='constant')

    if verbose is not None:
      print("Processing image....")
    #run prediction
    res = model.predict(np.expand_dims(resized_image, 0))
    
    labels = np.argmax(res.squeeze(), -1)
    
    # remove padding and resize back to original image
    if pad_x > 0:
      labels = labels[:-pad_x]
    if pad_y > 0:
      labels = labels[:, :-pad_y]

    raw_labels = labels
    
    """Run here the new function"""
    _, masks = ade20k_map_color_mask(raw_labels)
    
    """ Access the unique class ids of the masks """
    unique_labels = np.unique(raw_labels)
    raw_labels = np.array(Image.fromarray(raw_labels.astype('uint8')).resize((h, w)))
    
    
    """ Convert indexed masks to boolean """
    raw_labels = np.ma.make_mask(raw_labels)
    segvalues = {"class_ids":unique_labels,  "masks":raw_labels}   
   
    #Apply segmentation color map
    labels = labelAde20k_to_color_image(labels)   
    labels = np.array(Image.fromarray(labels.astype('uint8')).resize((h, w)))
    new_img = cv2.cvtColor(labels, cv2.COLOR_RGB2BGR)
    
    
    if overlay == True:
      alpha = 0.7
      cv2.addWeighted(new_img, alpha, image_overlay, 1 - alpha,0, image_overlay)

      if output_image_name is not None:
        cv2.imwrite(output_image_name, image_overlay)
        #print("Processed Image saved successfully in your current working directory.")

      return segvalues, image_overlay, masks

        
    else:  
        if output_image_name is not None:
  
          cv2.imwrite(output_image_name, new_img)

          #print("Processed Image saved successfuly in your current working directory.")

        return segvalues, new_img, masks

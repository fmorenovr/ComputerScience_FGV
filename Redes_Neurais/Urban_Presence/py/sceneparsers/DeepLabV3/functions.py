
import numpy as np
from scipy.io import loadmat
import csv

def ade20k_map_color_label(colors_path="data/categories/ade20k/color150.mat", names_path="data/categories/ade20k/object150.csv"):
  colors = loadmat(colors_path)['colors']
  colors = np.concatenate([np.zeros(shape=(1,3)), colors])
  names = {0: "no class"}
  with open(names_path) as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
      names[int(row[0])] = row[5].split(";")[0]
  
  names[59] = "screen_door"
  names[131] = "screen_projection"

  return names, colors

def ade20k_map_color_mask(raw_mask):
    names, colors = ade20k_map_color_label("data/categories/ade20k/color150.mat", "data/categories/ade20k/object150.csv")
    uniques, counts = np.unique(raw_mask, return_counts=True)
    
    class_index = []
    masks = []
    ratios = []
    class_name = []
    class_color = []

    d_dict = []

    for idx in np.argsort(counts)[::-1]:
        index_label = uniques[idx]
        label_mask = raw_mask == index_label

        class_index.append(index_label)
        masks.append(label_mask)
        ratios.append(counts[idx]/raw_mask.size *100)
        class_name.append(names[index_label])
        class_color.append(colors[index_label])

        d_dict.append({"classes": index_label,
                        "names": names[index_label], 
                        "colors": colors[index_label], 
                        "masks": label_mask, 
                        "ratios": counts[idx]/raw_mask.size *100})
    
    d_segment = {"classes": class_index,
                 "names": class_name, 
                 "colors": class_color, 
                 "masks": label_mask, 
                 "ratios": ratios}

    return d_segment, d_dict


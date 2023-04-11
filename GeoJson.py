import solaris as sol
from solaris.solaris.data import data_dir 
from solaris.solaris.data import coco as sc
import os
import json

import glob


sample_geojsons = []
sample_images = []

json_path = '/home/ubuntu/d2/detectron2/coastal/geojson/'
img_path = '/home/ubuntu/d2/detectron2/coastal/images/'

for file in os.listdir(json_path):
    site_name = file.split('_')[0]
    sample_geojsons.append(os.path.join(json_path, file))
    sample_images.append(os.path.join(img_path, site_name + '.png'))
    print()

#sample_geojsons = os.path.join(data_dir, 'geotiff_labels.geojson')
#sample_images = os.path.join(data_dir, 'sample_geotiff.tif')

coco_dict = sc.geojson2coco(sample_images, sample_geojsons,
                                       category_attribute='truncated')

print(coco_dict)

from matplotlib import pyplot as plt
from matplotlib import patches
import skimage

im = skimage.io.imread(sample_images[0])
f, ax = plt.subplots(figsize=(10, 10))
ax.imshow(im, cmap='gray')
colors = ['', 'r', 'b']
for anno in coco_dict['annotations']:
    patch = patches.Rectangle((anno['bbox'][0], anno['bbox'][1]), anno['bbox'][2], anno['bbox'][3], linewidth=1, edgecolor=colors[anno['category_id']], facecolor='none')
    ax.add_patch(patch)



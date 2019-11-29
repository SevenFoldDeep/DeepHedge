# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 11:31:05 2019

@author: ahls_st
"""

from fileMover import match_files, annotate
from window_maker import find_hedges
from augmentation import Augmenter
import pycococreatortools as coco
import os
from filegrabber import getFiles
import rasterio
import re
import json
from random import randrange

'''READ ME:
Need to change the folder paths and possibly the number of bands. All lines 
that require changing are labelled with #change

Recommended folder structure
----------------------------
Root
  ->imgs_orig
    ->train
    ->val
  ->masks_orig
    ->train
    ->val
  ->augs
    ->img
    ->mask
'''


# Augment original training set of images
in_img = r'C:\Users\ahls_st\Documents\MasterThesis\IKONOS\With_Hedges\FourBands\Splits\imgs\train'   #change
aug_img_dir = r'D:\Steve\IKONOS\4band_geo_only\imgs'       #change
in_mask = r'C:\Users\ahls_st\Documents\MasterThesis\IKONOS\With_Hedges\ThreeBands\Splits\BGR\Masks\train'    #change
aug_mask_dir = r'D:\Steve\IKONOS\4band_geo_only\masks'   #change

# Creates augmented versions of the training data. Can choose to do single, or combined augmentations
augmenter = Augmenter(in_img, in_mask, '.png')
#augmenter.augment(aug_img_dir, aug_mask_dir, n_bands=3) # performs all augmentations as singles
augmenter.augment_combo(aug_img_dir, aug_mask_dir, n_bands=4, times=23, n_geo=2, n_spec=0)#performs geo_only augmentations
augmenter.random_crop(aug_img_dir, aug_mask_dir, n_bands=4, num=8) #randomly crops

# Get rid of hedge masks where only a little bit is showing at the edge of the image
# Best to inspect the removed files before continuing to make sure the erase threshold is ok

if not os.path.exists(os.path.join(aug_mask_dir, 'Cleaned')):
    os.mkdir(os.path.join(aug_mask_dir, 'Cleaned'))
    print('Creating {}'.format(os.path.join(aug_mask_dir, 'Cleaned')))
if not os.path.exists(os.path.join(aug_mask_dir, 'Removed')):
    os.mkdir(os.path.join(aug_mask_dir, 'Removed'))
    print('Creating {}'.format(os.path.join(aug_mask_dir, 'Removed')))
    
find_hedges(aug_mask_dir, os.path.join(aug_mask_dir, 'Cleaned'), os.path.join(aug_mask_dir, 'Removed'), erase_thres=200)

if not os.path.exists(os.path.join(aug_img_dir, 'Cleaned')):
    print('Creating {}'.format(os.path.join(aug_img_dir, 'Cleaned')))
    os.mkdir(os.path.join(aug_img_dir, 'Cleaned'))
    
# Match the training images with the mask files 
# since some mask files removed we also remove the coresponding training images
match_files(os.path.join(aug_mask_dir, 'Cleaned'), aug_img_dir, os.path.join(aug_img_dir, 'Cleaned'))

# annotate the full training and validation dataset for use in Mask R-CNN
# First get original mask file names
train = in_mask
val = r'C:\Users\ahls_st\Documents\MasterThesis\IKONOS\With_Hedges\ThreeBands\Splits\BGR\Masks\val'    #change

# combine original files with augmented files
files = getFiles(train, ending='.png')
files2=getFiles(val, ending='.png')
files3=getFiles(os.path.join(aug_mask_dir, 'Cleaned'), ending='.png')
files = sorted(files+files2+files3)
len(files)

# Get mask annotations info
mask_anno = []
for i, file in enumerate(files):
    img_id = file.rsplit('\\', 1)[1]
    #Change to your own needs if needed. 
    # Important is only that ID between mask and image match, format of the ID is up to the user.
    i_id = str(re.findall(r'\d+', img_id)).strip('[]').replace("'", "").replace(",","").replace(" ", "_")
    with rasterio.open(file) as src:
        w = src.read(1)
    #if images with hedges havent been seperated yet then uncomment the following line and tab the other two over
    #if np.max(w)==1:
    m_info = coco.create_annotation_info(w, i_id, i, file, (320, 320))
    mask_anno.append(m_info)


# Get original satellite images from folder 
train = in_img
val = r'C:\Users\ahls_st\Documents\MasterThesis\IKONOS\With_Hedges\FourBands\Splits\imgs\val'     #change

# combine original files with augmented files
files = getFiles(train, ending='.png')
files2=getFiles(val, ending='.png')
files3=getFiles(os.path.join(aug_img_dir, 'Cleaned'), ending='.png')
files_full = sorted(files+files2+files3)
len(files_full)

#Get images annotation info
image_infos = []
for file in files_full:
    i_info = coco.create_image_info(file, (320,320))
    image_infos.append(i_info)

# ensure the lengths and indexes are matching properly
len(mask_anno)
len(image_infos)
random_index = randrange(0, len(mask_anno))
print(mask_anno[random_index])
print(image_infos[random_index])

if mask_anno[random_index][0]['filename'] != image_infos[random_index]['file_name']:
    raise ValueError('Mask and image annotations are not properly matched')

#Create the rest of the json sections
categories = [{'supercategory': 'Vegetation',
                   'id': 1,
                   'name': 'Hedge'}
                    ]
info= {'description': 'Hedges2019Dataset',
   'url': 'none',
   'version': '1.0',
   'year': 2019,
   'contributor': 'DLR',
   'date_created': '2019/07/10'}

licenses= [{
    'url': 'none',
    'id': 1,
    'name': 'none'
    }]

COCO = {'info': info,
        'images': image_infos,
        'annotations': mask_anno,
        'categories': categories,
        'licenses': licenses
        }

#create the json file
with open('IKONOS_4_band_geo_aug.json', 'w') as json_file:#change
    json.dump(COCO, json_file)
    
    
    
#Create an index file for DeepLab
files_anno=sorted(files+files3)
annotate(files_anno, 'train_4band_geo_aug.txt')#change
annotate(files2, 'val_4band_geo_aug.txt')#change

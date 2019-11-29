#!/usr/bin/env python3


import re
import datetime
import numpy as np
from itertools import groupby
from skimage import measure
from PIL import Image
from filegrabber import getFiles
import rasterio
import json
import os
from fileMover import annotate

convert = lambda text: int(text) if text.isdigit() else text.lower()
natrual_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]

def resize_binary_mask(array, new_size):
    image = Image.fromarray(array.astype(np.uint8)*255)
    image = image.resize(new_size)
    return np.asarray(image).astype(np.bool_)

def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour

def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
                counts.append(0)
        counts.append(len(list(elements)))

    return rle

def binary_mask_to_polygon(binary_mask, tolerance=0):
    """Converts a binary mask to COCO polygon representation

    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated
            polygonal chain. If tolerance is 0, the original coordinate array is returned.

    """
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation 
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)

    return polygons

def create_image_info(file_name, image_size, 
                      date_captured=datetime.datetime.utcnow().isoformat(' '),
                      license_id=1, coco_url="", flickr_url=""):
    file_n = file_name.rsplit('\\', 1)[1]
    i_id = str(re.findall(r'\d+', file_n)).strip('[]').replace("'", "").replace(",","").replace(" ", "_")
    image_info = {
            "id": i_id,
            "file_name": file_n,
            "width": image_size[0],
            "height": image_size[1],
            #"date_captured": "today",
            #"license": license_id,
            #"coco_url": coco_url,
            #"flickr_url": flickr_url
    }

    return image_info

def get_area_and_bbox(segmentation):

    seg_x = segmentation[::2]
    seg_y = segmentation[1::2]

    bounding_box = np.zeros(4)
    bounding_box[0] = min(seg_x)
    bounding_box[1] = min(seg_y)
    bounding_box[2] = max(seg_x)
    bounding_box[3] = max(seg_y)

    area = np.float64(0.5*np.abs(np.dot(seg_x,np.roll(seg_y,1))-np.dot(seg_y,np.roll(seg_x,1))))

    return area, bounding_box

def create_annotation_info(annotation_mask, image_id, segmentation_id, filename,
                           image_size, tolerance=2, bounding_box=None):
    annotation_info = []
    classes_ = np.unique(annotation_mask)
    file_n = filename.rsplit('\\', 1)[1]
    classes_ = classes_[1:]

    for class_id, class_ in enumerate(classes_):

        binary_mask = annotation_mask == class_
        #binary_mask = np.invert(binary_mask)
        segmentation = binary_mask_to_polygon(binary_mask, tolerance)

        category_info = {'id': class_id, 'is_crowd': 0}

        for segmentation_ in segmentation:

            area, bounding_box = get_area_and_bbox(segmentation_)
            bb = [bounding_box[0], bounding_box[3], bounding_box[2]-bounding_box[0], bounding_box[3]-bounding_box[1]]
            annotation_record = {
                "category_id": int(category_info["id"] + 1),
                "id": segmentation_id,  # Segmentation/Annotation ID
                "image_id": image_id, # Image ID
                "area": area.tolist(),
                "bbox": bb,
                "segmentation": [segmentation_],
                "filename": file_n
            }

            segmentation_id +=1
            annotation_info.append(annotation_record)

    return annotation_info#, seg_n, id#, image_info


def check_hedge_sizes(mask_folder, img_folder):
    
    mask_files = getFiles(mask_folder, ending='.png')
    
    for i, file in enumerate(mask_files):
        img_id = file.rsplit('\\', 1)[1]
        i_id = str(re.findall(r'\d+', img_id)).strip('[]').replace("'", "").replace(",","").replace(" ", "_")
        with rasterio.open(file) as src:
            w = src.read(1)

        if np.max(w)!=1:
            print('no hedge')
            print('DELETING', file)
            os.remove(file)
            file2 = os.path.join(img_folder, img_id)
            print('DELETING', file2)
            
            os.remove(file2)
            continue
            
        m_info = create_annotation_info(w, i_id, i, file, (320, 320))

        areas=[]
        for entry in m_info:
            areas.append(entry['area'])
        if all(area < 20 for area in areas):
            print('Hedges too small')
            print('DELETING', file)
            os.remove(file)
            file2 = os.path.join(img_folder, img_id)
            print('DELETING', file2)
            os.remove(file2)



if __name__ == '__main__':

    #Get hedge mask images
    train = r'C:\Users\ahls_st\Documents\MasterThesis\IKONOS\With_Hedges\ThreeBands\Splits\BGR\Masks\train' #change
    aug = r'C:\Users\ahls_st\Documents\MasterThesis\IKONOS\With_Hedges\ThreeBands\Splits\BGR\Augs\mask\Scale' #change
    val = r'C:\Users\ahls_st\Documents\MasterThesis\IKONOS\With_Hedges\ThreeBands\Splits\BGR\Masks\val' #change
    
    files = getFiles(train, ending='.png') #change **** ONLY IF files are not in png format
    files2=getFiles(val, ending='.png') #change **** ONLY IF files are not in png format
    files3=getFiles(aug, ending='.png') #change **** ONLY IF files are not in png format
    files = sorted(files+files2+files3)
    len(files)
    
    #Get mask annotations info
    mask_anno = []
    for i, file in enumerate(files):
        img_id = file.rsplit('\\', 1)[1]
        i_id = str(re.findall(r'\d+', img_id)).strip('[]').replace("'", "").replace(",","").replace(" ", "_")
        with rasterio.open(file) as src:
            w = src.read(1)
        #if images with hedges havent been seperated yet then uncomment the following line and tab the other two over
        #if np.max(w)==1:
        m_info = create_annotation_info(w, i_id, i, file, (320, 320))
        mask_anno.append(m_info)
    
    
    
    #Get satellite images from folder 
    in_fold = r'C:\Users\ahls_st\Documents\MasterThesis\IKONOS\With_Hedges\ThreeBands\Splits\BGR\Images\train' #change
    in_fold2 = r'C:\Users\ahls_st\Documents\MasterThesis\IKONOS\With_Hedges\ThreeBands\Splits\BGR\Images\val' #change
    aug = r'C:\Users\ahls_st\Documents\MasterThesis\IKONOS\With_Hedges\ThreeBands\Splits\BGR\Augs\img\Rot45' #change
    files = getFiles(in_fold, ending='.png') #change **** ONLY IF files are not in png format
    files2=getFiles(in_fold2, ending='.png') #change **** ONLY IF files are not in png format
    files3=getFiles(aug, ending='.png') #change **** ONLY IF files are not in png format
    files = sorted(files+files2+files3)
    len(files)
    
    #Get images annotation info
    image_infos = []
    for file in files:
        i_info = create_image_info(file, (320,320))
        image_infos.append(i_info)

    # ensure the indexes are matching properly
    len(mask_anno)
    len(image_infos)
    if len(mask_anno) != len(image_infos):
        raise ValueError('Mask and image annotations are not properly matched')
    
    print(mask_anno[15])
    print(image_infos[15])
    if mask_anno[15] != image_infos[15]:
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

    #create the json file for Mask R-CNN
    with open('IKONOS_3Band_Aug_Rot45.json', 'w') as json_file:  #change
        json.dump(COCO, json_file)
    
    #Creates annotation files for Deeplab
    files_anno=sorted(files+files3)
    annotate(files_anno, 'train_4band_geo_aug.txt') #change
    annotate(files2, 'val_4band_geo_aug.txt') #change
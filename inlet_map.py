# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 16:55:30 2019

@author: ahls_st
"""

import os
from filegrabber import getFiles
import rasterio
import numpy as np
from rasterio.merge import merge
from windows import erase

def overlap_mask_rasters(in_folder1, in_folder2, out_folder, pattern = None, erase_thresh = 150):
    '''Adds the values of raster maps together in areas where there is overlap.
    Can be used to display results from two different classifications in one 
    single image raster. Areas of agreement between two masks will be given
    a mask value of 5, while areas where only method 1 (in_folder1) classified
    will be given a value of 2, and areas where only method 2 (in_folder2)
    classified will be given a value of 3. Can then use this layer in ArcMap
    or similar to visualize the results. 
    
    Parameters
    ----------
    in_folder1: str
        Folder where mask images from method one are located. 
    in_folder2: str
        Folder where mask images from method two are located. 
    out_folder: str
        Folder to save the output mask files to.
    pattern: str (optional)
        A string defining a keyword or pattern to identify target mask images
        from the input folders.
    erase_thresh: int
        Value determining the minimum threshold for masks. If a single continuous
        cluster of pixels which make up an object mask in an image are less than
        the threshold, then this part of the mask will be removed. Useful for 
        removing small noisy detections.
    
    '''
    files_mask = sorted(getFiles(in_folder1, ending='.png', pattern = pattern))
    files_deep = sorted(getFiles(in_folder2, ending='.png', pattern = pattern))
    
    for fm, fd in zip (files_mask, files_deep):
        assert fm.rsplit('\\', 1)[1] == fd.rsplit('\\', 1)[1], 'Files dont match'
        print(fm.rsplit('\\', 1)[1])
        with rasterio.open(fm) as src:
            mask = src.read(1)
            mask = erase(mask, erase_thresh)
        with rasterio.open(fd) as src:
            deep = src.read(1)
            deep = erase(deep, erase_thresh)
        # pixels with value of 2 are from mask, with 3 are from deep, with 5 are from both agreeing
        # Will use this to symbolize the images in arcmap.
        mask[mask!=0] = 2
        deep[deep!=0] = 3
        out = mask + deep
        out = np.expand_dims(out, axis = 0)
        
        outfile = os.path.join(out_folder, fm.rsplit('\\', 1)[1])
        with rasterio.open(outfile, 'w', **src.meta) as dst:
            dst.write(out)


def mosaic_rasters(in_folder, out_folder, mosaic_name='mosaic', pattern = None, clean = False):
    '''Mosaics a batch of image tiles into one single continuous image.
    
    Parameters
    ----------
    in_folder: str
        Folder where images to be mosaiced are located.    
    out_folder: str
        Folder where mosaic file should be saved to.
    mosaic_name: str (optional)
        Name to be given to the mosaic files.
    clean: bool
        If true, will delete all individual image tiles from the in_folder.
    
    Note
    ----
    Raster files must have coordinate information for this method to work.
    '''
    
    mosaic = []
    img_tiles = getFiles(in_folder, ending = '.png', pattern = pattern)
    for file in img_tiles:
        src = rasterio.open(file)
        mosaic.append(src)
 
    meta_d = mosaic[0].meta.copy()

    arr, xform = merge(mosaic)
 
    print(arr.shape)
 
    meta_d.update({"driver": "PNG",
                        "height": arr.shape[1],
                        "width": arr.shape[2],
                        "count": 1,
                        "transform": xform,
                        "dtype": 'uint8'})
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    print('Creating {}'.format(out_folder))
    with rasterio.open(os.path.join(out_folder, mosaic_name + '.png'), 'w', **meta_d) as dest:
        dest.write_band(1, np.squeeze(arr.astype(np.uint8)))
    
    if clean == True:
        del src
        del mosaic
        for file in img_tiles:
            os.remove(file)
            os.remove(file+'.aux.xml')
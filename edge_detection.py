# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 15:25:06 2019

@author: ahls_st
"""
import numpy as np
import cv2
from skimage.morphology import reconstruction
import matplotlib.pyplot as plt
import rasterio

def normalize(array):
    """Normalizes numpy arrays into scale 0.0 - 1.0"""
    array_min, array_max = array.min(), array.max()
    return ((array - array_min)/(array_max - array_min))

def canny(img):
    #use panchromatic band as input image
    
    #convert image to 8-bit since remote sensing images are 16bit)
    norm = normalize(img)
    eight_bit = (norm*255).astype(np.uint8)
    
    #apply blur. Not sure if this is necessary
    eight_bit = cv2.GaussianBlur(eight_bit, (3,3), 0)
    
    #get threshold values for canny
    thresh = cv2.threshold(eight_bit, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[0]
    
    #apply canny
    edges = cv2.Canny(eight_bit, thresh/2, thresh)
    return edges

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
    norm = normalize(image)
    eight_bit = (norm*255).astype(np.uint8)
    eight_bit = cv2.GaussianBlur(eight_bit, (3,3), 0)
    v = np.median(eight_bit)
	# apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(np.squeeze(eight_bit), lower, upper)
 
	# return the edged image
    return edged

def granular_open(image):
    #gets rid of nonlinear features (apparently)
    seed = image.copy()
    seed[1:-1, 1:-1] = image.min()
    dilated = reconstruction(seed, image, method='dilation')
    return dilated

def granular_close(image):
    #fills holes
    seed = image.copy()
    seed[1:-1, 1:-1] = image.max()
    dilated = reconstruction(seed, image, method='erosion')
    return dilated

if __name__ == 'main':
    filename=r'C:\Users\ahls_st\Documents\MasterThesis\IKONOS\With_Hedges\po_2639640_0000000\Original_File\Pan_Clipped.tif'
    with rasterio.open(filename) as src:
        w = src.read(1)
    
    edged = auto_canny(w)
    
    meta_d = src.meta.copy()
    meta_d.update(dtype = np.uint8)
    
    with rasterio.open('canny_640_4.tif', 'w', **meta_d) as dest:
        dest.write_band(1, edged)
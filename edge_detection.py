# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 15:25:06 2019

@author: ahls_st
"""
import numpy as np
import cv2
from skimage.morphology import reconstruction
from augmentation import normalize

def canny(img, blur = None):
    '''Performs the Canny edge detection algorithm on a given image. Returns
    the edge image.
    
    Parameters
    ----------
    img: np.array
        Single band image array to perform edge detection on. Should be high 
        resolution (e.g. panchromatic band).
    blur: (optional)
        If set to any value then a blur will be performed the the image
        prior to the canny edge detection. Should not be necessary as the
        algorithm should already blur the image itself.
    
    '''
    
    #convert image to 8-bit since remote sensing images are 16bit)
    norm = normalize(img)
    eight_bit = (norm*255).astype(np.uint8)
    
    #apply blur, some say it helps
    if blur:
        eight_bit = cv2.GaussianBlur(eight_bit, (3,3), 0)
    
    #get threshold values for canny
    thresh = cv2.threshold(eight_bit, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[0]
    
    #apply canny
    edges = cv2.Canny(eight_bit, thresh/2, thresh)
    return edges

def granular_open(image):
    '''Performs morphological reconstruction on a binary image using dilation.
    TODO: fix, not working yet
    '''
    
    seed = image.copy()
    seed[1:-1, 1:-1] = image.min()
    dilated = reconstruction(seed, image, method='dilation')
    return dilated

def granular_close(image):
    '''Performs morphological reconstruction on a binary image using erosion.
    TODO: fix, not working yet'''
    
    seed = image.copy()
    seed[1:-1, 1:-1] = image.max()
    dilated = reconstruction(seed, image, method='erosion')
    return dilated

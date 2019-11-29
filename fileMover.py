# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 14:37:48 2019

@author: ahls_st
"""

import os
import numpy as np
import shutil
from filegrabber import getFiles

def annotate(files, file_path):
    '''Used for Deeplab to create the text index files
    
    Parameters
    ----------
    files: list
        List containing full file path strings to be annotated.
    file_path: str
        Outpath for the .txt file
    
    '''
    
    files = sorted(files)
    temp = open(file_path, 'w')
    for file in files:
        try: 
            i = file.rsplit('/', 1)[1]
        except IndexError:
            i = file.rsplit('\\', 1)[1]
        i = os.path.splitext(i)[0]
        temp.write(str(i) + '\n')
    temp.close()
    
def match_files(dir_to_match_with, dir_to_move_from, dir_to_move_to, copy=False):
    '''Moves (or copies) only files which have the same name as those in another 
    folder. Useful when one has a folder of mask files which have been split into
    training and validation sets, and one wishes to then move the same image
    files so that both mask and images folders have the same matching files.
    
    Parameters
    ----------
    dir_to_match_with: str
        Path of the directory containing the list of file names you wish to
        also have in a seperate directory

    dir_to_move_from: str 
        Path of the directory containing the files you wish to extract.
        
    dir_to_move_to: str
        Path of the directory where files should be moved into.
    '''
    
    files_match = os.listdir(dir_to_match_with)
    files_to_move = getFiles(dir_to_move_from, ending = '.png')

    for file in files_to_move:
        i = file.rsplit('\\', 1)[1]
        if i in files_match:
            if not copy:
                shutil.move(file, dir_to_move_to)
            if copy:
                shutil.copy(file, dir_to_move_to)
    

class TrainValSplit():
    """
    Splits a set of images into training and validation, then moves the files
    into new train and val folders. Has one process for Mask R-CNN and one for
    DeepLabv3+ as both have different data format requirements.
    
    """
    
    def __init__(self, 
                 img_origin, 
                 mask_origin, 
                 img_dest, 
                 mask_dest, 
                 index_path, 
                 split_percent=0.25):
        
        self.im_path = img_origin
        self.mask_path = mask_origin
        self.im_dest= img_dest
        self.mask_dest = mask_dest
        self.index_path = index_path
        self.split_percent = split_percent
        self.files = os.listdir(self.im_path)
        self.val_subset = np.random.choice(self.files, int(len(self.files)*self.split_percent), replace=False)
    
    def mask_rcnn(self):
        
        for file in self.val_subset:
            #move the images to validation folder
            full_im=os.path.join(self.im_path, file)
            shutil.move(full_im, self.im_dest)
            #move masks to validation folder
            full_mask=os.path.join(self.mask_path, file)
            shutil.move(full_mask, self.mask_dest)
    
    def deeplab(self):# 0.25 = 25% sample
        annotate(self.files, os.path.join(self.index_path, 'trainval.txt'))
        annotate(self.val_subset, os.path.join(self.index_path, 'val.txt'))
        left_over = [file for file in self.files if file not in self.val_subset]
        annotate(left_over, os.path.join(self.index_path, 'train.txt'))
    
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 10:34:17 2019

@author: ahls_st
"""
import os
import re


def getFiles(dirName, pattern=None, ending = '.tif'):
    """
    Gets all files within a directory tree which end with the given file ending
    and optionally which have a certain string pattern. Good for getting all 
    files stored across different sub-folders.
    
    Parameters
    ----------
    pattern: string
        A string determining a sequence of letters which can be used to identify
        the desired image files. Useful if a folder contains multiple .tif files
        and only one is desired.
    """
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = []
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getFiles(fullPath)
        else:
            allFiles.append(fullPath)
    tifs=[file for file in allFiles if file.endswith(ending)]
    if pattern is None:
        return tifs
    else:
        rgb = [img for img in tifs if re.search(pattern, img)]
        return rgb
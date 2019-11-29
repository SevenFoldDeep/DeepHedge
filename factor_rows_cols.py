# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 23:29:24 2019

@author: LENOVO
"""
import rasterio


def allfactors(n):
    return set(
        factor for i in range(1, int(n**0.5) + 1) if n % i == 0
        for factor in (i, n//i)
    )
    
def factchecker(nums):
    for x in nums:
        if x>225 and x<280:
            i = x
            return i
        else:
            i=0
    return i

def getFactor(image_path, ind=1):
    '''
    Function to help set up windows. 
    
    Parameters
    ----------
    image_path: str
        Path to the image file.
    ind: int
        Determines which shape index of the image will be used.
        
    Returns
    -------
    offset: int
        The increase in the window offset in each iteration.
    iterations: int
        Gives the number of loops that can be done with the check offset.
    '''
    with rasterio.open(image_path) as src:
        image=src.read()
    col = image.shape[ind]
    
    #trim the edges as Qgis cropping tends to leave a border of zeros at the edge.
    col=col-2
    
    #make sure the number of rows or cols are even
    if col % 2 == 1:
        col=col-1
    print(col)
    offset=0
    
    #search for an offset value that ranges between 225 and 280
    while offset not in range(225, 280):
        nums=allfactors(col)
        
        offset = factchecker(nums)   
        if offset>225 and offset<280:
            print(offset)
        else:
            col=col-2
    iterations = col/offset
    return offset, int(iterations)
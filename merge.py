# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 12:19:45 2019

@author: ahls_st
"""
from filegrabber import getFiles
import rasterio
from rasterio.merge import merge
import os
import numpy as np

mosaic = []
files = getFiles(r'H:\mosaics', ending='.png')

print(files[0])
for file in files:
    mos = rasterio.open(file)
    mosaic.append(mos)
print(len(mosaic))



#mosaic the mask tiles into one image

meta_d = mosaic[0].meta.copy()



arr, xform = merge(mosaic)
print(arr.shape)
meta_d.update({"driver": "PNG",
                       "height": arr.shape[1],
                       "width": arr.shape[2],
                       "count": 1,
                       "transform": xform,
                       "dtype": 'uint8'})

os.mkdir(os.path.join(r'H:\mosaics', 'deeplab_mosaic'))
with rasterio.open(os.path.join(r'H:\mosaics', 'deeplab_mosaic', 'mosaic_deeplab.png'), 'w', **meta_d) as dest:
    dest.write_band(1, np.squeeze(arr.astype(np.uint8)))
    
    
#delete the individual masks and their xml files after they have been mosaiced
#import os
#for file in files:
#    os.remove(file)
#    os.remove(file+'.aux.xml')
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 13:38:10 2019

@author: ahls_st
"""

import rasterio
from rasterio.windows import Window
import os

def mask_split(file, outpath):
    """Test change."""
    c_off = -97
    for j in range(35):
        c_off+=237
        r_off=0
        for i in range(27):
            a = str(i)+"_"+str(j)
            outfile = os.path.join(outpath, 'Split_%s' % a + '.tif') 
            win = Window(c_off,r_off, 320, 320)
            with rasterio.open(file) as src:
                w = src.read(1, window=win)
            xform = rasterio.windows.transform(win, src.meta['transform'])
            meta_d=src.meta.copy()
            meta_d.update({"driver": "GTiff",
                               "height": 320,
                               "width": 320,
                               "transform": xform})
            with rasterio.open(outfile, "w", **meta_d) as dest:
                    dest.write_band(1, w)
            r_off+=249
            print('Processing %s' % a)



def bands_split(file, outpath):
    c_off = -97
    for j in range(35):
        c_off+=237
        r_off=0
        for i in range(27):
            a = str(i)+"_"+str(j)
            outfile = os.path.join(outpath, 'Split_%s' % a + '.tif') 
            win = Window(c_off,r_off, 320, 320)
            with rasterio.open(file) as src:
                w = src.read(window=win)
            xform = rasterio.windows.transform(win, src.meta['transform'])
            meta_d=src.meta.copy()
            meta_d.update({"driver": "GTiff",
                               "height": 320,
                               "width": 320,
                               "transform": xform})
            with rasterio.open(outfile, "w", **meta_d) as dest:
                    dest.write(w)
            r_off+=249
            print('Processing %s' % a)


if __name__ == '__main__':
    out_sh = r'C:\Users\ahls_st\Documents\MasterThesis\ShapeFile_StudyArea\Hedges_Mask_Splits'
    in_sh = r'C:\Users\ahls_st\Documents\MasterThesis\ShapeFile_StudyArea\Hedge_Mask\hedge_mask_planet.tif'
    in_r = r'C:\Users\ahls_st\Documents\MasterThesis\PlanetData\Planet_Data\Mosaic\Clip_Mosaic.tif'
    out_r = r'C:\Users\ahls_st\Documents\MasterThesis\PlanetData\Split_Data'
    bands_split(in_r, out_r)
    mask_split(in_sh, out_sh)
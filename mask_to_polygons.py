# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 10:59:41 2019

@author: ahls_st
"""
import rasterio
from rasterio.features import shapes
import fiona
from filegrabber import getFiles
import os

def shapefile_from_raster(raster_file, outfile):
    '''Transforms binary raster masks into polygon shapefiles.
    
    Parameters
    ----------
    raster_file: str
        Path to binary raster file.
    outfile: str
        Full path and file name (with extension) of the desired output file.
    '''
    with rasterio.open(raster_file) as src:
        image = src.read(1)
        super_threshold_indices = image > 1
        image[super_threshold_indices] = 1
        mask = image == 1
    
    results = ({'properties': {'raster_val': v}, 'geometry': s} for i, (s, v) in 
               enumerate(shapes(image, mask=mask, transform=src.meta['transform'])))
    
    with fiona.open(
            outfile, 'w', 
            driver="ESRI Shapefile",
            crs=src.crs,
            schema={'properties': [('raster_val', 'int')],
                    'geometry': 'Polygon'}) as dst:
        dst.writerecords(results)


if __name__ == '__main__':
    directory = r'H:\predictions_fully_trained_geo'
    if not os.path.exists(os.path.join(directory, 'polygons')):
        os.mkdir(os.path.join(directory, 'polygons'))
        print('Creating {}'.format(os.path.join(directory, 'polygons')))
    
    files = getFiles(directory, ending = '.png')

    for file in files:
        outpath = os.path.join(directory, 'polygons')
        i = file.rsplit('\\', 1)[1]
        i = os.path.splitext(i)[0]
        outfile = os.path.join(outpath, i + '_polygon.shp')
        shapefile_from_raster(file, outfile)
        print(outfile)
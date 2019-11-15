# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 10:59:41 2019

@author: ahls_st
"""
import rasterio
from rasterio.features import shapes
import fiona

def shapefile_from_raster(raster_file, outfile, ):
    with rasterio.open(raster_file) as src:
        image = src.read(1)
        super_threshold_indices = image > 1
        image[super_threshold_indices] = 1
        mask = image == 1
    
    results = ({'properties': {'raster_val': v}, 'geometry': s} for i, (s, v) in enumerate(shapes(image, mask=mask, transform=src.meta['transform'])))
    
    with fiona.open(
            outfile, 'w', 
            driver="ESRI Shapefile",
            crs=src.crs,
            schema={'properties': [('raster_val', 'int')],
                    'geometry': 'Polygon'}) as dst:
        dst.writerecords(results)


if __name__ == '__main__':
    raster = r'H:\land_hedges\mosaic.png'
    outfile = 'landscape_polygons.shp'
    
    shapefile_from_raster(raster, outfile)
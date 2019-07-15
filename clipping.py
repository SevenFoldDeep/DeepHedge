# -*- coding: utf-8 -*-

import os
import rasterio
from rasterio import mask
from rasterio.merge import merge
import fiona
from matplotlib import pyplot
import numpy as np
import gdal
from xml.dom import minidom
import re
import fileGrabber

np.seterr(divide='ignore', invalid='ignore')

#give a directory, the class should find all tif rasters and clip them to a mask
class ManageRasters():
    """Reads and clips all .tif files within a directory
    
    Parameters
    ----------
    directory:
        Path to directory of target rasters
    mask:
        File path of mask to be applied to rasters
        
    Attributes
    ----------
    dir:
        Path of the directory given
    mask:
        Mask to use for clipping
    raster_paths:
        Paths of all .tif files in the directory
    rasters:
        Loaded raster files
    meta:
        Metadata for each .tif file
        
    """
    def __init__(self, directory, mask=None, pattern=None):
        self.dir=directory
        self.raster_paths=fileGrabber.getFiles(directory, pattern)
        self.rasters=[rasterio.open(_raster) for _raster in self.raster_paths]
        self.meta=[rast.meta.copy() for rast in self.rasters]
        if mask is not None:
            self.mask=fiona.open(mask, "r")
        
    def plot_raster(self, bands=1, i=0):
        """Plots the raster images which have been found within the given
        directory.
        
        Parameters
        ----------
        bands: 
            Optional. Image band or bands to display, default is 1.
        i: 
            Index of raster to plot, default is 0.
            
        """
        pyplot.imshow(self.rasters[i].read(bands), cmap="terrain")
        pyplot.show()
        
        
    def clip(self, outpath=None):
        """Applies a mask to the rasters
        
        Parameters
        ----------
        outpath:
            Optional. If specified then clipped files will be saved in outpath
            folder. If not given then clipped files will be stored in the
            same folder as the original (unclipped) rasters.
        
        
        """
        features = [feature["geometry"] for feature in self.mask]
        if outpath is None:
            for file in self.rasters:
                name=file.name.rsplit('\\', 1)[1]
                outfile = os.path.join(self.dir, 'Clip_%s' % name)  
                out_image, out_transform = rasterio.mask.mask(file, features, crop=True)
                meta_d=file.meta.copy()
                meta_d.update({"driver": "GTiff",
                               "height": out_image.shape[1],
                               "width": out_image.shape[2],
                               "transform": out_transform})
                with rasterio.open(outfile, "w", **meta_d) as dest:
                    dest.write(out_image)
        else:
            for file in self.rasters:
                name=file.name.rsplit('\\', 1)[1]
                outfile = os.path.join(outpath, 'Clip_%s' % name) 
                out_image, out_transform = rasterio.mask.mask(file, features, crop=True)
                meta_d=file.meta.copy()
                meta_d.update({"driver": "GTiff",
                               "height": out_image.shape[1],
                               "width": out_image.shape[2],
                               "transform": out_transform})
                with rasterio.open(outfile, "w", **meta_d) as dest:
                    dest.write(out_image)
    
    def parse_xml_meta(self):
        """Reads in the xml metadata file.
        """
        rasters=os.listdir(self.dir)
        xml=[file for file in rasters if file.endswith(".xml")]
        xmldoc = minidom.parse(os.path.join(self.dir, xml[0]))
        self.xml = xmldoc
    
    
    
    def get_band_meta(self):
        '''Reads in the band coefficients from metadata file.'''
        if hasattr(self, 'xml'):
            nodes = self.xml.getElementsByTagName("ps:bandSpecificMetadata")
            coeffs = {}
            for node in nodes:
                bn = node.getElementsByTagName("ps:bandNumber")[0].firstChild.data
                if bn in ['1', '2', '3', '4']:
                    i = int(bn)
                    value = node.getElementsByTagName("ps:reflectanceCoefficient")[0].firstChild.data
                    coeffs[i] = float(value)
        else:
            self.parse_xml_meta()
            nodes = self.xml.getElementsByTagName("ps:bandSpecificMetadata")
            coeffs = {}
            for node in nodes:
                bn = node.getElementsByTagName("ps:bandNumber")[0].firstChild.data
                if bn in ['1', '2', '3', '4']:
                    i = int(bn)
                    value = node.getElementsByTagName("ps:reflectanceCoefficient")[0].firstChild.data
                    coeffs[i] = float(value)
        self.band_coeff = coeffs
    
    
    
    def TOA(self, target_file=None):
        """Target file can specify a string ending for the file if you want to skip certain files.
            Ex. We have: file1_SR.tif
                     file1_mask.tif
                We can specify we want only the SR file by making target_file='SR'
        """
        
        if target_file is None:
            lst3=[]
            lst4=[]
            for file in self.rasters:
                b3 = file.read(3)
                b4 = file.read(4)
                b3c = self.band_coeff[3]
                b4c = self.band_coeff[4]
                TOA3 = b3 * b3c
                TOA4 = b4 * b4c
                lst3.append(TOA3)
                lst4.append(TOA4)
            self.TOA3 = lst3
            self.TOA4 = lst4
        else:
            for file in self.rasters:
                if target_file == re.split("[_.]", file.name.rsplit('\\', 1)[1])[-2]:
                    b3 = file.read(3)
                    b4 = file.read(4)
                    b3c = self.band_coeff[3]
                    b4c = self.band_coeff[4]
                    TOA3 = b3 * b3c
                    TOA4 = b4 * b4c
                    self.TOA3 = TOA3
                    self.TOA4 = TOA4
                    self.filename= target_file+'.tif'
                else:
                    print('Skipping file {}'.format(re.split("[_.]", file.name.rsplit('\\', 1)[1])[-2]))
    
    
    
    def calc_ndvi(self, outpath=None, inp=None):
        """Calculate NDVI.
        
        Parameters
        ----------
        outpath: str
            Specifies the folder to write the files to. If left as none the 
            files will be written into the directory specified by self.dir.
        inp: int
            Determines if we want to use the internally stored b3 and b4 values
            that come from self.TOA(). If inp is any value it will take the self.TOA()
            values.
        """
        if inp is None:
            
            if outpath is None:
                for file in self.rasters:
                    name=file.name.rsplit('\\', 1)[1]
                    b3 = file.read(3)
                    b4 = file.read(4)
                    ndvi = np.zeros(b3.shape, dtype=rasterio.float32)
                    ndvi_upper = b4.astype(rasterio.float32) - b3.astype(rasterio.float32)
                    ndvi_lower = b4.astype(rasterio.float32) + b3.astype(rasterio.float32)
                    ndvi = ndvi_upper / ndvi_lower
                    
                    #ndvi = ((b4.astype(float) - b3.astype(float)) / (b4 + b3))
                    outfile = os.path.join(self.dir, 'NDVI_%s' % name)
                    meta_d=file.meta
                    meta_d.update(dtype=rasterio.float32,
                                  count=1,
                                  compress='lzw')
                    print("Processing %s" % (name))
                    with rasterio.open(outfile, "w", **meta_d) as dest:
                        dest.write_band(1, ndvi.astype(rasterio.float32))
                    
            else:
                for file in self.rasters:
                    name=file.name.rsplit('\\', 1)[1]
                    b3 = file.read(3)
                    b4 = file.read(4)
                    ndvi = np.zeros(b3.shape, dtype=rasterio.float32)
                    ndvi_upper = b4.astype(float) - b3.astype(float)
                    ndvi_lower = b4.astype(float) + b3.astype(float)
                    ndvi = ndvi_upper / ndvi_lower
                    
                    #ndvi = ((b4.astype(float) - b3.astype(float)) / (b4 + b3))
                    outfile = os.path.join(outpath, 'NDVI_%s' % name)
                    meta_d=file.meta
                    meta_d.update(dtype=rasterio.float32,
                                  count=1,
                                  compress='lzw')
                    print("Processing %s" % (name))
                    with rasterio.open(outfile, "w", **meta_d) as dest:
                        dest.write_band(1, ndvi.astype(rasterio.float32))
                        
        else:
            ndvi = np.zeros(self.TOA3.shape, dtype=rasterio.float32)
            ndvi_upper = self.TOA4.astype(rasterio.float32) - self.TOA3.astype(rasterio.float32)
            ndvi_lower = self.TOA4.astype(rasterio.float32) + self.TOA3.astype(rasterio.float32)
            ndvi = ndvi_upper / ndvi_lower
                    
            #ndvi = ((b4.astype(float) - b3.astype(float)) / (b4 + b3))
            outfile = os.path.join(self.dir, 'NDVI_%s' % self.filename)
            meta_d=self.rasters[0].meta
            meta_d.update(dtype=rasterio.float32,
                          count=1,
                          compress='lzw')
            with rasterio.open(outfile, "w", **meta_d) as dest:
                dest.write_band(1, ndvi.astype(rasterio.float32))
            
    def threshold_ndvi(self, min_value=-1, max_value=1):
        '''Will threshold values from an NDVI image to fit the -1 to 1 range.'''
        
        for file in self.rasters:
            name=file.name.rsplit('\\', 1)[1]
            ndvi = file.read(1)
            ndvi[ndvi > 1] = 1
            ndvi[ndvi < -1] = -1
            outfile = os.path.join(self.dir, 'Threshold_%s' % name)
            meta_d=file.meta
            with rasterio.open(outfile, "w", **meta_d) as dest:
                dest.write_band(1, ndvi.astype(rasterio.float32))
            
    def mosaic(self, outpath=None, data='IKONOS'):
        """Merges the files together. 
        
        Parameters
        ----------
        data: str
            A string to be used in the filename to help identify which dataset
            has been mosaiced together.
        
        """
        
        if outpath is None:
            outfile = os.path.join(self.dir, 'Mosaic_%s' % data, '.tif')
            mosaic, out_trans = merge(self.rasters)
            meta_d = self.meta[0].meta.copy()
            meta_d.update({
                    "driver": "GTiff",
                    "height": mosaic.shape[1],
                    "width": mosaic.shape[2],
                    "transform": out_trans})
            with rasterio.open(outfile, "w", **meta_d) as dest:
                dest.write(mosaic)
                
        else:
            outfile = os.path.join(outpath, 'Mosaic_%s' % data, '.tif')
            mosaic, out_trans = merge(self.rasters)
            meta_d = self.meta[0].meta.copy()
            meta_d.update({
                    "driver": "GTiff",
                    "height": mosaic.shape[1],
                    "width": mosaic.shape[2],
                    "transform": out_trans})
            with rasterio.open(outfile, "w", **meta_d) as dest:
                dest.write(mosaic)
    

#directory=r'C:\Users\ahls_st\Documents\MasterThesis\PlanetData\Mosaic_NDVI'       
#directoryy=r'C:\Users\ahls_st\Documents\MasterThesis\PlanetData\planet_order_348405\20190422_093618_1002'  
#maskk=r'C:\Users\ahls_st\Documents\MasterThesis\PlanetData\Mosaic_NDVI\Mask'     
#out=r'C:\Users\ahls_st\Documents\MasterThesis\ShapeFile_StudyArea'

#file=gdal.Open(r'C:\Users\ahls_st\Documents\MasterThesis\ShapeFile_StudyArea\NDVI_SR_Mosiac_2019.tif').ReadAsArray().astype(float)
#ind = np.where((file == file.min()))

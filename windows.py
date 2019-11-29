# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 13:38:10 2019

@author: ahls_st
"""

import rasterio
from rasterio.windows import Window
import os
from scipy.ndimage import label
import numpy as np
from filegrabber import getFiles
import cv2
import shutil
from fileMover import match_files



def getFactor(image, ind=0):
    '''
    Function to help set up sliding windows to create smaller image tiles from
    large satellite images. Allows for a certain amount of overlap between 
    neighbouring images as this is standard practice in the literature.
    
    Parameters
    ----------
    image: np.array(shape = (n, m))
        A numpy arrary representing a single bands image where n is number of 
        rows and m is the number of columns.
    ind: int
        Determines which shape index of the image the factor will be calculated
        for. 1 for rows, 2 for columns.
        
    Returns
    -------
    offset: int
        The increase in the window offset in each iteration.
    iterations: int
        Gives the number of loops that can be done with the check offset.
        
    '''
    col = image.shape[ind]
    
    #trim the edges as Qgis cropping tends to leave a border of zeros at the edge.
    col=col-2
    
    #make sure the number of rows or cols are even
    if col % 2 == 1:
        col=col-1
    
    offset=0
    
    if ind == 0:
        print('number of rows: ', col)
    elif ind == 1:
        print('number of cols: ', col)
    
    def allfactors(n):
        '''Gets all factors of the number of rows or columns of the image'''
        return set(
            factor for i in range(1, int(n**0.5) + 1) if n % i == 0
            for factor in (i, n//i)
        )
    
    #function used to check if the amount of image overlap is within desired 
    #range
    def factchecker(nums):
        for x in nums:
            if x>225 and x<280:
                i = x
                return i
            else:
                i=0
        return i
    
    #search for an offset value that ranges between 225 and 280
    while offset not in range(225, 280):
        nums=allfactors(col)
        
        offset = factchecker(nums)   
        if offset>225 and offset<280:
            print('Factor:', offset)
        else:
            col=col-2
    iterations = col/offset
    return offset, int(iterations)



class ImageTilerPlanet():
    '''Take a georeferenced image and splits it into overlapping window tiles,
    retaining their original geolocations.
    Different from the IKONOS Tiler in that the naming convention for the outfile
    (Planet doesn't use a tile ID while IKONOS splitting does in my case).
    
    Parameters
    ----------
    in_file: str
        Path to the file which should be split.
    out_dir: str
        Folder path to store output files.
        
    Note
    ----
    Does not work with small images where only a few windows fit into the image
    
    TODO: make the iteration depend on a while statement. Get the total number 
    of columns or rows from the function above. Then use this to keep track of 
    how much space is left that the window can slide by minusing the offset value
    from the total number after each pass. Then make the sliding continue while 
    the number of rows or columns left are still large than the offset value.
    Once the offset is larger than the remaining r or c then we break the looping
    of this dimension. 
    
    '''
    def __init__(self, in_file, out_dir):
        self.file = in_file
        with rasterio.open(self.file) as src:
            self.opened = src.read(1)
        self.outpath = out_dir
    
    def mask_split(self,  height=320, width=320):
        '''Splits a one band input image.
        
        Parameters
        ----------
        height: int
            height of the desired output window tiles
        width: int
            width of the desired output window tiles
            
        Note
        ----
        Adjust the c_off and r_off manually. They represent the index of the 
        top left corner of the window from which point the width and height
        are then applied.
        To have overlap, move these offsets an amount that is smaller than the
        width/height.
        
        '''
        #get window sliding parameters for row and column, as well as the 
        #number of iterations for sliding the rows and columns
        r_plus, ri = getFactor(self.opened, 0)
        c_plus, ci = getFactor(self.opened, 1)
        
        #set a negative starting point for the window as it will get shifted to
        #start at column 1 once the loop starts.
        c_off = -c_plus+1
        
        for c in range(ci):
            
            c_off += c_plus
            r_off = 1
            for r in range(ri):
                
                a = str(r)+"_"+str(c)
                outfile = os.path.join(self.outpath, 'Split_{}'.format(a) + '.png') 
                win = Window(c_off,r_off, width, height)
                #open file
                with rasterio.open(self.file) as src:
                    w = src.read(1, window=win)
                #write meta data
                xform = rasterio.windows.transform(win, src.meta['transform'])
                meta_d=src.meta.copy()
                meta_d.update({"driver": "PNG",
                               "height": height,
                                   "width": width,
                                   "transform": xform})
                #write output
                with rasterio.open(outfile, "w", **meta_d) as dest:
                        dest.write_band(1, w)
                
                r_off += r_plus
                
                print('Processing tile number {}'.format(a))

    def bands_split(self, height=320, width=320):
        '''Splits a multi-band input image. 
        
        Parameters
        ----------
        height: int
            height of the desired output window tiles.
        width: int
            width of the desired output window tiles.
            
        '''
        #get window sliding parameters for row and column, as well as the 
        #number of iterations for sliding the rows and columns
        r_plus, ri = getFactor(self.opened, 0)
        c_plus, ci = getFactor(self.opened, 1)
        
        #set a negative starting point for the window as it will get shifted to
        #start at column 1 once the loop starts.
        c_off = -c_plus+1
        
        for c in range(ci):
            c_off += c_plus
            r_off = 1
            for r in range(ri):
                a = str(r)+"_"+str(c)
                outfile = os.path.join(self.outpath, 'Split_{}'.format(a) + '.png')
                win = Window(c_off,r_off, width, height)
                with rasterio.open(self.file) as src:
                    w = src.read(window=win)
                xform = rasterio.windows.transform(win, src.meta['transform'])
                meta_d=src.meta.copy()
                meta_d.update({"driver": "PNG",
                               "height": height,
                                   "width": width,
                                   "transform": xform})
                with rasterio.open(outfile, "w", **meta_d) as dest:
                        dest.write(w)
                r_off += r_plus
                print('Processing tile number {}'.format(a))
                

class ImageTilerIKONOS():
    '''Take a georeferenced image and splits it into overlapping window tiles,
    retaining their original geolocations.
    Different from the PlanetTiler in that the naming convention for the outfile
    (Planet doesn't use a tile ID while IKONOS splitting does in my case).
    
    Parameters
    ----------
    in_file: str
        Path to the file which should be split.
    out_dir: str
        Folder path to store output files.
    
    '''
    def __init__(self, in_file, out_dir):
        self.file = in_file
        with rasterio.open(self.file) as src:
            self.opened = src.read(1)
        self.outpath = out_dir
    
    def mask_split(self,  height=320, width=320, tile_id='533'):
        '''Splits a one band input image.
        
        Parameters
        ----------
        height: int
            height of the desired output window tiles
        width: int
            width of the desired output window tiles
            
        Note
        ----
        Adjust the c_off and r_off manually. They represent the index of the 
        top left corner of the window from which point the width and height
        are then applied.
        To have overlap, move these offsets an amount that is smaller than the
        width/height.
        
        '''
        #get window sliding parameters for row and column, as well as the 
        #number of iterations for sliding the rows and columns
        r_plus, ri = getFactor(self.opened, 0)
        c_plus, ci = getFactor(self.opened, 1)
        
        #set a negative starting point for the window as it will get shifted to
        #start at column 1 once the loop starts.
        c_off = -c_plus+1
        
        for c in range(ci):
            c_off += c_plus
            r_off=1
            for r in range(ri):
                a = str(r)+"_"+str(c)
                outfile = os.path.join(self.outpath, 'Split_{}_{}'.format(a, tile_id) + '.png') 
                win = Window(c_off,r_off, width, height)
                with rasterio.open(self.file) as src:
                    w = src.read(1, window=win)
                xform = rasterio.windows.transform(win, src.meta['transform'])
                meta_d=src.meta.copy()
                meta_d.update({"driver": "PNG",
                               "height": height,
                                   "width": width,
                                   "transform": xform})
                with rasterio.open(outfile, "w", **meta_d) as dest:
                        dest.write_band(1, w)
                r_off += r_plus
                print('Processing tile {} number {}'.format(tile_id, a))

    def bands_split(self, height=320, width=320, tile_id='533'):
        '''Splits a multi-band input image. c_off and r_off represent the 
        
        Parameters
        ----------
        height: int
            height of the desired output window tiles.
        width: int
            width of the desired output window tiles.
            
        '''
        #get window sliding parameters for row and column, as well as the 
        #number of iterations for sliding the rows and columns
        r_plus, ri = getFactor(self.opened, 0)
        c_plus, ci = getFactor(self.opened, 1)
        
        #set a negative starting point for the window as it will get shifted to
        #start at column 1 once the loop starts.
        c_off = -c_plus+1
        
        for c in range(ci):
            c_off+= c_plus
            r_off= 1
            for r in range(ri):
                a = str(r)+"_"+str(c)
                outfile = os.path.join(self.outpath, 'Split_{}_{}'.format(a, tile_id) + '.png')
                win = Window(c_off,r_off, width, height)
                with rasterio.open(self.file) as src:
                    w = src.read(window=win)
                xform = rasterio.windows.transform(win, src.meta['transform'])
                meta_d=src.meta.copy()
                meta_d.update({"driver": "PNG",
                               "height": height,
                                   "width": width,
                                   "transform": xform})
                with rasterio.open(outfile, "w", **meta_d) as dest:
                        dest.write(w)
                r_off+= r_plus
                print('Processing tile {} number {}'.format(tile_id, a))




def erase(mask, min_pixels):
    """
    Erases small cut-off hedges within windowed image tiles.
    
    Parameters
    ----------
    mask: array(N, N, 1)
        1-D array of equal sized sides.
    
    min_pixels: int
        Threshold value for how many pixels a mask must be in size to be kept.
        
    Returns
    -------
    refined_mask: array(N, N, 1)
        Returns the mask after having removed segments that were too small/only
        contained a small piece of a hedge object.
        
    """
    # make sure it is binary
    mask[mask != 0] = 1
    
    # Now, we perform region labelling. This way, every connected component
    # will have their own colour value.
    labelled_mask, num_labels = label(mask)
    
    # Let us now remove all the regions which are too small.
    refined_mask = mask.copy()
    minimum_cc_sum = min_pixels
    for lab in range(num_labels):
        if np.sum(refined_mask[labelled_mask == lab+1]) < minimum_cc_sum:
            refined_mask[labelled_mask == lab+1] = 0
    return refined_mask
            
def find_hedges(in_path, outpath, move=None, erase_thres=200):
    """
    Finds all mask images where appropriately sized hedges are present. Deletes
    mask files without hedges and moves those with hedges to a new folder.
    
    Parameters
    ----------
    inpath: str
        Path to mask files
    outpath: str
        Path where masks containing hedges should be moved to.
    move: str (optional)
        String giving the folder path of where to move masks without hedges.
        If left as none then masks without hedges will simply be deleted. 
        When first testing different erase thresholds this is not recommended.
        More CAUTIOUS approach is to first move the masks and inspect that
        only unwanted masks have been moved, and then deleting them manually.
    
    """
    #get all mask image tiles
    files = getFiles(in_path, ending = '.png')
    
    for file in files:
        name=file.rsplit('\\', 1)[1]
        #load hedge mask
        mask = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        #erase any hedge mask segments that are too small
        mask = erase(mask, erase_thres)
        #check if any hedges are present within the mask image
        if np.max(mask) < 1:
            print('removing {}'.format(file))
            if move:
                shutil.move(file, move)
            else:
                shutil.remove(file)
        
        else:
            cv2.imwrite(os.path.join(outpath, name), mask)


if __name__ == '__main__':
    # Window the mask images.
    out_sh = r'D:\Steve\val_imgs_coords\all_masks'
    in_sh = r'C:\Users\ahls_st\Documents\MasterThesis\IKONOS\With_Hedges\po_2643534_0000000\PanSharp\Mask_Images\2_Mask_534.png'
    ImageTilerIKONOS(in_sh, out_sh).mask_split(tile_id='534_1')
    
    # Window the raster images.
    #out_r = r'D:\Steve\IKONOS\302_Images\Processed_with_Windows\imgs'
    #in_r = r'D:\Steve\IKONOS\302_Images\Clipped_to_Freyung\Clipped_302_2.tif'
    #ImageTilerIKONOS(in_r, out_r).bands_split(tile_id='302_2')
    
    # Find windows that contain hedges.
    #out = r'D:\Steve\IKONOS\302_Images\Processed_with_Windows\masks\Cleaned'
    #in_ = r'C:\Users\ahls_st\Documents\MasterThesis\IKONOS\With_Hedges\ThreeBands\Splits\BGR\Augs\mask\full'
    #mv = r'D:\Steve\IKONOS\302_Images\Processed_with_Windows\masks\Removed'
    #find_hedges(out_sh, out, mv)
    
    # Moves the images which match to the mask files
    #mv2 = r'D:\Steve\IKONOS\302_Images\Processed_with_Windows\imgs\Cleaned'
    #match_files(out, out_r, mv2)
    
    #ImageTilerPlanet(in_r, out_r).bands_split()
    #ImageTilerPlanet(in_sh, out_sh).mask_split()
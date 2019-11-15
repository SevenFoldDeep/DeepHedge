# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 19:35:04 2019

@author: ahls_st
"""

from imgaug import augmenters as iaa
import numpy as np
import rasterio
import cv2
import warnings
import os
import re
from random import randrange, uniform

def anisodiff(img,niter=1,kappa=50,gamma=0.1,step=(1.,1.),option=1,ploton=False):
    """
    Anisotropic diffusion.

    Usage:
    imgout = anisodiff(im, niter, kappa, gamma, option)

    Arguments:
            img    - input image
            niter  - number of iterations
            kappa  - conduction coefficient 20-100 ?
            gamma  - max value of .25 for stability
            step   - tuple, the distance between adjacent pixels in (y,x)
            option - 1 Perona Malik diffusion equation No 1
                     2 Perona Malik diffusion equation No 2
            ploton - if True, the image will be plotted on every iteration

    Returns:
            imgout   - diffused image.

    kappa controls conduction as a function of gradient.  If kappa is low
    small intensity gradients are able to block conduction and hence diffusion
    across step edges.  A large value reduces the influence of intensity
    gradients on conduction.

    gamma controls speed of diffusion (you usually want it at a maximum of
    0.25)

    step is used to scale the gradients in case the spacing between adjacent
    pixels differs in the x and y axes

    Diffusion equation 1 favours high contrast edges over low contrast ones.
    Diffusion equation 2 favours wide regions over smaller ones.

    Reference: 
    P. Perona and J. Malik. 
    Scale-space and edge detection using ansotropic diffusion.
    IEEE Transactions on Pattern Analysis and Machine Intelligence, 
    12(7):629-639, July 1990.

    Original MATLAB code by Peter Kovesi  
    School of Computer Science & Software Engineering
    The University of Western Australia
    pk @ csse uwa edu au
    <http://www.csse.uwa.edu.au>

    Translated to Python and optimised by Alistair Muldal
    Department of Pharmacology
    University of Oxford
    <alistair.muldal@pharm.ox.ac.uk>

    June 2000  original version.       
    March 2002 corrected diffusion eqn No 2.
    July 2012 translated to Python
    """

    # ...you could always diffuse each color channel independently if you
    # really want
    if img.ndim == 3:
        warnings.warn("Only grayscale images allowed, converting to 2D matrix")
        img = img.mean(2)

    # initialize output array
    img = img.astype('float32')
    imgout = img.copy()

    # initialize some internal variables
    deltaS = np.zeros_like(imgout)
    deltaE = deltaS.copy()
    NS = deltaS.copy()
    EW = deltaS.copy()
    gS = np.ones_like(imgout)
    gE = gS.copy()

    # create the plot figure, if requested
    if ploton:
        import pylab as pl
        from time import sleep

        fig = pl.figure(figsize=(20,5.5),num="Anisotropic diffusion")
        ax1,ax2 = fig.add_subplot(1,2,1),fig.add_subplot(1,2,2)

        ax1.imshow(img,interpolation='nearest')
        ih = ax2.imshow(imgout,interpolation='nearest',animated=True)
        ax1.set_title("Original image")
        ax2.set_title("Iteration 0")

        fig.canvas.draw()

    for ii in range(niter):

        # calculate the diffs
        deltaS[:-1,: ] = np.diff(imgout,axis=0)
        deltaE[: ,:-1] = np.diff(imgout,axis=1)

        # conduction gradients (only need to compute one per dim!)
        if option == 1:
            gS = np.exp(-(deltaS/kappa)**2.)/step[0]
            gE = np.exp(-(deltaE/kappa)**2.)/step[1]
        elif option == 2:
            gS = 1./(1.+(deltaS/kappa)**2.)/step[0]
            gE = 1./(1.+(deltaE/kappa)**2.)/step[1]

        # update matrices
        E = gE*deltaE
        S = gS*deltaS

        # subtract a copy that has been shifted 'North/West' by one
        # pixel. don't as questions. just do it. trust me.
        NS[:] = S
        EW[:] = E
        NS[1:,:] -= S[:-1,:]
        EW[:,1:] -= E[:,:-1]

        # update the image
        imgout += gamma*(NS+EW)

        if ploton:
            iterstring = "Iteration %i" %(ii+1)
            ih.set_data(imgout)
            ax2.set_title(iterstring)
            fig.canvas.draw()
            # sleep(0.01)

    return imgout


def normalize(array):
    """Normalizes numpy arrays into scale 0.0 - 1.0 """
    array_min, array_max = array.min(), array.max()
    return ((array - array_min)/(array_max - array_min))

def invert(array):
    return np.iinfo(array.dtype).max - array

def invert_half(array):
    return (np.iinfo(array.dtype).max/2) - array

class Augmenter():
    '''Augments data within a specified image and mask directory.
    
    Parameters
    ----------
    img_dir: str
        Path to the input image directory.
    mask_dir: str
        Path to input directory containing image mask files
    img_format: str
        String stating the file extention of the target images.
    augmentations: list of augmenters
        Total list of augmentations that can be applied to the image and mask.
    
    '''
    def __init__(self, img_dir, mask_dir, img_format):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_type = img_format
        self.augment_geometric = [iaa.OneOf([iaa.Affine(rotate=40)]),   #0
                                  iaa.OneOf([iaa.Affine(rotate=80)]),
                                  iaa.OneOf([iaa.Affine(rotate=120)]),
                                  iaa.OneOf([iaa.Affine(rotate=160)]),
                                  iaa.OneOf([iaa.Affine(rotate=200)]),
                                  iaa.OneOf([iaa.Affine(rotate=240)]),   #5
                                  iaa.OneOf([iaa.Affine(rotate=280)]),
                                  iaa.OneOf([iaa.Affine(rotate=320)]),
                                  iaa.OneOf([iaa.Affine(scale={"x": (0.8), "y": (0.8)})]),
                                  iaa.OneOf([iaa.Affine(scale={"x": (0.7), "y": (0.7)})]),
                                  iaa.OneOf([iaa.Affine(scale={"x": (1.2), "y": (1.2)})]),   #10
                                  iaa.OneOf([iaa.Affine(scale={"x": (1.3), "y": (1.3)})]),
                                  iaa.OneOf([iaa.Flipud(1)]),
                                  iaa.OneOf([iaa.Fliplr(1)])]     #13
        
        self.augment_spectral = [iaa.OneOf([iaa.GaussianBlur(sigma=(0.75))]),   #0
                                 iaa.OneOf([iaa.AddElementwise((-60, -10))]),
                                 iaa.OneOf([iaa.AddElementwise((10, 60))]),                  
                                 iaa.OneOf([iaa.LogContrast(gain=(0.6, 1.4))]), 
                                 iaa.OneOf([iaa.LogContrast(gain=(0.6, 1.4), per_channel=True)]),
                                 iaa.OneOf([iaa.Add((-60, -40))]),#5
                                 iaa.OneOf([iaa.Add((40, 80))]),
                                 iaa.OneOf([iaa.Add((-40, 0), per_channel=True)]),
                                 iaa.OneOf([iaa.Add((0, 40), per_channel=True)])]     #8
        
        self.augmentations = [
                            iaa.OneOf([iaa.Affine(rotate=40)]),   #0
                            iaa.OneOf([iaa.Affine(rotate=80)]),
                            iaa.OneOf([iaa.Affine(rotate=120)]),
                            iaa.OneOf([iaa.Affine(rotate=160)]),
                            iaa.OneOf([iaa.Affine(rotate=200)]),
                            iaa.OneOf([iaa.Affine(rotate=240)]),   
                            iaa.OneOf([iaa.Affine(rotate=280)]),
                            iaa.OneOf([iaa.Affine(rotate=320)]),
                            iaa.OneOf([iaa.Affine(scale={"x": (0.8), "y": (0.8)})]),
                            iaa.OneOf([iaa.Affine(scale={"x": (1.2), "y": (1.2)})]),
                            iaa.OneOf([iaa.Flipud(1)]),
                            iaa.OneOf([iaa.Fliplr(1)]),
                            iaa.OneOf([iaa.AddElementwise((-60, -10))]),#12
                            iaa.OneOf([iaa.AddElementwise((10, 60))]), 
                            iaa.OneOf([iaa.GaussianBlur(sigma=(0.75))]),
                            iaa.OneOf([iaa.GaussianBlur(sigma=(1.00))]),
                            iaa.OneOf([iaa.LogContrast(gain=(0.6, 1.4))]),
                            iaa.OneOf([iaa.LogContrast(gain=(0.6, 1.4), per_channel=True)]),
                            iaa.OneOf([iaa.Add((-60, -40)), iaa.Add((40, 80))]),
                            iaa.OneOf([iaa.Add((40, 80))]),
                            iaa.OneOf([iaa.Add((-40, 0), per_channel=True)]),
                            iaa.OneOf([iaa.Add((0, 40), per_channel=True)])     #20
                              ]
    
    def getFiles(self, dirName, pattern=None, ending = '.tif'):
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
                allFiles = allFiles + self.getFiles(fullPath)
            else:
                allFiles.append(fullPath)
        images=[file for file in allFiles if file.endswith(ending)]
        if pattern is None:
            return images
        else:
            specific_imgs = [img for img in images if re.search(pattern, img)]
            return specific_imgs
    
    
    def load_img_and_mask(self, img_path, mask_path=False, n_bands=3, shape=320, data_type=np.uint16):
        '''Function used in augment() to load image and mask files. 
        
        '''
        if mask_path:
            #create empty array
            img = np.zeros((shape, shape, n_bands)).astype(data_type)
            #read the bands into the array
            if n_bands == 4:
                with rasterio.open(img_path) as src:
                     '''need to read it in this way, otherwise the output that cv2 writes has the bands in the wrong order, don't ask me why'''
                     img[:,:,0] = src.read(4)
                     img[:,:,1] = src.read(3)
                     img[:,:,2] = src.read(2)
                     img[:,:,3] = src.read(1)
            if n_bands == 3:
                with rasterio.open(img_path) as src:
                     '''need to read it in this way, otherwise the output that cv2 writes has the bands in the wrong order, don't ask me why'''
                     img[:,:,0] = src.read(3)
                     img[:,:,1] = src.read(2)
                     img[:,:,2] = src.read(1)
            #read the mask
            mask = cv2.imread(mask_path)[:,:,0]
            
            return img, mask#, img_meta, mask_meta
        
        else:
            img = np.zeros((shape, shape, n_bands)).astype(data_type)
            if n_bands == 4:
                with rasterio.open(img_path) as src:
                     '''need to read it in this way, otherwise the output that cv2 writes has the bands in the wrong order, don't ask me why'''
                     img[:,:,0] = src.read(4)
                     img[:,:,1] = src.read(3)
                     img[:,:,2] = src.read(2)
                     img[:,:,3] = src.read(1)
            if n_bands == 3:
                with rasterio.open(img_path) as src:
                     '''need to read it in this way, otherwise the output that cv2 writes has the bands in the wrong order, don't ask me why'''
                     img[:,:,0] = src.read(3)
                     img[:,:,1] = src.read(2)
                     img[:,:,2] = src.read(1)
            
            return img
        
        
        
    def augmentation_picker(self, max_num_augs, category):
        '''
        Randomly chooses the number of different augmentations and then randomly 
        chooses which of the available augmentations to use.
        '''
        if category == 'geo':
            augs = len(self.augment_geometric)
            b = randrange(1, max_num_augs)
            out = np.random.choice(augs, b, replace=False)
    
            return out
        
        elif category == 'spectral':
            augs = len(self.augment_spectral)
            b = randrange(0, max_num_augs)
            out = np.random.choice(augs, b, replace=False)
    
            return out
            
        else:
            print('''Category must be either 'geo' or 'spectral'.''')
        
    def augment_combo(self, outpath_img, outpath_mask, n_bands, times=25, n_geo=3, n_spec=4):
        '''Performs image and mask augmentations applying a combination of
        random augmentations to each image. First geometric augmentations are 
        applied, followed by spectral augmentations.
        
        Parameters
        ----------
        outpath_img: str
            Path to the output directory where augmented images should be stored.
        outpath_mask: str
            Path to the output directory where augmented masks should be stored.

        '''
        
        #get the files within each folder
        img_paths = sorted(self.getFiles(self.img_dir, ending=self.img_type))
        mask_paths = sorted(self.getFiles(self.mask_dir, ending=self.img_type))
        
        #start looping through each file and performing augmentations on each
        for im, msk in zip(img_paths, mask_paths):
            img_id = im.rsplit('\\', 1)[1]
            mask_id = msk.rsplit('\\', 1)[1]
            
            if img_id != mask_id:
                print('Error: mask and image do not match')
                break
            
            else:
                #load the image and mask
                _im, _mask = self.load_img_and_mask(im, msk, n_bands)
                
                for num in range(times):
                    
                    augs_geo = self.augmentation_picker(n_geo, 'geo')
                    for i, aug in enumerate(augs_geo):
                        if i == 0:
                            _im_copy = self.augment_geometric[aug](images=[_im])[0]
                            _mask_copy = self.augment_geometric[aug](images=[_mask])[0]
                        else:
                            _im_copy = self.augment_geometric[aug](images=[_im_copy])[0]
                            _mask_copy = self.augment_geometric[aug](images=[_mask_copy])[0]
                    
                    if n_spec != 0:
                        augs_spectral = self.augmentation_picker(n_spec, 'spectral')
                        for j, aug in enumerate(augs_spectral):
                            if j == 0:
                            
                                _im_copy_2 = self.augment_spectral[aug](images=[_im_copy])[0]
                            
                            elif j > 0:
                                _im_copy_2 = self.augment_spectral[aug](images=[_im_copy_2])[0]
    
                    #create file names
                    out_img = os.path.join(outpath_img, 'A{}_{}'.format(num, img_id))
                    out_mask = os.path.join(outpath_mask, 'A{}_{}'.format(num, mask_id))
                   
                    #write files
                    if n_spec != 0:
                        if len(augs_spectral) > 0:
                            cv2.imwrite(out_img, _im_copy_2.astype(np.uint16))
                            print('Image: {} written'.format(img_id))
                            cv2.imwrite(out_mask, _mask_copy.astype(np.uint8))
                            print('Mask: {} written'.format(mask_id))
                            
                        elif len(augs_spectral) == 0:
                            cv2.imwrite(out_img, _im_copy.astype(np.uint16))
                            print('Image: {} written'.format(img_id))
                            cv2.imwrite(out_mask, _mask_copy.astype(np.uint8))
                            print('Mask: {} written'.format(mask_id))
                    else:
                        cv2.imwrite(out_img, _im_copy.astype(np.uint16))
                        print('Image: {} written'.format(img_id))
                        cv2.imwrite(out_mask, _mask_copy.astype(np.uint8))
                        print('Mask: {} written'.format(mask_id))

    def augment(self, outpath_img, outpath_mask, n_bands):
        '''Performs image and mask augmentations applying only a single 
        augmentation to each image.
        
        Parameters
        ----------
        outpath_img: str
            Path to the output directory where augmented images should be stored.
        outpath_mask: str
            Path to the output directory where augmented masks should be stored.

        '''
        #get the files within each folder
        img_paths = sorted(self.getFiles(self.img_dir, ending=self.img_type))
        mask_paths = sorted(self.getFiles(self.mask_dir, ending=self.img_type))
        
        #start looping through each file and performing augmentations on each
        for im, msk in zip(img_paths, mask_paths):
            img_id = im.rsplit('\\', 1)[1]
            mask_id = msk.rsplit('\\', 1)[1]
            
            if img_id != mask_id:
                print('Error: mask and image do not match')
                break
            
            else:
                #load the image and mask
                _im, _mask = self.load_img_and_mask(im, msk, n_bands)
                
                
                
                #apply all the different augmentations to the mask and image
                for num, aug in enumerate(self.augmentations):
                    
                    if num in range(12, 20): #add the index of any augmentations that will screw up the binary mask to this
                        
                        #only apply this to augmentations where values get changed b.c it will totally screw up the mask by
                        #adding/minusing ect. values from the mask, but mask should stay as binary (0,1)
                        
                        img_aug = aug(images=[_im])
                        
                        #create outpath names
                        out_img = os.path.join(outpath_img, 'A{}_{}'.format(num, img_id))
                        out_mask = os.path.join(outpath_mask, 'A{}_{}'.format(num, mask_id))
                        
                        #write files
                        cv2.imwrite(out_img, img_aug[0].astype(np.uint16))
                        print('Image: {} written'.format(img_id))
                        cv2.imwrite(out_mask, _mask.astype(np.uint8))
                        print('Mask: {} written'.format(mask_id))
                    
                    else:
                    
                        img_aug = aug(images=[_im])
                        
                        msk_aug = aug(images=[_mask])
                        #create file names
                        
                        out_img = os.path.join(outpath_img, 'A{}_{}'.format(num, img_id))
                        out_mask = os.path.join(outpath_mask, 'A{}_{}'.format(num, mask_id))
                       
                        #write files
                        cv2.imwrite(out_img, img_aug[0].astype(np.uint16))
                        print('Image: {} written'.format(img_id))
                        cv2.imwrite(out_mask, msk_aug[0].astype(np.uint8))
                        print('Mask: {} written'.format(mask_id))
            
    def aniso_aug(self, outpath_img):
        """Performs anisotropic diffusion on images as a type of augmentation"""
        img_paths = sorted(self.getFiles(self.img_dir, ending=self.img_type))
        for im in img_paths:
            img_id = im.rsplit('\\', 1)[1]
            empty = np.zeros((320, 320, 3)).astype(np.uint16)
            with rasterio.open(im) as src:
                 for i, j in enumerate([3,2,1]):
                     w = src.read(j)
                     empty[:,:,i] = anisodiff(w, 10, 25, option=2)
                     
            out_img = os.path.join(outpath_img, 'A1_{}'.format(img_id))        
            cv2.imwrite(out_img, empty.astype(np.uint16))
            print('Image: {} written'.format(img_id))
            
    def invert_aug(self, outpath_img):
        img_paths = sorted(self.getFiles(self.img_dir, ending=self.img_type))
        for im in img_paths:
            img_id = im.rsplit('\\', 1)[1]
            _im = self.load_img_and_mask(im, None)
            img_aug_0 = invert(_im)
            #img_aug_1 = invert_half(_im)
            out_img_0 = os.path.join(outpath_img, 'A0_{}'.format(img_id))
            #out_img_1 = os.path.join(outpath_img, 'A1_{}'.format(img_id))
            cv2.imwrite(out_img_0, img_aug_0.astype(np.uint16))
            #cv2.imwrite(out_img_1, img_aug_1.astype(np.uint16))
            print('Image: {} written'.format(img_id))
            
    def random_crop(self, outpath_img, outpath_mask, n_bands, num=1):
        img_paths = sorted(self.getFiles(self.img_dir, ending=self.img_type))
        mask_paths = sorted(self.getFiles(self.mask_dir, ending=self.img_type))
        
        #start looping through each file and performing augmentations on each
        for im, msk in zip(img_paths, mask_paths):
            
            for i in range(num):
                random_top = uniform(-0.25, 0.25)
                random_bot = uniform(-0.25, 0.25)
                random_right = uniform(-0.25, 0.25)
                random_left = uniform(-0.25, 0.25)
                
                img_id = im.rsplit('\\', 1)[1]
                mask_id = msk.rsplit('\\', 1)[1]
                
                aug = iaa.OneOf([iaa.CropAndPad(percent=(random_top, 
                                                         random_right, 
                                                         random_bot, 
                                                         random_left), 
                                                )])
                
                
                _im, _mask = self.load_img_and_mask(im, msk, n_bands)
                
                img_aug = aug(images=[_im])
                msk_aug = aug(images=[_mask])
                
                out_img = os.path.join(outpath_img, 'C{}_{}'.format(num, img_id))
                out_mask = os.path.join(outpath_mask, 'C{}_{}'.format(num, mask_id))
               
                #write files
                cv2.imwrite(out_img, img_aug[0].astype(np.uint16))
                print('Image: {} written'.format(img_id))
                cv2.imwrite(out_mask, msk_aug[0].astype(np.uint8))
                print('Mask: {} written'.format(mask_id))
                        
#if __name__ == '__main__':
   # in_img = r'C:\Users\ahls_st\Documents\MasterThesis\IKONOS\With_Hedges\ThreeBands\Splits\BGR\Images\train'
   # out_img_dir = r'D:\Steve\IKONOS\combo_augs\imgs'
   # in_mask = r'C:\Users\ahls_st\Documents\MasterThesis\IKONOS\With_Hedges\ThreeBands\Splits\BGR\Masks\train'
   # out_mask_dir = r'D:\Steve\IKONOS\combo_augs\masks'
   # augmentor = Augmenter(in_img, in_mask, '.png')
   # augmentor.augment_combo(out_img_dir, out_mask_dir, n_bands=3, times=50)
    #augmentor.augment(out_img_dir, out_mask_dir, n_bands=3)
   # augmentor.aniso_aug(out_img_dir)
    #augmentor.invert_aug(out_img_dir)
    #augmentor.crop(out_img_dir, out_mask_dir, 3)
    
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 12:17:47 2019

@author: ahls_st
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 14:38:47 2019

@author: LENOVO
"""

"""
Mask R-CNN
Configurations and data loading code for the synthetic Shapes dataset.
This is a duplicate of the code in the noteobook train_shapes.ipynb for easy
import into other notebooks, such as inspect_model.ipynb.
Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""


import keras.layers as KL
import keras.backend as K
import tensorflow as tf
import keras.models as KM
import os
import sys
from mrcnn import visualize
import numpy as np
from imgaug import augmenters as iaa
import datetime
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
import skimage
import json
import re
import scipy.ndimage
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau, LearningRateScheduler
from mrcnn.model import data_generator, log
import multiprocessing
import matplotlib.pyplot as plt
import rasterio
from skimage.draw import polygon
import cv2
import math
import scipy.ndimage
from mrcnn.model import (resnet_graph, build_rpn_model, norm_boxes_graph, ProposalLayer,parse_image_meta_graph, DetectionTargetLayer,
                         fpn_classifier_graph, build_fpn_mask_graph, rpn_class_loss_graph, rpn_bbox_loss_graph, mrcnn_class_loss_graph,
                         mrcnn_bbox_loss_graph, mrcnn_mask_loss_graph, DetectionLayer)


# Root directory of the project (goes back a maximum of two folder directories from where this file is located)
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library


if __name__ == '__main__':
    import matplotlib
    # Agg backend runs without a display
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

##############################
# Some Functions
##############################
    
def GetFiles(dirName, pattern=None, ending = '.png'):
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
            allFiles = allFiles + GetFiles(fullPath)
        else:
            allFiles.append(fullPath)
    tifs=[file for file in allFiles if file.endswith(ending)]
    if pattern is None:
        return tifs
    else:
        rgb = [img for img in tifs if re.search(pattern, img)]
        return rgb


##########################################
# Mask RCNN Patch (Fixes an error due to version incompatibility)
##########################################

class HedgeCNN(modellib.MaskRCNN):
    '''Patched the training to remove the built in callbacks to avoid conflict 
    with my custom callbacks. Also patches an error due to Keras version 
    incompatibility
    '''
    def train(self, train_dataset, val_dataset, learning_rate, epochs, layers,
              augmentation=None, custom_callbacks=None, no_augmentation_sources=None):
        """Train the model.
        train_dataset, val_dataset: Training and validation Dataset objects.
        learning_rate: The learning rate to train with
        epochs: Number of training epochs. Note that previous training epochs
                are considered to be done alreay, so this actually determines
                the epochs to train in total rather than in this particaular
                call.
        layers: Allows selecting wich layers to train. It can be:
            - A regular expression to match layer names to train
            - One of these predefined values:
              heads: The RPN, classifier and mask heads of the network
              all: All the layers
              3+: Train Resnet stage 3 and up
              4+: Train Resnet stage 4 and up
              5+: Train Resnet stage 5 and up
        augmentation: Optional. An imgaug (https://github.com/aleju/imgaug)
            augmentation. For example, passing imgaug.augmenters.Fliplr(0.5)
            flips images right/left 50% of the time. You can pass complex
            augmentations as well. This augmentation applies 50% of the
            time, and when it does it flips images right/left half the time
            and adds a Gaussian blur with a random sigma in range 0 to 5.
                augmentation = imgaug.augmenters.Sometimes(0.5, [
                    imgaug.augmenters.Fliplr(0.5),
                    imgaug.augmenters.GaussianBlur(sigma=(0.0, 5.0))
                ])
	    custom_callbacks: Optional. Add custom callbacks to be called
	        with the keras fit_generator method. Must be list of type keras.callbacks.
        no_augmentation_sources: Optional. List of sources to exclude for
            augmentation. A source is string that identifies a dataset and is
            defined in the Dataset class.
        """
        assert self.mode == "training", "Create model in training mode."

        # Pre-defined layer regular expressions
        layer_regex = {
            # all layers but the backbone
            "heads": r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # From a specific Resnet stage and up
            "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "5+": r"(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # All layers
            "all": ".*",
        }
        if layers in layer_regex.keys():
            layers = layer_regex[layers]

        # Data generators
        train_generator = data_generator(train_dataset, self.config, shuffle=True,
                                         augmentation=augmentation,
                                         batch_size=self.config.BATCH_SIZE,
                                         no_augmentation_sources=no_augmentation_sources)
        val_generator = data_generator(val_dataset, self.config, shuffle=True,
                                       batch_size=self.config.BATCH_SIZE)

        # Create log_dir if it does not exist
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Callbacks
        callbacks = []

        # Add custom callbacks to the list
        if custom_callbacks:
            callbacks += custom_callbacks

        # Train
        log("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate))
        log("Checkpoint Path: {}".format(self.checkpoint_path))
        self.set_trainable(layers)
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM)

        # Work-around for Windows: Keras fails on Windows when using
        # multiprocessing workers. See discussion here:
        # https://github.com/matterport/Mask_RCNN/issues/13#issuecomment-353124009
        if os.name == 'nt':
            workers = 0
        else:
            workers = multiprocessing.cpu_count()
            
        self.keras_model.fit_generator(
            train_generator,
            initial_epoch=self.epoch,
            epochs=epochs,
            steps_per_epoch=self.config.STEPS_PER_EPOCH,
            callbacks=callbacks,
            validation_data=val_generator,
            validation_steps=self.config.VALIDATION_STEPS,
            max_queue_size=100,
            workers=workers,
            use_multiprocessing=True,
        )
        self.epoch = max(self.epoch, epochs)
        
    def load_weights(self, filepath, by_name=False, exclude=None):
        """Modified version of the corresponding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exclude: list of layer names to exclude
        """
        import h5py
        # Conditional import to support versions of Keras before 2.2
        # TODO: remove in about 6 months (end of 2018)
        
        from keras.engine import saving
        

        if exclude:
            by_name = True

        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        f = h5py.File(filepath, mode='r')
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        keras_model = self.keras_model
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
            else keras_model.layers

        # Exclude some layers
        if exclude:
            layers = filter(lambda l: l.name not in exclude, layers)

        if by_name:
            saving.load_weights_from_hdf5_group_by_name(f, layers)
        else:
            saving.load_weights_from_hdf5_group(f, layers)
        if hasattr(f, 'close'):
            f.close()

        # Update the log directory
        self.set_log_dir(filepath)


        
class HedgeFourCNN(HedgeCNN):
    '''Patch of the MaskRCNN model to allow for four band inputs'''
        

    def build(self, mode, config):
        """Build Mask R-CNN architecture.
            input_shape: The shape of the input image.
            mode: Either "training" or "inference". The inputs and
                outputs of the model differ accordingly.
        """
        assert mode in ['training', 'inference']

        # Image size must be dividable by 2 multiple times
        h, w = config.IMAGE_SHAPE[:2]
        if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
            raise Exception("Image size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")

        # Inputs
        input_image = KL.Input(
            shape=[None, None, 4], name="input_image")
        input_image_meta = KL.Input(shape=[config.IMAGE_META_SIZE],
                                    name="input_image_meta")
        if mode == "training":
            # RPN GT
            input_rpn_match = KL.Input(
                shape=[None, 1], name="input_rpn_match", dtype=tf.int32)
            input_rpn_bbox = KL.Input(
                shape=[None, 4], name="input_rpn_bbox", dtype=tf.float32)

            # Detection GT (class IDs, bounding boxes, and masks)
            # 1. GT Class IDs (zero padded)
            input_gt_class_ids = KL.Input(
                shape=[None], name="input_gt_class_ids", dtype=tf.int32)
            # 2. GT Boxes in pixels (zero padded)
            # [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in image coordinates
            input_gt_boxes = KL.Input(
                shape=[None, 4], name="input_gt_boxes", dtype=tf.float32)
            # Normalize coordinates
            gt_boxes = KL.Lambda(lambda x: norm_boxes_graph(
                x, K.shape(input_image)[1:3]))(input_gt_boxes)
            # 3. GT Masks (zero padded)
            # [batch, height, width, MAX_GT_INSTANCES]
            if config.USE_MINI_MASK:
                input_gt_masks = KL.Input(
                    shape=[config.MINI_MASK_SHAPE[0],
                           config.MINI_MASK_SHAPE[1], None],
                    name="input_gt_masks", dtype=bool)
            else:
                input_gt_masks = KL.Input(
                    shape=[config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], None],
                    name="input_gt_masks", dtype=bool)
        elif mode == "inference":
            # Anchors in normalized coordinates
            input_anchors = KL.Input(shape=[None, 4], name="input_anchors")

        # Build the shared convolutional layers.
        # Bottom-up Layers
        # Returns a list of the last layers of each stage, 5 in total.
        # Don't create the thead (stage 5), so we pick the 4th item in the list.
        if callable(config.BACKBONE):
            _, C2, C3, C4, C5 = config.BACKBONE(input_image, stage5=True,
                                                train_bn=config.TRAIN_BN)
        else:
            _, C2, C3, C4, C5 = resnet_graph(input_image, config.BACKBONE,
                                             stage5=True, train_bn=config.TRAIN_BN)
        # Top-down Layers
        # TODO: add assert to varify feature map sizes match what's in config
        P5 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c5p5')(C5)
        P4 = KL.Add(name="fpn_p4add")([
            KL.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
            KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c4p4')(C4)])
        P3 = KL.Add(name="fpn_p3add")([
            KL.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
            KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c3p3')(C3)])
        P2 = KL.Add(name="fpn_p2add")([
            KL.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
            KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c2p2')(C2)])
        # Attach 3x3 conv to all P layers to get the final feature maps.
        P2 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p2")(P2)
        P3 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p3")(P3)
        P4 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p4")(P4)
        P5 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p5")(P5)
        # P6 is used for the 5th anchor scale in RPN. Generated by
        # subsampling from P5 with stride of 2.
        P6 = KL.MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")(P5)

        # Note that P6 is used in RPN, but not in the classifier heads.
        rpn_feature_maps = [P2, P3, P4, P5, P6]
        mrcnn_feature_maps = [P2, P3, P4, P5]

        # Anchors
        if mode == "training":
            anchors = self.get_anchors(config.IMAGE_SHAPE)
            # Duplicate across the batch dimension because Keras requires it
            # TODO: can this be optimized to avoid duplicating the anchors?
            anchors = np.broadcast_to(anchors, (config.BATCH_SIZE,) + anchors.shape)
            # A hack to get around Keras's bad support for constants
            anchors = KL.Lambda(lambda x: tf.Variable(anchors), name="anchors")(input_image)
        else:
            anchors = input_anchors

        # RPN Model
        rpn = build_rpn_model(config.RPN_ANCHOR_STRIDE,
                              len(config.RPN_ANCHOR_RATIOS), config.TOP_DOWN_PYRAMID_SIZE)
        # Loop through pyramid layers
        layer_outputs = []  # list of lists
        for p in rpn_feature_maps:
            layer_outputs.append(rpn([p]))
        # Concatenate layer outputs
        # Convert from list of lists of level outputs to list of lists
        # of outputs across levels.
        # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
        outputs = list(zip(*layer_outputs))
        outputs = [KL.Concatenate(axis=1, name=n)(list(o))
                   for o, n in zip(outputs, output_names)]

        rpn_class_logits, rpn_class, rpn_bbox = outputs

        # Generate proposals
        # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
        # and zero padded.
        proposal_count = config.POST_NMS_ROIS_TRAINING if mode == "training"\
            else config.POST_NMS_ROIS_INFERENCE
        rpn_rois = ProposalLayer(
            proposal_count=proposal_count,
            nms_threshold=config.RPN_NMS_THRESHOLD,
            name="ROI",
            config=config)([rpn_class, rpn_bbox, anchors])

        if mode == "training":
            # Class ID mask to mark class IDs supported by the dataset the image
            # came from.
            active_class_ids = KL.Lambda(
                lambda x: parse_image_meta_graph(x)["active_class_ids"]
                )(input_image_meta)

            if not config.USE_RPN_ROIS:
                # Ignore predicted ROIs and use ROIs provided as an input.
                input_rois = KL.Input(shape=[config.POST_NMS_ROIS_TRAINING, 4],
                                      name="input_roi", dtype=np.int32)
                # Normalize coordinates
                target_rois = KL.Lambda(lambda x: norm_boxes_graph(
                    x, K.shape(input_image)[1:3]))(input_rois)
            else:
                target_rois = rpn_rois

            # Generate detection targets
            # Subsamples proposals and generates target outputs for training
            # Note that proposal class IDs, gt_boxes, and gt_masks are zero
            # padded. Equally, returned rois and targets are zero padded.
            rois, target_class_ids, target_bbox, target_mask =\
                DetectionTargetLayer(config, name="proposal_targets")([
                    target_rois, input_gt_class_ids, gt_boxes, input_gt_masks])

            # Network Heads
            # TODO: verify that this handles zero padded ROIs
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox =\
                fpn_classifier_graph(rois, mrcnn_feature_maps, input_image_meta,
                                     config.POOL_SIZE, config.NUM_CLASSES,
                                     train_bn=config.TRAIN_BN,
                                     fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE)

            mrcnn_mask = build_fpn_mask_graph(rois, mrcnn_feature_maps,
                                              input_image_meta,
                                              config.MASK_POOL_SIZE,
                                              config.NUM_CLASSES,
                                              train_bn=config.TRAIN_BN)

            # TODO: clean up (use tf.identify if necessary)
            output_rois = KL.Lambda(lambda x: x * 1, name="output_rois")(rois)

            # Losses
            rpn_class_loss = KL.Lambda(lambda x: rpn_class_loss_graph(*x), name="rpn_class_loss")(
                [input_rpn_match, rpn_class_logits])
            rpn_bbox_loss = KL.Lambda(lambda x: rpn_bbox_loss_graph(config, *x), name="rpn_bbox_loss")(
                [input_rpn_bbox, input_rpn_match, rpn_bbox])
            class_loss = KL.Lambda(lambda x: mrcnn_class_loss_graph(*x), name="mrcnn_class_loss")(
                [target_class_ids, mrcnn_class_logits, active_class_ids])
            bbox_loss = KL.Lambda(lambda x: mrcnn_bbox_loss_graph(*x), name="mrcnn_bbox_loss")(
                [target_bbox, target_class_ids, mrcnn_bbox])
            mask_loss = KL.Lambda(lambda x: mrcnn_mask_loss_graph(*x), name="mrcnn_mask_loss")(
                [target_mask, target_class_ids, mrcnn_mask])

            # Model
            inputs = [input_image, input_image_meta,
                      input_rpn_match, input_rpn_bbox, input_gt_class_ids, input_gt_boxes, input_gt_masks]
            if not config.USE_RPN_ROIS:
                inputs.append(input_rois)
            outputs = [rpn_class_logits, rpn_class, rpn_bbox,
                       mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_mask,
                       rpn_rois, output_rois,
                       rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss, mask_loss]
            model = KM.Model(inputs, outputs, name='mask_rcnn')
        else:
            # Network Heads
            # Proposal classifier and BBox regressor heads
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox =\
                fpn_classifier_graph(rpn_rois, mrcnn_feature_maps, input_image_meta,
                                     config.POOL_SIZE, config.NUM_CLASSES,
                                     train_bn=config.TRAIN_BN,
                                     fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE)

            # Detections
            # output is [batch, num_detections, (y1, x1, y2, x2, class_id, score)] in
            # normalized coordinates
            detections = DetectionLayer(config, name="mrcnn_detection")(
                [rpn_rois, mrcnn_class, mrcnn_bbox, input_image_meta])

            # Create masks for detections
            detection_boxes = KL.Lambda(lambda x: x[..., :4])(detections)
            mrcnn_mask = build_fpn_mask_graph(detection_boxes, mrcnn_feature_maps,
                                              input_image_meta,
                                              config.MASK_POOL_SIZE,
                                              config.NUM_CLASSES,
                                              train_bn=config.TRAIN_BN)

            model = KM.Model([input_image, input_image_meta, input_anchors],
                             [detections, mrcnn_class, mrcnn_bbox,
                                 mrcnn_mask, rpn_rois, rpn_class, rpn_bbox],
                             name='mask_rcnn')

        # Add multi-GPU support.
        if config.GPU_COUNT > 1:
            #from mrcnn.parallel_model import ParallelModel
            #from parallel import ParallelModel
            #from keras.utils import multi_gpu_model
            from parallel import multi_gpu_model
            #model = ParallelModel(model, config.GPU_COUNT)
            model = multi_gpu_model(model, gpus=2)
            
            
        return model
#########################################
# Configuration settings for training and inference modes
#########################################

class HedgeConfig(Config):
    """Configuration for training on the hedges dataset.
    Derives from the base Config class and overrides values specific
    to the hedges dataset.
    """
    # Give the configuration a recognizable name
    NAME = "hedges"
    
    GPU_COUNT = 1
    
    #Can train on more images per batch when we only unfreeze the network heads, otherwise we need to train on less per GPU (below)
    IMAGES_PER_GPU = 6
#    # Num of training images / batch size (I add a few more steps because data augmentation creates a few more images?)
    STEPS_PER_EPOCH = 1916
#    # Num of valid. images / batch size
    VALIDATION_STEPS = 28
    
 #   IMAGES_PER_GPU = 3
    # Num of training images / batch size (I add a few more steps because data augmentation creates a few more images?)
    #STEPS_PER_EPOCH = 85
    # Num of valid. images / batch size
    #VALIDATION_STEPS = 28
    
    
    
    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + hedge

    # Use small images for faster training by allowing more images per batch. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 320
    IMAGE_MAX_DIM = 320
    
    #If 'adam' then uses adam optimizer, 'SGD' is other option
    OPTIMIZER = 'SGD'

    #size of anchors for the object region proposals
    RPN_ANCHOR_SCALES = (30, 60, 100, 150, 200)  # anchor side in pixels

    #the minimum confidence required for a region proposal to be kept as a positive detection
    DETECTION_MIN_CONFIDENCE = 0.7
    #amount of overlap two detected bounding boxes need to have before one of them gets supressed.
    DETECTION_NMS_THRESHOLD = 0.2
    
    #Want to have long and narrow anchors, and one square shaped.
    RPN_ANCHOR_RATIOS = [0.2, 1.2, 5]
    
    #Values from the Mask R-CNN paper
    RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    
    #amount of overlap that two bounding box proposals need to have for one of them to be supressed. So if its low then we get less BB.
    RPN_NMS_THRESHOLD = 0.9
    
    #True is good when using high resolution images. Planet isnt that high?
    USE_MINI_MASK = False
    
    #mean pixel values for each band.
    #MEAN_PIXEL = np.array([338, 390, 809])
    MEAN_PIXEL = np.array([0, 0, 0])
    
    MAX_GT_INSTANCES = 10

    DETECTION_MAX_INSTANCES = 50

    #keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 200
    
    LEARNING_RATE = 0.01
    LEARNING_MOMENTUM = 0.93
    #LEARNING_MOMENTUM = 0.90
    
    # Weight decay regularization
    WEIGHT_DECAY = 0.0001
    
    #only train if batch size is large
    TRAIN_BN = False  
    
    #loss weights. 
    #Giving more importance to the accurate classification of region proposals as well as masking. Bounding box accuracy is less important 
    LOSS_WEIGHTS = {
        "rpn_class_loss": 25.,
        "rpn_bbox_loss": 0.6,
        "mrcnn_class_loss": 6.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 2.0
    }
    
class InfHedgeConfig(Config):
    """Configuration for training on the hedges dataset.
    Derives from the base Config class and overrides values specific
    to the hedges dataset.
    """
    # Give the configuration a recognizable name
    NAME = "hedges"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    
    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + hedge

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 320
    IMAGE_MAX_DIM = 320

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (30, 60, 100, 150, 200)   # anchor side in pixels
    
    RPN_NMS_THRESHOLD = 0.9
    
    #Use for mAP
    DETECTION_MIN_CONFIDENCE = 0.80
    DETECTION_NMS_THRESHOLD = 0.95
    
    #Use for creating the mask maps
    #can play with this to see what gives best accuracy
    #DETECTION_MIN_CONFIDENCE = 0.60
    #DETECTION_NMS_THRESHOLD = 0.95
    
    #todo
    RPN_ANCHOR_RATIOS = [0.2, 1.2, 5]

    #True is good when using high resolution images. Planet isnt that high?
    USE_MINI_MASK = False
    
    #mean pixel values for each band.
    #MEAN_PIXEL = np.array([338, 390, 809])
    MEAN_PIXEL = np.array([0, 0, 0])
    
    MAX_GT_INSTANCES = 10

    DETECTION_MAX_INSTANCES = 50

    #keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 200
   

    # Weight decay regularization
    WEIGHT_DECAY = 0.0001
    
    #maybe set to true since the batches might be large enough with DLR GPUs
    TRAIN_BN = False  # Defaulting to False since batch size is often small
    
    
    
    
class HedgeConfigFour(Config):
    """Configuration for training on the hedges dataset.
    Derives from the base Config class and overrides values specific
    to the hedges dataset.
    """
    # Give the configuration a recognizable name
    NAME = "hedges"
    
    
    # Number  of input channels
    IMAGE_CHANNEL_COUNT = 4
    
    
    GPU_COUNT = 2
    
    #Can train on more images per batch when we only unfreeze the network heads, otherwise we need to train on less per GPU (below)
    IMAGES_PER_GPU = 1
#    # Num of training images / batch size (I add a few more steps because data augmentation creates a few more images?)
    STEPS_PER_EPOCH = 64
#    # Num of valid. images / batch size
    VALIDATION_STEPS = 20
    
 #   IMAGES_PER_GPU = 3
    # Num of training images / batch size (I add a few more steps because data augmentation creates a few more images?)
  #  STEPS_PER_EPOCH = 171
    # Num of valid. images / batch size
   # VALIDATION_STEPS = 57
    
    
    
    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + hedge
    
    # Use small images for faster training by allowing more images per batch. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 320
    IMAGE_MAX_DIM = 320
    
    #If 'adam' then uses adam optimizer, 'SGD' is other option
    OPTIMIZER = 'SGD'

    #size of anchors for the object region proposals
    RPN_ANCHOR_SCALES = (30, 60, 100, 150, 200)  # anchor side in pixels

    #the minimum confidence required for a region proposal to be kept as a positive detection
    DETECTION_MIN_CONFIDENCE = 0.7
    #amount of overlap two detected bounding boxes need to have before one of them gets supressed.
    DETECTION_NMS_THRESHOLD = 0.2
    
    #Want to have long and narrow anchors, and one square shaped.
    RPN_ANCHOR_RATIOS = [0.2, 0.4, 1.2, 2.5, 5]
    
    #Values from the Mask R-CNN paper
    RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    
    #amount of overlap that two bounding box proposals need to have for one of them to be supressed. So if its low then we get less BB.
    RPN_NMS_THRESHOLD = 0.9
    
    #True is good when using high resolution images. Planet isnt that high?
    USE_MINI_MASK = False
    
    #mean pixel values for each band.
    #MEAN_PIXEL = np.array([338, 390, 809])
    MEAN_PIXEL = np.array([0, 0, 0])
    
    MAX_GT_INSTANCES = 10

    DETECTION_MAX_INSTANCES = 50

    #keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 200
    
    LEARNING_RATE = 0.01
    #LEARNING_MOMENTUM = 0.93
    LEARNING_MOMENTUM = 0.90
    
    # Weight decay regularization
    WEIGHT_DECAY = 0.0001
    
    #only train if batch size is large
    TRAIN_BN = True
    
    #loss weights. 
    #Giving more importance to the accurate classification of region proposals as well as masking. Bounding box accuracy is less important 
    LOSS_WEIGHTS = {
        "rpn_class_loss": 25.,
        "rpn_bbox_loss": 0.6,
        "mrcnn_class_loss": 6.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 2.0
    }
############################################################
#  Dataset
############################################################

class HedgeDataset(utils.Dataset):
        
    def mask_anno(self, dataset_dir, json_path):
        with open(os.path.join(dataset_dir, json_path), "r") as read_file:
            data = json.load(read_file) 
        self.mask_a = data['annotations']
        
    def add_hedge(self, dataset_dir, subset):
        """Load a subset of the hedge dataset.
        dataset_dir: Root directory of the dataset
        subset: string giving the name of the folder where the data we want
            to load is located
        """
        # Add classes. We have one class.
        # Naming the dataset hedge2019dataset, and the class hedge
        self.add_class("Hedges2019Dataset", 1, "hedge")
        
        subset_dir = subset
        dataset_dir = os.path.join(dataset_dir, subset_dir)
        
        # Get image ids from directory names
        image_ids = GetFiles(dataset_dir, ending='.png')

        # Add images
        for image_id in image_ids:
            try: 
                i = image_id.rsplit('/', 1)[1]
            except IndexError:
                i = image_id.rsplit('\\', 1)[1]
            
           
            image_inf = {'id': i,
                          'path': image_id,
                          'source': 'Hedges2019Dataset'
                    }
            self.image_info.append(image_inf)
            
    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array. Uses the 
        self.image_info['path'] dictionary value to load the image so make sure 
        that the full path to each image is set by the add_hedge function first
        
        Parameters:
            image_id: int
                Should be an interative number which indexes individual image 
                entries in the self.image_info list. 
        
        """
        # Load image
        image = np.zeros(shape=(320, 320, 3)).astype(np.uint16)
        with rasterio.open(self.image_info[image_id]['path']) as src:
            image[:,:,0] = src.read(1)
            image[:,:,1] = src.read(2)
            image[:,:,2] = src.read(3)
        #im_mean = np.mean(image)
        #im_std = np.std(image)
        
        #image = (image-im_mean)/im_std
        return image

   
    
    
    def load_mask(self, image_id):
        """Generate instance masks for a single image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        instance_masks = []
        check = self.image_info[image_id]['id']
        #for entry in self.image_info:
        #    if entry['index_ordered'] == image_id:
         #       check = entry['id']
        for img in self.mask_a:
            for seg in img:
                if seg['filename'] == check:
                    y = np.squeeze(seg['segmentation'])[::2]
                    x = np.squeeze(seg['segmentation'])[1::2]
                    imge = np.zeros((320, 320), np.uint8)
                    xx, yy = polygon(x, y)
                    imge[xx, yy] = 1
                    #imge = scipy.ndimage.morphology.binary_dilation(imge)
                    instance_masks.append(imge)
        masks = np.zeros((320,320,len(instance_masks)))
        
        for n, i in enumerate(instance_masks):
            masks[:,:,n] = i.astype(np.uint8)
        class_id = np.ones(len(instance_masks), np.int32)
        
        return masks, class_id
    
    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "Hedges2019Dataset":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

class HedgeDatasetFour(HedgeDataset):
    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array. Uses the 
        self.image_info['path'] dictionary value to load the image so make sure 
        that the full path to each image is set by the add_hedge function first
        
        Parameters:
            image_id: int
                Should be an interative number which indexes individual image 
                entries in the self.image_info list. 
        
        """
        # Load image
        image = np.zeros(shape=(320, 320, 4)).astype(np.uint16)
        with rasterio.open(self.image_info[image_id]['path']) as src:
            image[:,:,0] = src.read(1)
            image[:,:,1] = src.read(2)
            image[:,:,2] = src.read(3)
            image[:,:,3] = src.read(4)

        return image

############################################################
#  Training
############################################################

def train(model, dataset_dir, subset, json, weight_path, val_dir):
    """Train the model."""
    # Training dataset.
    dataset_train = HedgeDataset()
    dataset_train.add_hedge(dataset_dir, subset)
    dataset_train.add_class('Hedges2019Dataset', 1, 'Hedge')
    dataset_train.prepare()
    dataset_train.mask_anno(dataset_dir, json)

    # Validation dataset
    dataset_val = HedgeDataset()
    dataset_val.add_hedge(dataset_dir, val_dir)
    dataset_val.add_class('Hedges2019Dataset', 1, 'Hedge')
    dataset_val.prepare()
    dataset_val.mask_anno(dataset_dir, json)

    check = ModelCheckpoint(weight_path, monitor='val_loss', verbose=0, 
                            save_best_only=False, save_weights_only=True, mode='min')
    tensor = TensorBoard(log_dir='./logs/', histogram_freq=0, write_graph=True, write_images=True)
 #   reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, mode='auto', min_delta=0.1, cooldown = 5, min_lr=10e-9)
    
    def step_decay(epoch, lr):
        lrate = lr
        if epoch % 40 == 0:
            lrate = lr * 0.1
        return lrate
    
   # def exp_decay(epoch, lr):
    #    lrate = lr * np.exp(-0.1*epoch)
     #   return lrate
    
    schedule = LearningRateScheduler(step_decay, verbose=1)
    callbacks = [check, tensor, schedule]
    # If starting from imagenet, train heads only for a bit
    # since they have random weights
    print("Train network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=model.config.LEARNING_RATE,
                epochs=100,
                #augmentation=augmentation, 
                custom_callbacks=callbacks,
                layers='heads')
    
    
############################################################
#  Detection
############################################################

def detect(model, dataset_dir, subset, output_dir):
    """Run detection on images in the given directory."""
    print("Running on {}".format(dataset_dir))

    # Create directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    submit_dir = "submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    submit_dir = os.path.join(output_dir, submit_dir)
    os.makedirs(submit_dir)
    
    # Read dataset
    dataset = HedgeDataset()
    dataset.add_hedge(dataset_dir, subset)
    dataset.add_class('Hedges2019Dataset', 1, 'Hedge')
    dataset.prepare()
    # Load over images
    submission = []
    for image_id in dataset.image_ids:
        # Load image and run detection
        image = dataset.load_image(image_id)
        # Detect objects
        r = model.detect([image], verbose=0)[0]
        # Encode image to RLE. Returns a string of multiple lines
        #source_id = dataset.image_info[image_id]["id"]
        #rle = mask_to_rle(source_id, r["masks"], r["scores"])
        #submission.append(rle)
        submission.append(r)
        # Save image with masks
        visualize.display_instances(
            image, r['rois'], r['masks'], r['class_ids'],
            dataset.class_names, r['scores'],
            show_bbox=True, show_mask=True,
            title="Predictions")
        plt.savefig("{}/{}.png".format(submit_dir, dataset.image_info[image_id]["id"]))

    # Save to csv file
    submission = "ImageId,EncodedPixels\n" + "\n".join(submission)
    file_path = os.path.join(submit_dir, "submit.csv")
    with open(file_path, "w") as f:
        f.write(submission)
    print("Saved to ", submit_dir)


############################################################
#  Command Line
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Mask R-CNN for hedge detection and segmentation')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'detect'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/dataset/",
                        help='Root directory of the dataset')
  #  parser.add_argument('--weights', required=True,
   #                     metavar="/path/to/weights.h5",
    #                    help="Can be 'last', 'crowdai', or 'imagenet'")
    parser.add_argument('--logs', required=False,
                        default='/logs/',
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--subset', required=False,
                        metavar="Dataset sub-directory",
                        help="Subset of dataset to run prediction on")
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "detect":
        assert args.subset, "Provide --subset to run prediction on"

 #   print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    if args.subset:
        print("Subset: ", args.subset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = HedgeConfigFour()
  #  else:
   #     config = HedgeConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = HedgeFourCNN(mode="training", config=config,
                                  model_dir=args.logs)

    # Train or evaluate
    if args.command == "train":
        trainfour(model, args.dataset, args.subset)
    elif args.command == "detect":
        detect(model, args.dataset, args.subset)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'detect'".format(args.command))
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 16:22:48 2019

@author: ahls_st
"""
from scipy.ndimage import label
import numpy as np


def accuracy(gt, pred):
    tp = np.sum(np.logical_and(gt==1, pred ==1))
    fp = np.sum(np.logical_and(gt==0, pred ==1))
    fn = np.sum(np.logical_and(gt==1, pred ==0))
    output = tp/(tp+fp+fn)
    return output

def ap(gt, pred):
    """Out of all the pixels predicted as a hedge, what percentage was correct
    """
    tp = np.sum(np.logical_and(gt==1, pred ==1))
    fp = np.sum(np.logical_and(gt==0, pred ==1))
    output = tp/(tp+fp)
    return output

def batch_accuracy_ap(image_ids):
    total_acc = []
    mAP = []
    for image_id in image_ids:
        
        # Load image
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            model.load_image_gt(dataset_val, d_cnn.config,
                                   image_id, use_mini_mask=False)
        
        # Get all gt masks into one image
        gt = np.zeros((320,320)).astype(np.uint8)
        for i in range(gt_mask.shape[2]):
            gt += gt_mask[:,:,i].astype(np.uint8)
            
        # Run object detection
        results = d_cnn.detect([image], verbose=0)
        r = results[0]
        
        # Get all predicted object masks into a list
       # all_masks = []
        pred_mask = np.zeros((320,320)).astype(np.uint8)
        for i in range(r['masks'].shape[2]):
            pred_mask += r['masks'][:,:,i].astype(np.uint8)
            
        # make sure its a binary mask
        pred_mask[pred_mask != 0] = 1

        # Compute iou
        acc = accuracy(gt, pred_mask)
        app = ap(gt, pred_mask)
        
        total_acc.append(acc)
        mAP.append(app)
    output1 = np.sum(total_acc)/len(total_acc)
    output2 = np.sum(mAP)/len(mAP)
    return output1, output2



def find_matching_masks(gt_masks, pred_masks, overlap_for_positive_match = 0.5):
    """
    Parameters
    ----------
    
    Returns
    -------
    matches: list of tuples
        Tuples giving the indexes for a GT mask which was matched with a predicted
        mask. First tuple values are the GT index, while the second is the index
        value of the predicted mask.
        
    false_negatives: list of ints
        Indexes of the GT masks which were not matched with any mask (either no
        predicted mask matched or the overlap between pred and GT was too low)
        
    false_positves: list of ints
        Indexes of the predicted masks which were found but did not match to any
        GT masks (either no GT mask or the overlap between pred and GT was too low)
    
    """
    matches = []
    false_negative = []
    false_positive = []
    
    if pred_masks.ndim == 1:
        pred_masks.reshape(pred_masks.shape[0], 1)
        assert(pred_masks.ndim>1), 'prediction masks lack self.shape[1] value'
    
    if gt_masks.ndim == 1:
        gt_masks.reshape(gt_masks.shape[0], 1)
        assert(gt_masks.ndim>1), 'ground truth mask lack self.shape[1] value'
        
    for gt in range(gt_masks.shape[1]):
        gt_size = np.sum(gt_masks[:,gt])
        
        for pred in range(pred_masks.shape[1]):        
            if np.sum((gt_masks[:, gt] * pred_masks[:, pred])) > (gt_size*overlap_for_positive_match):
                matches.append((gt, pred))
                
        # check if the current gt mask has been matched. If not, then we have a false negative
        if gt not in [row[0] for row in matches]:
            false_negative.append(gt)
    
    # check if any pred masks went unmatched, if so then it is false positive
    for p in range(pred_masks.shape[1]):
        if p not in [row[1] for row in matches]:
            false_positive.append(p)    
            
    return matches, false_negative, false_positive



def single_image_instance_detection_metric(gt, pred, overlap_threshold = 0.5):
    """
    Will find out the number of TP, FP, and FN between a single GT and predicted
    mask image. Default overlap between the pred and GT is set to 50%.
    
    
    """
    # give each binary hedge mask its own unique numeric label
    labelled_gt, num_labels_gt = label(gt)
    labelled_pred, num_labels_pred = label(pred)
    
    # flatten the images for faster processing
    labelled_gt = labelled_gt.flatten()
    labelled_pred = labelled_pred.flatten()
    
    # create a container for the individual masks
    gt_masks = np.zeros((320*320, num_labels_gt)).astype(np.uint8)
    
    # get each mask instance into the container
    for i in range(num_labels_gt):
        index = np.where(labelled_gt == i+1)[0]
        for j in range(len(index)):
            gt_masks[index[j], i] = 1    
    
    # create container for pred masks
    pred_masks = np.zeros((320*320, num_labels_pred)).astype(np.uint8)
    
    #get pred instances into container
    for i in range(num_labels_pred):
        ind = np.where(labelled_pred == i+1)[0]
        for j in range(len(ind)):
            pred_masks[ind[j], i] = 1
    
    #find the indices of the gt and pred masks that match or are fp/fn
    matches, fn, fp = find_matching_masks(gt_masks, pred_masks, overlap_threshold)
    
    return len(matches), len(fn), len(fp)

# SPDX-License-Identifier: EUPL-1.2
# Copyright 2024 dp-lab Università della Svizzera Italiana

import argparse
import sys, os, json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from glob import glob
import pandas as pd
from scipy import ndimage
import skimage.transform as trans
from skimage.feature import peak_local_max
from utils.windowed_utils import *
from utils.windower import windower
from utils.plots import plotL1, plotL3, draw_center
from L3scripts.model_L3 import dice_coef

FIXED_SIZE = (512, 1024, 3)
INPUT_SIZE = (128, 256, 3)

#---------------------------------------------------------------------------c

def scores(VAT_s, SAT_s, SMA_s, L1_s, L1, L3, spacing, outFolder, suffix=""):
    ## Compute scores
    # calculate the scores and write them as a list 
    # of numbers separated by , in DATA_DIR+'/'+'scores.txt'
    areaL3SAT = np.sum(SAT_s.flatten())*(spacing[0]*spacing[0])*0.01
    densitystdL3SAT = np.std(L3[SAT_s==1])
    areaL3VAT = np.sum(VAT_s.flatten())*(spacing[0]*spacing[0])*0.01
    areaL3SMA = np.sum(SMA_s.flatten())*(spacing[0]*spacing[0])*0.01
    
    L1_slice_trabecular = L1[L1_s==1]
    densitystdL1trabecular = np.std(L1_slice_trabecular)
    densityavgL1trabecular = np.mean(L1_slice_trabecular)
    areaL1trabecular = np.sum(L1_s)*(spacing[0]*spacing[0])*0.01
    strScores = {"L1BMDavg":                str(round(densityavgL1trabecular, 1)),
                 "L1BMDstd":                str(round(densitystdL1trabecular, 1)),
                 "L1trabecularbonearea":    str(round(areaL1trabecular, 1)),
                 "L3SATarea":               str(round(areaL3SAT, 1)),
                 "L3SMAarea":               str(round(areaL3SMA,1)),
                 "L3VATarea":               str(round(areaL3VAT,1)),
                 "L3SATdensitystd":         str(round(densitystdL3SAT, 1))}
    with open(os.path.join(outFolder, 'scores{}.json'.format(suffix)), 'w') as f:
        json.dump(strScores, f, indent=4)
    return strScores

#---------------------------------------------------------------------------

def projection(volume, spacing):
    #extract sagital & coronal projections, scaled and save a preview in the input folder
    sagital = extract_images(volume, axis=0, spacing=spacing) #Sagital
    sagital = np.flipud(sagital)
    nopadded_shape = sagital.shape
    
    if sagital.shape[1] >= FIXED_SIZE[1]:
        sagital = sagital[:, :FIXED_SIZE[1],:]
    elif sagital.shape[1] < FIXED_SIZE[1]:
        sagital = np.pad(sagital, ((0,0),(0,FIXED_SIZE[1]-sagital.shape[1]),(0,0)), mode='edge')
    return sagital, nopadded_shape

#---------------------------------------------------------------------------

def extract_seg_slice(seg):
    try:
        L1 = ndimage.center_of_mass(seg==1)
        L1 = (int(round(L1[0])),int(round(L1[1])),int(round(L1[2])))
        L3 = ndimage.center_of_mass(seg==3)
        L3 = (int(round(L3[0])),int(round(L3[1])),int(round(L3[2])))
    except:
        L1 = [None,None,None]
        L3 = [None,None,None]
    return L1, L3

#---------------------------------------------------------------------------

def import_images(img_path):
    ## Read the nifti image
    (img, spacing) = readData(img_path, order=3)
    ## Save projections: extract the 3 windowed channels and resize the image to be isotropic
    sagital, nopadded_shape = projection(img, spacing)
    return sagital, img, spacing, nopadded_shape

#---------------------------------------------------------------------------------------------

def L3segmentation(img, outFolder, model):
    # U-net to segment the CT slice at the L3 spinal level in the following regions: 
    # visceral adipose tissue (VAT), subcutaneous adipose tissue (SAT), skeletal muscle area (SMA).
    rgb_img = np.zeros((img.shape[0], img.shape[1], 3))
    rgb_img[:,:,0] = windower(img, -1024, 2048)
    rgb_img[:,:,1] = windower(img, -190, -30)
    rgb_img[:,:,2] = windower(img, 40, 100)

    im_width, im_height = img.shape #(512, 512)
    
    X = np.zeros((1, im_height, im_width, 3), dtype=np.float32)
    X[0,:,:,:] = rgb_img/255.
    X = skimage.transform.resize(X, (1, im_width, im_height, 3), mode='constant',cval=0,anti_aliasing=True,preserve_range=True,order=3)

    results = model.predict(X, verbose=0)
    results = skimage.transform.resize(results, (1, im_height, im_width, results.shape[-1]), 
                                       mode='constant', cval=0, anti_aliasing=True,
                                       preserve_range=True, order=0)
    res_seg = np.round(results)[0,:,:,:]
    
    # Set the pixels on the edges left/right to zero with a thickness of 20 pixels 
    thickness = 20
    res_seg[:, :thickness, 1:] = 0  # Left edge
    res_seg[:, -thickness:, 1:] = 0  # Right edge

    plotL3(img, res_seg, outFolder)
    vat, sat, sma = res_seg[:,:,1].astype(int), res_seg[:,:,2].astype(int), res_seg[:,:,3].astype(int)
    return vat, sat, sma

#---------------------------------------------------------------------------------------------

def L1segmentation(img, outFolder, model):
    # U-net to segment the CT slice at the L1 spinal level in the following regions: 
    # L1 Trabecular bone area and L1 Cortical area.
    im_width, im_height = img.shape #(512, 512)
    
    X = np.zeros((1, im_height, im_width, 1), dtype=np.float32)
    X[0,:,:,0] = windower(img, -1024, 500)/255. #windower(img, img.min(), img.max()) / 255

    results = model.predict(X, verbose=0)
    res_seg = np.round(results)[0,:,:,:]
    
    # Set the pixels on the edges to zero with a thickness of 20 pixels
    thickness = 20
    res_seg[:thickness, :, 1:] = 0  # Top edge
    res_seg[-thickness:, :, 1:] = 0  # Bottom edge
    res_seg[:, :thickness, 1:] = 0  # Left edge
    res_seg[:, -thickness:, 1:] = 0  # Right edge

    plotL1(np.fliplr(img), np.fliplr(res_seg), outFolder)
    cort, spun = np.fliplr(res_seg[:,:,2].astype(int)), np.fliplr(res_seg[:,:,1].astype(int))
    return cort, spun

#---------------------------------------------------------------------------

def process(MODEL_SLICE, MODEL_L1, MODEL_L3, img_path, outFolder):
      
    # Read the nifti image
    sagital, volume, spacing, nopadded_shape = import_images(img_path)

    X = np.zeros((1, FIXED_SIZE[0], FIXED_SIZE[1], FIXED_SIZE[2]), dtype=np.float32)
    X[0,:,:,:] = sagital.squeeze()/255.
    X = trans.resize(X, (1, INPUT_SIZE[0], INPUT_SIZE[1], INPUT_SIZE[2]), anti_aliasing=True)
    # Predict 
    results = MODEL_SLICE.predict(X, verbose=0)
    results = trans.resize(results, (1, FIXED_SIZE[0], FIXED_SIZE[1], results.shape[-1]), anti_aliasing=True)
    results = (results-results.min())/(results.max()-results.min())
    res = results[0,:,:,0]*255.
    
    # Save prediction
    #plt.imsave(os.path.join(outFolder,'prediction.png'),res.astype(np.uint8))
    # Find Local Maxima                
    [pr_L3y, pr_L3x], [pr_L1y, pr_L1x] = peak_local_max(res, min_distance=10, num_peaks=2)
    if pr_L1x<pr_L3x:
        pr_L1x, pr_L3x = pr_L3x, pr_L1x
        pr_L1y, pr_L3y = pr_L3y, pr_L1y
    centers = ((pr_L1x, pr_L1y), (pr_L3x, pr_L3y))
    # Draw ceneters
    draw_center(sagital[:,:,0], res, nopadded_shape, outFolder)
        
    L1_slice = volume[:,:,round(centers[0][0]*spacing[0]/spacing[2])]
    L3_slice = volume[:,:,round(centers[1][0]*spacing[0]/spacing[2])]
    # Save slices
    plt.imsave(os.path.join(outFolder,'L3slice.png'),L3_slice,cmap='gray')
    plt.imsave(os.path.join(outFolder,'L1slice.png'),L1_slice,cmap='gray')
    ## L1 segmentation
    cort, L1_mask = L1segmentation(np.fliplr(L1_slice), outFolder, MODEL_L1)
    ## L3 segmentation
    VAT_mask, SAT_mask, SMA_mask = L3segmentation(L3_slice, outFolder, MODEL_L3)
    # Compute scores
    predScores = scores(VAT_mask, SAT_mask, SMA_mask, L1_mask, L1_slice, L3_slice, spacing, outFolder)
       
    return predScores
    
#---------------------------------------------------------------------------

def main(args):

    img_path = args.input_path
    out_path = args.output_folder
    os.makedirs(out_path, exist_ok=True)
    WEIGHTS_SLICE = args.weights_slicer
    WEIGHTS_L1 = args.weights_L1
    WEIGHTS_L3 = args.weights_L3
    
    print('Loading models ... ')
    MODEL_SLICE = tf.keras.models.load_model(WEIGHTS_SLICE)
    MODEL_L1 = tf.keras.models.load_model(WEIGHTS_L1, custom_objects={"dice_coef": dice_coef })
    MODEL_L3 = tf.keras.models.load_model(WEIGHTS_L3, custom_objects={"dice_coef": dice_coef })

    predScores = process(MODEL_SLICE, MODEL_L1, MODEL_L3, img_path, out_path)
    print(predScores)

if __name__=="__main__":
    """Read command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", help='Path to NiFTI volume')
    parser.add_argument("--output_folder", help='Path to output')
    parser.add_argument("--weights_slicer", help='Path to weights slicer', default='./slicer/weights/multires.hdf5')
    parser.add_argument("--weights_L1", help='Path to weights L1 segmentation', default='./slicer/weights/unet_L1.hdf5')
    parser.add_argument("--weights_L3", help='Path to weights L3 segmentation',default='./slicer/weights/unet_L3.hdf5')
    args = parser.parse_args()
    main(args)


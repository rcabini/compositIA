# SPDX-License-Identifier: EUPL-1.2
# Copyright 2024 dp-lab Università della Svizzera Italiana

import numpy as np
import os
import matplotlib.pyplot as plt
from .windower import windower

#---------------------------------------------------------------------------

def draw_center(projection, scores_smooth, nonpadded_shape, DATA_DIR):
    # RGB image with L1 and L3 positions
    img = np.repeat(projection[...,None],3,axis=-1).astype(np.ubyte)
    scores_smooth = windower(scores_smooth, scores_smooth.min(), scores_smooth.max())
    img[:,:,0] = np.maximum(projection, scores_smooth)
    img = img[:,:nonpadded_shape[1],:]
    plt.imsave(os.path.join(DATA_DIR,'planes.png'),img)

#---------------------------------------------------------------------------

def plotL3(L3slice, seg, DATA_DIR):
	# RGB image with SAT SMA and VAT
	L3slice = windower(L3slice, L3slice.min(), L3slice.max())
	img = np.repeat(L3slice[...,None],3,axis=-1).astype(np.ubyte)
	img[:,:,0] = np.maximum(L3slice, seg[:,:,2]*255)
	img[:,:,1] = np.maximum(L3slice, seg[:,:,3]*255)
	img[:,:,2] = np.maximum(L3slice, seg[:,:,1]*255)
	plt.imsave(os.path.join(DATA_DIR,'L3segmentation.png'),img)

#---------------------------------------------------------------------------

def plotL1(L1slice, seg, DATA_DIR):
	# RGB image with L1 segmentations
	L1slice = windower(L1slice, L1slice.min(), L1slice.max())
	img = np.repeat(L1slice[...,None],3,axis=-1).astype(np.ubyte)
	img[:,:,0] = np.maximum(L1slice, seg[:,:,1]*255)
	img[:,:,1] = np.maximum(L1slice, seg[:,:,2]*255)
	plt.imsave(os.path.join(DATA_DIR,'L1segmentation.png'),img)

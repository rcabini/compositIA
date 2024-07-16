import numpy as np
import os
import matplotlib.pyplot as plt
from L3scripts.data_generator import windower

def plotL1L3(projection, scoresL1_smooth, scoresL3_smooth, DATA_DIR):
	# RGB image with L1 and L3 positions
	img = np.repeat(projection[...,None],3,axis=-1).astype(np.ubyte)
	scoresL1_smooth = windower(scoresL1_smooth, scoresL1_smooth.min(), scoresL1_smooth.max())
	scoresL3_smooth = windower(scoresL3_smooth, scoresL3_smooth.min(), scoresL3_smooth.max())
	img[:,:,0] = np.maximum(projection, scoresL1_smooth)
	img[:,:,1] = np.maximum(projection, scoresL3_smooth)
	plt.imsave(os.path.join(DATA_DIR,'combinedSagital.png'),img)

def plotL3(L3slice, seg, DATA_DIR):
	# RGB image with SAT SMA and VAT
	L3slice = windower(L3slice, L3slice.min(), L3slice.max())
	img = np.repeat(L3slice[...,None],3,axis=-1).astype(np.ubyte)
	img[:,:,0] = np.maximum(L3slice, seg[:,:,2]*255)
	img[:,:,1] = np.maximum(L3slice, seg[:,:,3]*255)
	img[:,:,2] = np.maximum(L3slice, seg[:,:,1]*255)
	plt.imsave(os.path.join(DATA_DIR,'combinedSATSMAVAT.png'),img)

def plotL1(L1slice, seg, DATA_DIR):
	# RGB image with L1 segmentations
	L1slice = windower(L1slice, L1slice.min(), L1slice.max())
	img = np.repeat(L1slice[...,None],3,axis=-1).astype(np.ubyte)
	img[:,:,1] = np.maximum(L1slice, seg[:,:,0]*255)
	img[:,:,0] = np.maximum(L1slice, seg[:,:,1]*255)
	plt.imsave(os.path.join(DATA_DIR,'pred_L1slice.png'),img)

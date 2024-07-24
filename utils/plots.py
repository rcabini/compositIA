import numpy as np
import os
import matplotlib.pyplot as plt
from L3scripts.data_generator import windower

#---------------------------------------------------------------------------

def draw_center(buff_s, results, centers, DATA_DIR):
    # Intermediate step to dar the L1 and L3 centers on a preview image for verification
    plt.imshow(buff_s, cmap='gray')
    plt.imshow(results, cmap='coolwarm', alpha=0.5)
    plt.plot(centers[0][0],centers[0][1],'cx', linewidth=7.0)
    plt.plot(centers[1][0],centers[1][1],'bx', linewidth=7.0)
    plt.axis("off")
    plt.savefig(os.path.join(DATA_DIR,'planes.png'))

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

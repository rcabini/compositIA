import sys, os 
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import cv2
import tensorflow as tf
import nibabel as nib
import matplotlib.pyplot as plt
from skimage import io
import skimage.transform
import argparse
from slicer.windowed_utils import *
from L3scripts.model_depth_4 import dice_coef
from L3scripts.data_generator import windower
from plots import plotL1, plotL3, plotL1L3

#---------------------------------------------------------------------------------------------

def selectSlices(projection, results, jdata, DATA_DIR, SZ_VOTING_SPACE=[512, 1024]):
	# Calculates the position of the L1 and L3 vertebrae from the distances estimated by the NN. 
	# Input: distance of the window centers to the L1/L3 position; 
	# Output: position of L1 and L3 as index in the voting space 
	# (these indexes will have to be transformed into the original size of the CT).
	voting_L1_coronal = np.zeros(SZ_VOTING_SPACE)
	voting_L1_sagital = np.zeros(SZ_VOTING_SPACE)
	voting_L3_coronal = np.zeros(SZ_VOTING_SPACE)
	voting_L3_sagital = np.zeros(SZ_VOTING_SPACE)
	votes_sagital_L1, votes_sagital_L3, votes_coronal_L1, votes_coronal_L3 = 0, 0, 0, 0
	#Reconstruct votes
	rr = 0
	for filename in jdata.keys():
		imgtype = jdata[filename]["type"]
		center_x = jdata[filename]["center_x"]
		center_y = jdata[filename]["center_y"]
			
		# Convert predictions to points that fit in the map
		pred_L1_x = max(0, min(SZ_VOTING_SPACE[1]-1, round(center_x+results[rr][0]))) #TODO: discard votes outside image
		pred_L3_x = max(0, min(SZ_VOTING_SPACE[1]-1, round(center_x+results[rr][1]))) #TODO: check if x and y are correct
		pred_L1_y = max(0, min(SZ_VOTING_SPACE[0]-1, round(center_y+results[rr][2])))
		pred_L3_y = max(0, min(SZ_VOTING_SPACE[0]-1, round(center_y+results[rr][3])))
		
		print("{} ->  {}: C_l1_s -> ({},{})".format(filename,results[rr],pred_L1_y, pred_L1_x))
			
		if(imgtype == "sagital"):
			voting_L1_sagital[pred_L1_y, pred_L1_x] = voting_L1_sagital[pred_L1_y, pred_L1_x] + 1
			votes_sagital_L1 += 1
			voting_L3_sagital[pred_L3_y, pred_L3_x] = voting_L3_sagital[pred_L3_y, pred_L3_x] + 1
			votes_sagital_L3 += 1
		else:
			voting_L1_coronal[pred_L1_y, pred_L1_x] = voting_L1_coronal[pred_L1_y, pred_L1_x] + 1
			votes_coronal_L1 += 1
			voting_L3_coronal[pred_L3_y, pred_L3_x] = voting_L3_coronal[pred_L3_y, pred_L3_x] + 1
			votes_coronal_L3 += 1
		rr = rr + 1

	scoresL1s = voting_L1_sagital/votes_sagital_L1
	scoresL1c = voting_L1_coronal/votes_coronal_L1
	scoresL3s = voting_L3_sagital/votes_sagital_L3
	scoresL3c = voting_L3_coronal/votes_coronal_L3

	## Save images with votes
	#plt.imsave(DATA_DIR+os.path.sep + 'test_L1_sag.png',scoresL1s,cmap='gray')
	#plt.imsave(DATA_DIR+os.path.sep+'test_L3_sag.png',scoresL3s,cmap='gray')
	#plt.imsave(DATA_DIR+os.path.sep + 'test_L1_cor.png',scoresL1c,cmap='gray')
	#plt.imsave(DATA_DIR+os.path.sep+'test_L3_cor.png',scoresL3c,cmap='gray')
	
	W = projection.shape[1]
	scoresL1s = scoresL1s[:, 0:W]
	scoresL1c = scoresL1c[:, 0:W]
	scoresL3s = scoresL3s[:, 0:W]
	scoresL3c = scoresL3c[:, 0:W]

	scoresL1s_smoothed = cv2.GaussianBlur(scoresL1s, ksize=(0, 0), sigmaX=5, borderType=cv2.BORDER_REPLICATE)
	scoresL1c_smoothed = cv2.GaussianBlur(scoresL1c, ksize=(0, 0), sigmaX=5, borderType=cv2.BORDER_REPLICATE)
	scoresL3s_smoothed = cv2.GaussianBlur(scoresL3s, ksize=(0, 0), sigmaX=5, borderType=cv2.BORDER_REPLICATE)
	scoresL3c_smoothed = cv2.GaussianBlur(scoresL3c, ksize=(0, 0), sigmaX=5, borderType=cv2.BORDER_REPLICATE)

	(pred_L1s, val), row = max(map(lambda x: (max(enumerate(x[1]), key= lambda x: x[1]), x[0]), enumerate(scoresL1s_smoothed)), key=lambda x: x[0][1])
	(col, val), pred_L1c = max(map(lambda x: (max(enumerate(x[1]), key= lambda x: x[1]), x[0]), enumerate(scoresL1c_smoothed)), key=lambda x: x[0][1])
	(pred_L3s, val), row = max(map(lambda x: (max(enumerate(x[1]), key= lambda x: x[1]), x[0]), enumerate(scoresL3s_smoothed)), key=lambda x: x[0][1])
	(col, val), pred_L3c = max(map(lambda x: (max(enumerate(x[1]), key= lambda x: x[1]), x[0]), enumerate(scoresL3c_smoothed)), key=lambda x: x[0][1])

	#plt.imsave(DATA_DIR+os.path.sep+'test_L1_sag_smoothed.png',scoresL1s_smoothed,cmap='gray')
	#plt.imsave(DATA_DIR+os.path.sep+'test_L1_cor_smoothed.png',scoresL1c_smoothed,cmap='gray')
	#plt.imsave(DATA_DIR+os.path.sep+'test_L3_sag_smoothed.png',scoresL3s_smoothed,cmap='gray')
	#plt.imsave(DATA_DIR+os.path.sep+'test_L3_cor_smoothed.png',scoresL3c_smoothed,cmap='gray')

	# save an image with the projection and an overlay with the votes
	plotL1L3(projection[:,:,0], scoresL1s_smoothed, scoresL3s_smoothed, DATA_DIR)
	
	return pred_L1s, pred_L1c, pred_L3s, pred_L3c

#---------------------------------------------------------------------------------------------

def slicer(volume, spacing, folder, DATA_DIR, MODEL_FILE):
	# Extracts L1 and L3 slices from the whole CT volume. 
	# Input: CT volume; 
	# Output: L1 and L3 slices.
	print("==== Selecting slices ====")
	## Save projections
	sagital, coronal = save_projections(folder, volume, spacing)
	## INITIALIZATION
	tf.keras.backend.clear_session()
	# Prepare and save sliding windows
	CATALOGUE, TEST_SET, jdata = createWindows(folder, ((0,0), (0,0), (0,0), (0,0)))
	# Predicts the values for the windows in the test set
	print('Loading model ... ')
	model = tf.keras.models.load_model(MODEL_FILE)
	print(' ... Loaded!')
	results = model.predict(TEST_SET, batch_size=24, workers=4, verbose=1)
	# Select slices
	pred_L1s, pred_L1c, pred_L3s, pred_L3c = selectSlices(sagital, results, jdata, DATA_DIR)
	#TODO: save an image with the projection and an overlay with the votes
	print("L1s: "+str(pred_L1s)+"\tL1c: "+str(pred_L1c)+"\tL3s: "+str(pred_L3s)+"\tL3c: "+str(pred_L3c))

	# using sagital values. not sure why coronal does not work
	L1idx = round(pred_L1s*spacing[0]/spacing[2])
	print("L1 idx: "+str(L1idx))
	L1_slice = volume[:,:,L1idx]

	L3idx = round(pred_L3s*spacing[0]/spacing[2])
	print("L3 idx: "+str(L3idx))
	L3_slice = volume[:,:,L3idx]

	plt.imsave(DATA_DIR+os.path.sep+'L3slice.png',L3_slice,cmap='gray')
	plt.imsave(DATA_DIR+os.path.sep+'L1slice.png',np.fliplr(L1_slice),cmap='gray')

	return L1_slice, L3_slice

#---------------------------------------------------------------------------------------------

def L3segmentation(img, DATA_DIR, MODEL_L3):
	# U-net to segment the CT slice at the L3 spinal level in the following regions: 
	# visceral adipose tissue (VAT), subcutaneous adipose tissue (SAT), skeletal muscle area (SMA).
	print("==== L3 Segmentation ====")
	rgb_img = np.zeros((img.shape[0], img.shape[1], 3))
	rgb_img[:,:,0] = windower(img, -1024, 2048)
	rgb_img[:,:,1] = windower(img, -190, -30)
	rgb_img[:,:,2] = windower(img, 40, 100)

	im_width, im_height = img.shape #(512, 512)
	input_size = (256, 256)

	print('Loading model ... ')
	model = tf.keras.models.load_model(MODEL_L3, custom_objects={"dice_coef": dice_coef })
	print(' ... Loaded!')
	X = np.zeros((1, im_height, im_width, 3), dtype=np.float32)
	X[0,:,:,:] = rgb_img / 255
	X = skimage.transform.resize(X, (1, input_size[0], input_size[1], 3), mode='constant',cval=0,anti_aliasing=True,preserve_range=True,order=3)

	results = model.predict(X, verbose=1)
	results = skimage.transform.resize(results, (1, im_height, im_width, results.shape[-1]),mode='constant',cval=0,anti_aliasing=True,preserve_range=True,order=0)
	res_seg = np.round(results)[0,:,:,:]

	#plt.imsave(DATA_DIR+'/VAT.png', res_seg[:,:,1], cmap='gray')
	#plt.imsave(DATA_DIR+'/SAT.png', res_seg[:,:,2], cmap='gray')
	#plt.imsave(DATA_DIR+'/SMA.png', res_seg[:,:,3], cmap='gray')

	plotL3(img, res_seg, DATA_DIR)
	vat, sat, sma = res_seg[:,:,1].astype(int), res_seg[:,:,2].astype(int), res_seg[:,:,3].astype(int)
	return vat, sat, sma

#---------------------------------------------------------------------------------------------

def L1segmentation(img, DATA_DIR, MODEL_L1):
	# U-net to segment the CT slice at the L1 spinal level in the following regions: 
	# L1 Spungiosa area and L1 Cortical area.
	print("==== L1 Segmentation ====")
	im_width, im_height = img.shape #(512, 512)

	print('Loading model ... ')
	model = tf.keras.models.load_model(MODEL_L1)
	print(' ... Loaded!')
	X = np.zeros((1, im_height, im_width, 1), dtype=np.float32)
	X[0,:,:,0] = windower(img, img.min(), img.max()) / 255

	results = model.predict(X, verbose=1)
	res_seg = np.round(results)[0,:,:,:]

	plotL1(img, res_seg, DATA_DIR)
	#plt.imsave(DATA_DIR+'/pred_L1.png', res_seg, cmap='gray')
	cort, spun = res_seg[:,:,0].astype(int), res_seg[:,:,1].astype(int)
	return cort, spun

#---------------------------------------------------------------------------------------------

def main(args):
	## Get the file to process
	fn = args.image_path
	if(not os.path.exists(fn)):
		raise RuntimeError("File {} not found!".format(fn))
	folder = os.path.join(os.path.dirname(fn),os.path.basename(fn[0:-7]))

	## SETTINGS
	DATA_DIR = os.path.join(folder,'windows')
	MODEL_SLICE    = './slicer/weights/VGG_19.hdf5' #'/var/www/html/compositia/proc/VGG_19.hdf5'
	MODEL_L1       = './L1scripts/weights/unet_DB1_mcL1.hdf5' #'/var/www/html/compositia/proc/unet_DB1_mcL1.hdf5'
	MODEL_L3       = './L3scripts/weights/unet_DB1_multi.hdf5' #'/var/www/html/compositia/proc/unet_DB1_multi_256.hdf5'
	OUT_SUB_FOLDER = 'windows'
	
	output_folder = folder +  os.path.sep + OUT_SUB_FOLDER
	if not os.path.exists(folder):
		os.mkdir(folder)
	if not os.path.exists(output_folder):
		os.mkdir(output_folder)

	## Read the nifti image
	(volume, spacing) = readData(fn)
	print(volume.shape)
	
	## predict L1 and L3 slices
	L1_slice, L3_slice = slicer(volume, spacing, folder, DATA_DIR, MODEL_SLICE)
	## L1 segmentation
	cort, L1_mask = L1segmentation(np.fliplr(L1_slice), DATA_DIR, MODEL_L1)
	## L3 segmentation
	VAT_mask, SAT_mask, SMA_mask = L3segmentation(L3_slice, DATA_DIR, MODEL_L3)

	## Compute scores
	# calculate the scores and write them as a list 
	# of numbers separated by , in DATA_DIR+'/'+'scores.txt'
	areaL3SAT = np.sum(np.sum(SAT_mask))*(spacing[0]*spacing[0])*0.01
	densitystdL3SAT = np.std(L3_slice[SAT_mask].flatten())
	areaL3VAT = np.sum(np.sum(VAT_mask))*(spacing[0]*spacing[0])*0.01
	areaL3SMA = np.sum(np.sum(SMA_mask))*(spacing[0]*spacing[0])*0.01

	L1_mask = np.fliplr(L1_mask)
	L1_slice_spungiosa = L1_slice[L1_mask]
	densitystdL1spungiosa = np.std(L1_slice_spungiosa.flatten())
	densityavgL1spungiosa = np.mean(L1_slice_spungiosa.flatten())
	areaL1spungiosa = np.sum(np.sum(L1_mask))*(spacing[0]*spacing[0])*0.01

	strScores = str(round(densityavgL1spungiosa, 1))+" "+ \
	            str(round(densitystdL1spungiosa, 1))+" "+ \
			    str(round(areaL1spungiosa, 1))+" "+ \
			    str(round(areaL3SAT, 1))+" "+ \
			    str(round(areaL3SMA,1))+" "+ \
			    str(round(areaL3VAT,1))+" "+ \
			    str(round(densitystdL3SAT, 1))
	print(DATA_DIR+'/'+'scores.txt')
	f = open(DATA_DIR+'/'+'scores.txt', "a")
	f.write(strScores)
	f.close()


if __name__=="__main__":

	"""Read command line arguments"""
	parser = argparse.ArgumentParser()
	parser.add_argument("image_path", help='enter image path')
	args = parser.parse_args()
	main(args)

	#fn = '../DataNIFTI/Images/BC041/data.nii.gz'
	#main(fn)

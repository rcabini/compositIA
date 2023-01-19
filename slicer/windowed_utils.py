import os
from turtle import window_height
from unicodedata import name
import numpy as np
import nibabel as nib
import skimage.io
import skimage.draw
import skimage.color
import skimage.transform
import json
import argparse
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import img_to_array, load_img


WINDOW_STRIDE = (64,64)
WINDOW_SIZE = (224,224)
PREVIEW_SUBFOLDER = "L1L3WindowedPrep"

# WINDOWING (center, width)
BONE_WINDOW_C0 = (1000, 2000)  # optimal values for bones
BONE_WINDOW_C1 = (400, 500)  # optimal values for bones but more saturated
BONE_WINDOW_C2 = (800, 1900)  # Value optimized for bones

OUT_SUB_FOLDER = 'windows'

def readData(fn):
	#Read the data.nii.gz and segmentation.nii.gz file in the patient directory and return the volumes & spacing
	# return [volume, segmentation, spacing]
	data_file = fn
	if(not os.path.exists(data_file)):
		raise RuntimeError("Could not Find file {} ".format(data_file))
	
	medical_image = nib.load(data_file)
	medical_image = nib.as_closest_canonical(medical_image)
	header = medical_image.header
	volume = medical_image.get_fdata()  # Convert to nparray
	spacing = [header['pixdim'][1],header['pixdim'][2],header['pixdim'][3]] #there is an offset of 1
	#Check if the volume is 512x512, if not raise exception
	if volume.shape[0] != 512 or volume.shape[1] != 512:
		raise Exception("{} has non standard resolution!".format(fn))
	return (volume, spacing)

def resize(projection, spacing):
	# resize prjected images to an isotropic pixel scaling
	#Step 1 calculate the cuurent spatial dimensions
	orig_size = projection.shape
	curr_dims = (spacing[0] * orig_size[0], spacing[2] * orig_size[1])
	scaled_size = (orig_size[0], int(round(curr_dims[1]/spacing[0])))
	return skimage.transform.resize(projection, scaled_size)
	
def window_image(image, win):
	#Apply a window to an image
	img_min = win[0] - win[1] // 2
	img_max = win[0] + win[1] // 2
	#Now convert to uint8 and rerange
	value = 0.0
	r = img_max-img_min
	result = np.zeros(image.shape, dtype=np.ubyte)
	for row in range(image.shape[0]):
		for col in range(image.shape[1]):
			value = image[row,col]
			if value < img_min:
				value = img_min
			if value > img_max:
				value = img_max
			result[row,col] = np.ubyte((value-img_min)/(r)*255.0+0.5)
	return result

def extract_images(volume, axis, spacing):
	#extract the 3 windowed channels and resize the image to be isotropic
	projection = np.max(volume,axis)
	projection = resize(projection, spacing)
	ch0 = window_image(projection, BONE_WINDOW_C0)
	ch1 = window_image(projection, BONE_WINDOW_C1)
	ch2 = window_image(projection, BONE_WINDOW_C2)
	result = np.zeros((projection.shape[0],projection.shape[1],3), dtype=np.ubyte)
	result[:,:,0] = ch0
	result[:,:,1] = ch1
	result[:,:,2] = ch2
	return result

def draw_center_images(centers, dstFolder):
	# Intermediate step to dar the L1 and L3 centers on a preview image for verification
	buff_s = skimage.io.imread(dstFolder + os.path.sep + "sagital.png")
	buff_c = skimage.io.imread(dstFolder + os.path.sep + "coronal.png")
	buff_s = skimage.color.gray2rgb(buff_s)
	buff_c = skimage.color.gray2rgb(buff_c)
	#L1 on sagital
	rr,cc = skimage.draw.circle_perimeter(centers[0][0],centers[0][1],5,shape=buff_s.shape)
	buff_s[rr,cc,:] = [255,0,0]
	# L3 on sagital
	rr,cc = skimage.draw.circle_perimeter(centers[1][0],centers[1][1],5,shape=buff_s.shape)
	buff_s[rr,cc,:] = [0,0,255]
	#save
	skimage.io.imsave(dstFolder + os.path.sep + "sagital_L1_L3.png", buff_s)
	#L1 on coronal
	rr,cc = skimage.draw.circle_perimeter(centers[2][0],centers[2][1],5,shape=buff_c.shape)
	buff_c[rr,cc,:] = [255,0,0]
	# L3 on coronal
	rr, cc = skimage.draw.circle_perimeter(centers[3][0],centers[3][1],5,shape=buff_c.shape)
	buff_c[rr,cc,:] = [0,0,255]
	#save
	skimage.io.imsave(dstFolder + os.path.sep + "coronal_L1_L3.png", buff_c)
	
def save_projections(folder, volume, spacing):
	dstFolder = folder + os.path.sep + PREVIEW_SUBFOLDER
	if(not os.path.exists(dstFolder)):
		os.makedirs(dstFolder)
	#extract sagital & coronal projections, scaled and save a preview in the input folder
	sagital = extract_images(volume, axis=0, spacing=spacing) #Sagital
	sagital = np.flipud(sagital)
	skimage.io.imsave(dstFolder + os.path.sep + "sagital.png", sagital[:,:,0])
	skimage.io.imsave(dstFolder + os.path.sep + "sagital_all.png", sagital)
	coronal = extract_images(volume, axis=1, spacing=spacing) #Coronal
	skimage.io.imsave(dstFolder + os.path.sep + "coronal.png", coronal[:,:,0])
	skimage.io.imsave(dstFolder + os.path.sep + "coronal_all.png", coronal)
	return sagital, coronal

def createWindows(folder, centers):
	# Creates the slidinng windows over the dataset and retuns them with their shift from center in two arrays of objects:
	# uses the centers estabished in the steps above
	# return dagital, coronal  --> sagital = {"L1_dx": dx, "L1_dy": dy, "data": image}
	# where dx and dy are the distance of the window center to the L1/3 center, data is the 227x227 rgb imgagee data
	output_folder = folder +  os.path.sep + OUT_SUB_FOLDER
	if not os.path.exists(output_folder):
		os.mkdir(output_folder)
	dstFolder = folder + os.path.sep + PREVIEW_SUBFOLDER
	sagital = skimage.io.imread(dstFolder + os.path.sep + "sagital_all.png")
	coronal = skimage.io.imread(dstFolder + os.path.sep + "coronal_all.png")
	# sizes are in width, heigth
	width = sagital.shape[1]
	heigth = sagital.shape[0]
	window_width = WINDOW_SIZE[0]
	window_height = WINDOW_SIZE[1]
	sagital_center_L1 = centers[0]  # As in [row, col]
	sagital_center_L3 = centers[1]
	coronal_center_L1 = centers[2]
	coronal_center_L3 = centers[3]
	sag_wins = []
	cor_wins = []
	# position of window top left corner as in row, col
	curr_row = 0
	curr_col = 0
	# iterate over a row shifting the window by 32 cols
	while curr_row + window_height < heigth:
		# shift along columns
		while curr_col + window_width < width:
			center = (curr_row+window_height/2, curr_col +
					  window_width/2)  # as in row, col
			block = {
				"data": sagital[
					curr_row:curr_row+window_height,
					curr_col:curr_col+window_width,
					:],
				"L1_dx": sagital_center_L1[1] - center[1],  # diff of col
				"L1_dy": sagital_center_L1[0] - center[0],  # diff of row
				"L3_dx": sagital_center_L3[1] - center[1],
				"L3_dy": sagital_center_L3[0] - center[0],
				"center_x": center[1],
				"center_y": center[0]
			}
			sag_wins.append(block)
			block2 = {
				"data": coronal[
					curr_row:curr_row+window_height,
					curr_col:curr_col+window_width,
					:],
				"L1_dx": coronal_center_L1[1] - center[1],
				"L1_dy": coronal_center_L1[0] - center[0],
				"L3_dx": coronal_center_L3[1] - center[1],
				"L3_dy": coronal_center_L3[0] - center[0],
				"center_x": center[1],
				"center_y": center[0]
			}
			cor_wins.append(block2)
			curr_col += WINDOW_STRIDE[0]
		# reset window to row init and down by stride
		curr_row += WINDOW_STRIDE[0]
		curr_col = 0
	numWindows=0
	json_data = {}
	IMG_SET = np.zeros((999, 224, 224, 3), dtype=np.float32)
	for window in sag_wins:
		name = "{:06d}.png".format(numWindows)
		x_img = img_to_array(window["data"]).astype(np.float32)
		x_img = x_img/255.0
		IMG_SET[numWindows, ..., :] = x_img
		#skimage.io.imsave(output_folder + os.path.sep + name,
		#				  window["data"], check_contrast=False)
		json_data[name] = {
			"L1_dx": window["L1_dx"],
			"L3_dx": window["L3_dx"],
			"L1_dy": window["L1_dy"],
			"L3_dy": window["L3_dy"],
			"center_x": window["center_x"],
			"center_y": window["center_y"],
			"type": "sagital",
			"patient": folder
		}
		numWindows += 1
	#for window in cor_wins:
	#	name = "{:06d}.png".format(numWindows)
	#	skimage.io.imsave(output_folder + os.path.sep + name,
	#					  window["data"], check_contrast=False)
	#	json_data[name] = {
	#		"L1_dx": window["L1_dx"],
	#		"L3_dx": window["L3_dx"],
	#		"L1_dy": window["L1_dy"],
	#		"L3_dy": window["L3_dy"],
	#		"center_x": window["center_x"],
	#		"center_y": window["center_y"],
	#		"type": "coronal",
	#		"patient": folder
	#	}
	#	numWindows += 1
	#write cyclic updates
	with open(output_folder + os.path.sep + "windows.json", 'w') as outfile:
		json.dump(json_data, outfile, indent=4)
	IMG_SET = IMG_SET[0:numWindows,...]
	return output_folder + os.path.sep + "windows.json", IMG_SET, json_data
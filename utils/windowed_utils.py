import os
import numpy as np
import nibabel as nib
import skimage.io
import skimage.draw
import skimage.color
import skimage.transform
import json
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt

BONE_WINDOW_C0 = (1000, 2000)  # optimal values for bones
BONE_WINDOW_C1 = (400, 500)  # optimal values for bones but more saturated
BONE_WINDOW_C2 = (800, 1900)  # Value optimized for bones

def readData(fn, order):
    # Read the data.nii.gz and segmentation.nii.gz file in the patient directory and return the volumes & spacing
    # return [volume, segmentation, spacing]
    data_file = fn
    if(not os.path.exists(data_file)):
        raise RuntimeError("Could not Find file {} ".format(data_file))

    medical_image = nib.load(data_file)
    medical_image = nib.as_closest_canonical(medical_image)
    header = medical_image.header
    volume = medical_image.get_fdata()  # Convert to nparray
    spacing = [header['pixdim'][1],header['pixdim'][2],header['pixdim'][3]] #there is an offset of 1
    #Check if the volume is 512x512, if not resampling
    if volume.shape[0] != 512 or volume.shape[1] != 512:
        shape_old = volume.shape
        spacing = [spacing[0]/512*volume.shape[0], spacing[1]/512*volume.shape[1], spacing[2]]
        volume = skimage.transform.resize(volume, (512, 512, volume.shape[-1]), anti_aliasing=True, order=order)
        print("{} has non standard resolution! Rehsaped from {} to {}".format(fn, shape_old, volume.shape))
        #Check if the volume is 512x512, if not raise exception
        if volume.shape[0] != 512 or volume.shape[1] != 512:
            raise Exception("{} has non standard resolution!".format(fn))
    return (volume, spacing)

def resize(projection, spacing):
    # resize prjected images to an isotropic pixel scaling
    orig_size = projection.shape
    curr_dims = (spacing[0] * orig_size[0], spacing[2] * orig_size[1])
    scaled_size = (orig_size[0], int(round(curr_dims[1]/spacing[0]))) #spacing[0]
    resized = skimage.transform.resize(projection, scaled_size)
    return resized

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
	

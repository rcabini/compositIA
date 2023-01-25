#Step 1: Load libraries for the U-net Model
import numpy as np 
import os
from glob import glob
import matplotlib.pyplot as plt
import skimage.io as io
import skimage.transform as trans
import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as keras
import tensorflow as tf

#Step 2: Import the U-net model
from model_depth_4 import *

img_size = (512,512) #(256,256)
n_class=3
#Create Groundtruth with 5 planes:[Red Lesions(0), Bright Lesions(1), background (2) ]

#Step 3:Define functions for pre-processing data
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import skimage.io as io
import skimage.transform as trans
import scipy.misc as sc

#---------------------------------------------------------------------------

def adjustData(img,mask,flag_multi_class,num_class):
    if(flag_multi_class):
        img = img / 255
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        mask = mask / 255 * (num_class-1)
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            new_mask[mask == i,i] = 1
        new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1],new_mask.shape[2],new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1],new_mask.shape[2]))
        mask = new_mask
    elif(np.max(img) > 1):
        img = img / 255
        mask = mask /255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img,mask)

#---------------------------------------------------------------------------

def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = True,n_class = n_class,save_to_dir = None,target_size = img_size,seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        #data_format = "channels_last",
        class_mode=None,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        #data_format = "channels_last",
        class_mode=None,#"categorical",
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask,flag_multi_class,num_class=n_class)
        yield (img,mask)
        
#---------------------------------------------------------------------------

def valGenerator(batch_size,val_path,image_folder,mask_folder,image_color_mode = "grayscale",
                 mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                 flag_multi_class = True,n_class = n_class,save_to_dir = None,target_size = img_size,seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator()
    mask_datagen = ImageDataGenerator()
    image_generator = image_datagen.flow_from_directory(
        val_path,
        classes = [image_folder],
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        #data_format = "channels_last",
        class_mode=None,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        val_path,
        classes = [mask_folder],
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        #data_format = "channels_last",
        class_mode=None,
        seed = seed)
    val_generator = zip(image_generator, mask_generator)
    for (img,mask) in val_generator:
        img,mask = adjustData(img,mask,flag_multi_class,num_class=n_class)
        yield (img,mask)

#---------------------------------------------------------------------------
        
data_gen_args = dict(#rescale=1./255, #intensity normalization
                     rotation_range=0.5,
                     width_shift_range=0.3,
                     height_shift_range=0.3,
                     shear_range=0.3,
                     zoom_range=0.3,
                     horizontal_flip=True,
                     vertical_flip=True,
                     fill_mode='constant',
                     cval=0) #'constant'
                     #validation_split=0.2)

#---------------------------------------------------------------------------
                     
def plot_history(history, results_path):
    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    plt.semilogy(history.history['loss'], label='Train')
    plt.semilogy(history.history['val_loss'], label='Validation')
    plt.grid(True)
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.subplot(1,2,2)
    plt.semilogy(history.history['dice_coef'], label='Train')
    plt.semilogy(history.history['val_dice_coef'], label='Validation')
    plt.grid(True)
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('dice_coef')
    plt.savefig(os.path.join(results_path, 'history.png'), bbox_inches = "tight", dpi=200)

#-------------------------------------------------------------------------

def main():

    #PATH = '../DATA_input/'
    TRAIN_PATH = '../../DATA_input/L1/train/'
    VALID_PATH = '../../DATA_input/L1/validation/'
    CHECK_PATH = "./weights/"

    if not os.path.exists(CHECK_PATH):
        os.makedirs(CHECK_PATH)
    #if not os.path.exists(PATH+'aug'):
    #    os.makedirs(PATH+'aug')  
    
    N_TRAIN = len(glob(TRAIN_PATH+'image/*.png'))
    N_VALID = len(glob(VALID_PATH+'label/*.png'))
    BATCH_SIZE = 1
    train_gen = trainGenerator(BATCH_SIZE,TRAIN_PATH,'image','label',data_gen_args, save_to_dir = False)
    val_gen = valGenerator(BATCH_SIZE,VALID_PATH,'image','label', save_to_dir = False)

    #Initialize the model. Train from scratch!
    model = unet(input_size=(img_size[0],img_size[1],1), n_class=n_class)
    model.summary()
    
    #Fit the u-net model
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join(CHECK_PATH, 'unet_DB1_mcL1.hdf5'),
                                                          monitor='val_dice_coef',
                                                          verbose=1,
                                                          save_best_only=True,
                                                          mode = 'max')
    
    STEPS_PER_EPOCH = N_TRAIN // BATCH_SIZE
    VALIDATION_STEPS = N_VALID // BATCH_SIZE
    history = model.fit(train_gen,
              validation_data=val_gen,
              epochs=300,
              steps_per_epoch=STEPS_PER_EPOCH,
              validation_steps=VALIDATION_STEPS,
              verbose=1, 
              callbacks=[model_checkpoint])
              
    plot_history(history, CHECK_PATH)

if __name__=="__main__":
    main()

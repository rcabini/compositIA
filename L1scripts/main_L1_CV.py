# SPDX-License-Identifier: EUPL-1.2
# Copyright 2024 dp-lab Università della Svizzera Italiana

import os
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
from model_L1 import *

#---------------------------------------------------------------------------

def train_generator(data_frame, batch_size, aug_dict,
                    target_size,
                    image_color_mode="grayscale",
                    mask_color_mode="rgb",
                    image_save_prefix="image",
                    mask_save_prefix="mask",
                    flag_multi_class = True,
                    save_to_dir=None,
                    seed=1):
    '''
    can generate image and mask at the same time use the same seed for
    image_datagen and mask_datagen to ensure the transformation for image
    and mask is the same if you want to visualize the results of generator,
    set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    
    image_generator = image_datagen.flow_from_dataframe(
        data_frame,
        x_col = "filename",
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)

    mask_generator = mask_datagen.flow_from_dataframe(
        data_frame,
        x_col = "mask",
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)

    train_gen = zip(image_generator, mask_generator)
    
    for (img, mask) in train_gen:
        yield (img,mask)

#---------------------------------------------------------------------------
        
train_generator_args = dict(rescale=1./255., #intensity normalization
                            rotation_range=0.5,
                            width_shift_range=0.3,
                            height_shift_range=0.3,
                            shear_range=0.3,
                            zoom_range=0.3,
                            horizontal_flip=True,
                            vertical_flip=True,
                            fill_mode='constant',
                            cval=0)

#---------------------------------------------------------------------------
                     
def plot_history(history, results_path, fname):
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
    plt.savefig(os.path.join(results_path, fname), bbox_inches = "tight", dpi=200)

#-------------------------------------------------------------------------

def main(args):
    
    img_size = (512,512)
    n_class=3
    os.makedirs(args.weights_folder, exist_ok=True)
    BATCH_SIZE = 1
    PVAL = 0.2 # percentage of validation

    ## Training with K-fold cross validation
    images_file_paths = glob(os.path.join(args.data_folder,'image/*.png'))
    labels_file_paths = glob(os.path.join(args.data_folder,'label/*.png'))
    Folds = pd.read_csv(args.ktxt, sep=" ", header=None)
    K_FOLDS = len(Folds)
    
    df = pd.DataFrame(data={"filename": images_file_paths, 'mask' : labels_file_paths})
    
    for k, TEST_NAME in Folds.iterrows():
        TEST_NAME = TEST_NAME.dropna().values
        print("Fold {}/{}".format(k+1, K_FOLDS))
        df_names = [fff for fff in df['filename'] if os.path.basename(fff).split('.png')[0] not in TEST_NAME]
        train_df = df[df['filename'].isin(df_names)]
        #train validation split
        train_x, val_x, train_y, val_y = train_test_split(train_df['filename'], train_df['mask'], test_size=PVAL, random_state=1)
        
        train_data_frame=pd.DataFrame(columns=['filename', 'mask'])
        train_data_frame['filename']=train_x
        train_data_frame['mask']=train_y
        
        val_data_frame=pd.DataFrame(columns=['filename', 'mask'])
        val_data_frame['filename']=val_x
        val_data_frame['mask']=val_y
        
        train_gen = train_generator(train_data_frame, BATCH_SIZE, train_generator_args, img_size)
        val_gen = train_generator(val_data_frame, BATCH_SIZE, dict(rescale=1./255.), img_size)
        # Define the model
        model = unet(input_size=(img_size[0],img_size[1],1), n_class=n_class)
        model.summary()
        #Fit the u-net model
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join(args.weights_folder, 'unet_L1_k{}.hdf5'.format(k)),
                                                              monitor='val_dice_coef',
                                                              verbose=1,
                                                              save_best_only=True,
                                                              mode = 'max')

        N_TRAIN = len(train_data_frame)
        N_VALID = len(val_data_frame)
        print('Number of training samples: ', N_TRAIN, 'Number of validation samples: ', N_VALID)
        STEPS_PER_EPOCH = N_TRAIN // BATCH_SIZE
        VALIDATION_STEPS = N_VALID // BATCH_SIZE
        
        history = model.fit(train_gen,
                  validation_data=val_gen,
                  epochs=100,
                  steps_per_epoch=STEPS_PER_EPOCH,
                  validation_steps=VALIDATION_STEPS,
                  verbose=1, 
                  callbacks=[model_checkpoint])
                  
        plot_history(history, args.weights_folder, 'history_k{}.png'.format(k))
        tf.keras.backend.clear_session()
        
    k = 'ALL'
    train_df = df
    #train validation split
    train_x, val_x, train_y, val_y = train_test_split(train_df['filename'], train_df['mask'], test_size=PVAL, random_state=1)
    
    train_data_frame=pd.DataFrame(columns=['filename', 'mask'])
    train_data_frame['filename']=train_x
    train_data_frame['mask']=train_y
    val_data_frame=pd.DataFrame(columns=['filename', 'mask'])
    val_data_frame['filename']=val_x
    val_data_frame['mask']=val_y
    
    train_gen = train_generator(train_data_frame, BATCH_SIZE, train_generator_args, img_size)
    val_gen = train_generator(val_data_frame, BATCH_SIZE, dict(rescale=1./255.), img_size)
    
    # Define the model
    model = unet(input_size=(img_size[0],img_size[1],1), n_class=n_class)
    model.summary()

    #Fit the u-net model
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join(args.weights_folder, 'unet_L1_k{}.hdf5'.format(k)),
                                                          monitor='val_dice_coef',
                                                          verbose=1,
                                                          save_best_only=True,
                                                          mode = 'max')

    N_TRAIN = len(train_data_frame)
    N_VALID = len(val_data_frame)
    print('Number of training samples: ', N_TRAIN, 'Number of validation samples: ', N_VALID)
    STEPS_PER_EPOCH = N_TRAIN // BATCH_SIZE
    VALIDATION_STEPS = N_VALID // BATCH_SIZE
    
    history = model.fit(train_gen,
              validation_data=val_gen,
              epochs=100,
              steps_per_epoch=STEPS_PER_EPOCH,
              validation_steps=VALIDATION_STEPS,
              verbose=1, 
              callbacks=[model_checkpoint])
              
    plot_history(history, args.weights_folder, 'history_k{}.png'.format(k))
    tf.keras.backend.clear_session()
        
if __name__=="__main__":
    """Read command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", help='Path to dataset')
    parser.add_argument("--ktxt", help='Path to k-splits txt file')
    parser.add_argument("--weights_folder", help='Path to output weights')
    args = parser.parse_args()
    main(args)

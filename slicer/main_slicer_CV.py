# SPDX-License-Identifier: EUPL-1.2
# Copyright 2024 dp-lab Università della Svizzera Italiana

import os, json
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam # Optimizer 
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import argparse
from multiUnet import *

#---------------------------------------------------------------------------

def train_generator(data_frame, batch_size, aug_dict,
                    target_size,
                    image_color_mode="rgb",
                    mask_color_mode="grayscale",
                    image_save_prefix="image",
                    mask_save_prefix="mask",
                    flag_multi_class=True,
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
        x_col = "image",
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)

    mask_generator = mask_datagen.flow_from_dataframe(
        data_frame,
        x_col = "label",
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)

    train_gen = zip(image_generator, mask_generator)
    
    for (img, mask) in train_gen:
        #img, mask = adjust_data(img, mask)
        yield (img,mask)

#---------------------------------------------------------------------------
        
train_generator_args = dict(rescale=1./255.)
             
#---------------------------------------------------------------------------
                     
def plot_history(history, results_path, k):
    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    plt.semilogy(history.history['loss'], label='Train')
    plt.semilogy(history.history['val_loss'], label='Validation')
    plt.grid(True)
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.subplot(1,2,2)
    plt.semilogy(history.history['mean_absolute_error'], label='Train')
    plt.semilogy(history.history['val_mean_absolute_error'], label='Validation')
    plt.grid(True)
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('MAE')
    plt.savefig(os.path.join(results_path, 'history_kfold{}.png'.format(k)), bbox_inches = "tight", dpi=200)
    
#---------------------------------------------------------------------------

def load_dataset(exceldir, k, WINDOW_SIZE, BATCH_SIZE):
    # Load X and Y vectors
    df = pd.read_excel(exceldir)
    df = df[[os.path.isfile(i) for i in df['image']]]
    df = df[df['k']!=k][["image", "label"]]

    #to select the augmented files
    all_base_names = np.unique([ff.split("/")[-1].split(".png")[0] for ff in df["image"] if "_aug" not in ff])
    train_nam, test_nam = train_test_split(all_base_names, test_size=0.1, random_state=1)
    train_nam = [e for s in train_nam for e in [s,s+'_aug']]
    test_nam = [e for s in test_nam for e in [s,s+'_aug']]
    train_df = df[df['image'].str.contains('|'.join(r"\b{}\b".format(x) for x in train_nam))]
    test_df = df[df['image'].str.contains('|'.join(r"\b{}\b".format(x) for x in test_nam))]
    
    train_gen = train_generator(train_df, BATCH_SIZE, dict(rescale=1./255.), WINDOW_SIZE)
    test_gen = train_generator(test_df, BATCH_SIZE, dict(rescale=1./255.), WINDOW_SIZE)

    SET_SIZE = (train_df.shape[0], test_df.shape[0])
    print("****************************")
    print(SET_SIZE)
    print("****************************")
    return train_gen, test_gen, SET_SIZE

#---------------------------------------------------------------------------

def train_model(model, train_gen, val_gen, BATCH_SIZE, WEIGHTS_DIR, DATASET_SIZE, epochs, k):
    #Add a callback for saving model
    model_checkpoint = ModelCheckpoint(os.path.join(WEIGHTS_DIR, 'multires_kfold{}.hdf5'.format(k)), monitor='val_mean_absolute_error',verbose=1, save_best_only=True)
    model.compile(loss="mean_squared_error", optimizer=Adam(learning_rate = 1e-3), metrics=["mean_absolute_error"])
    
    STEPS_PER_EPOCH = DATASET_SIZE[0] // BATCH_SIZE
    VALIDATION_STEPS = DATASET_SIZE[1] // BATCH_SIZE
    history=model.fit(
                      train_gen,
                      epochs = epochs,
                      steps_per_epoch=STEPS_PER_EPOCH,
                      validation_steps=VALIDATION_STEPS,
                      batch_size = BATCH_SIZE,
                      verbose = 1,
                      validation_data = val_gen,
                      callbacks = [model_checkpoint]
                     )
    
    return model, history
    
#---------------------------------------------------------------------------

def main(args):    
    tf.keras.backend.clear_session()
    DATA_PATH = os.path.join(args.data_folder, "dataset.xlsx")
    WEIGHTS_DIR = args.weights_folder
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    Folds = pd.read_csv(args.ktxt, sep=" ", header=None)
    K_FOLDS = len(Folds)
        
    BATCH_SIZE = 4
    EPOCHS = 800
    WINDOW_SIZE = (128, 256, 3) 
    
    for k, TEST_NAME in Folds.iterrows():
        print("Fold {}/{}".format(k+1, K_FOLDS))
        TEST_NAME = TEST_NAME.dropna().values
    
        train_gen, val_gen, DATASET_SIZE = load_dataset(DATA_PATH, k, (WINDOW_SIZE[0],WINDOW_SIZE[1]), BATCH_SIZE)
        model = MultiResUnet(height=WINDOW_SIZE[0], width=WINDOW_SIZE[1], n_channels=WINDOW_SIZE[2])
        model.summary()
        #for fine-learning
        #model = tf.keras.models.load_model(WEIGHTS_DIR +os.path.sep+'Unet_kfold{}.hdf5'.format(k))
        #Fit the model
        model, history = train_model(model, train_gen, val_gen, BATCH_SIZE, WEIGHTS_DIR, DATASET_SIZE, EPOCHS, k)         
        plot_history(history, WEIGHTS_DIR, k)
        tf.keras.backend.clear_session()
    
    # Retrain the model on the complete dataset
    k='ALL'
    train_gen, val_gen, DATASET_SIZE = load_dataset(DATA_PATH, k, (WINDOW_SIZE[0],WINDOW_SIZE[1]), BATCH_SIZE)
    model = MultiResUnet(height=WINDOW_SIZE[0], width=WINDOW_SIZE[1], n_channels=WINDOW_SIZE[2])
    model.summary()
    model, history = train_model(model, train_gen, val_gen, BATCH_SIZE, WEIGHTS_DIR, DATASET_SIZE, EPOCHS, k)         
    plot_history(history, WEIGHTS_DIR, k)
    tf.keras.backend.clear_session()
    
if __name__=="__main__":
    """Read command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", help='Path to dataset')
    parser.add_argument("--ktxt", help='Path to k-splits txt file')
    parser.add_argument("--weights_folder", help='Path to output weights')
    args = parser.parse_args()
    main(args)
    
    

import numpy as np 
import os
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from model_L3 import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from sklearn.model_selection import train_test_split

#---------------------------------------------------------------------------

def adjustData(img,mask,flag_multi_class,num_class):
    if(flag_multi_class):
        img = img / 255.
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        mask = mask / 255. * (num_class-1)
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            new_mask[mask == i,i] = 1
        new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1],new_mask.shape[2],new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1],new_mask.shape[2]))
        mask = new_mask
    elif(np.max(img) > 1):
        img = img / 255.
        mask = mask / 255.
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img,mask)

#---------------------------------------------------------------------------

def train_generator(data_frame, batch_size, aug_dict,
                    target_size,
                    n_class,
                    image_color_mode="rgb",
                    mask_color_mode="grayscale",
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
        img,mask = adjustData(img,mask,flag_multi_class,num_class=n_class)
        yield (img,mask)

#---------------------------------------------------------------------------
        
train_generator_args = dict(rotation_range=0.5,
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

def main():
    
    img_size = (512,512)
    n_class=4
    
    DATA_PATH = './DATA/'
    CHECK_PATH = './weights/'
    os.makedirs(CHECK_PATH, exist_ok=True)
    K_FOLDS = 5
    BATCH_SIZE = 1
    PVAL = 0.2 # percentage of validation

    ## Training with K-fold cross validation
    images_file_paths = glob(os.path.join(DATA_PATH,'image','*.png'))
    labels_file_paths = glob(os.path.join(DATA_PATH, 'label', '*.png'))
    file1 = open('/home/debian/compositIA/compositIA/multires_2024/k-fold-test.txt', 'r')
    Lines = file1.readlines()
    
    df = pd.DataFrame(data={"filename": images_file_paths, 'mask' : labels_file_paths})

    for k, TEST_NAME in enumerate(Lines):
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
        
        train_gen = train_generator(train_data_frame, BATCH_SIZE, train_generator_args, img_size, n_class)
        val_gen = train_generator(val_data_frame, BATCH_SIZE, dict(), img_size, n_class)

        #Define the model
        model = unet(input_size=(img_size[0],img_size[1],3), n_class=n_class)
        model.summary()

        #Fit the u-net model
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join(CHECK_PATH, 'unet_L3_k{}.hdf5'.format(k)),
                                                              monitor='val_dice_coef',
                                                              verbose=1,
                                                              save_best_only=True,
                                                              mode = 'max')
        N_TRAIN = len(train_data_frame)
        N_VALID = len(val_data_frame)
        print(N_TRAIN, N_VALID)
        STEPS_PER_EPOCH = N_TRAIN // BATCH_SIZE
        VALIDATION_STEPS = N_VALID // BATCH_SIZE
        
        history = model.fit(train_gen,
                  validation_data=val_gen,
                  epochs=100,
                  steps_per_epoch=STEPS_PER_EPOCH,
                  validation_steps=VALIDATION_STEPS,
                  verbose=1, 
                  callbacks=[model_checkpoint])
                  
        plot_history(history, CHECK_PATH, 'history_k{}.png'.format(k))
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
    
    train_gen = train_generator(train_data_frame, BATCH_SIZE, train_generator_args, img_size, n_class)
    val_gen = train_generator(val_data_frame, BATCH_SIZE, dict(), img_size, n_class)

    #Define the model
    model = unet(input_size=(img_size[0],img_size[1],3), n_class=n_class)
    model.summary()

    #Fit the u-net model
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join(CHECK_PATH, 'unet_L3_k{}.hdf5'.format(k)),
                                                          monitor='val_dice_coef',
                                                          verbose=1,
                                                          save_best_only=True,
                                                          mode = 'max')

    N_TRAIN = len(train_data_frame)
    N_VALID = len(val_data_frame)
    print(N_TRAIN, N_VALID)
    STEPS_PER_EPOCH = N_TRAIN // BATCH_SIZE
    VALIDATION_STEPS = N_VALID // BATCH_SIZE
    
    history = model.fit(train_gen,
              validation_data=val_gen,
              epochs=100,
              steps_per_epoch=STEPS_PER_EPOCH,
              validation_steps=VALIDATION_STEPS,
              verbose=1, 
              callbacks=[model_checkpoint])
              
    plot_history(history, CHECK_PATH, 'history_k{}.png'.format(k))
    tf.keras.backend.clear_session()
        
if __name__=="__main__":
    main()

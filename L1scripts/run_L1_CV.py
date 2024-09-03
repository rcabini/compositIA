# SPDX-License-Identifier: EUPL-1.2
# Copyright 2024 dp-lab Università della Svizzera Italiana

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image
import matplotlib.patches as mpatches
import skimage.transform as trans
import glob
import pandas as pd
import skimage.io as io
import argparse
from model_L1 import dice_coef

#---------------------------------------------------------------------------

def err_map(seg, gt_seg):
    err = np.zeros(seg.shape+(3,))
    err[:,:,0] = (seg * gt_seg)
    err[:,:,1] = (seg > gt_seg) # false positive
    err[:,:,2] = (seg < gt_seg) # false negative
    return err

#---------------------------------------------------------------------------
    
def dice_error(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred)
    return np.mean( (2. * intersection) / (union))

#---------------------------------------------------------------------------

def main(args):

    im_width, im_height = (512, 512)
    input_size = (512, 512)
    
    weights_path = args.weights_folder
    path = os.path.join(args.data_folder, "image/")
    seg_path = os.path.join(args.data_folder, "label/")
    res_path = os.path.join(args.output_folder, "res/")
    cort_path = os.path.join(args.output_folder, "cort/")
    trab_path = os.path.join(args.output_folder, "trab/")
    os.makedirs(res_path, exist_ok=True)
    os.makedirs(cort_path, exist_ok=True)
    os.makedirs(trab_path, exist_ok=True)
    Folds = pd.read_csv(args.ktxt, sep=" ", header=None)

    print('Finding images ... ')
    row = []
    for k, TEST_NAME in Folds.iterrows():
        TEST_NAME = TEST_NAME.dropna().values
        tf.keras.backend.clear_session()
        MODEL_PATH = os.path.join(weights_path, 'unet_L1_k{}.hdf5'.format(k))
        print('Loading model ... ')
        model = tf.keras.models.load_model(MODEL_PATH, custom_objects={"dice_coef": dice_coef })
        print(' ... Loaded!')
        files = [os.path.join(path, ff + ".png") for ff in TEST_NAME]
        for file in files:
            fn = os.path.basename(file)
            try:
                print(fn)
                ids = []
                ids.append(fn)
                img = io.imread(os.path.join(path, fn), as_gray = True)
                x_img = trans.resize(img, (input_size[0], input_size[1]), mode='constant',cval=0,anti_aliasing=True,preserve_range=True,order=3)
                X = np.zeros((1, im_height, im_width, 1), dtype=np.float32)
                X[0,:,:,0] = (x_img.squeeze()-x_img.min())/(x_img.max()-x_img.min())
                
                mask = io.imread(os.path.join(seg_path, fn), as_gray = False)
                x_mask = trans.resize(mask, (input_size[0], input_size[1], 3), mode='constant',cval=0,anti_aliasing=True,preserve_range=True,order=0)
                x_mask = np.round((x_mask.squeeze()-x_mask.min())/(x_mask.max()-x_mask.min()), decimals=2)            

                results = model.predict(X, verbose=1)
                res_seg = np.round(results)[0,:,:,:]
                
                fig, ax = plt.subplots(2,3, figsize=(20,15), tight_layout=True)
                for a in ax:
                    for a1 in a:
                        a1.set_xticks([]) 
                        a1.set_yticks([]) 
                ax[0,0].imshow(x_img)
                ax[0,1].set_title("True")
                ax[0,1].imshow(x_mask[:,:,1])
                ax[0,2].set_title("Predicted")
                ax[0,2].imshow(res_seg[:,:,1])
                ax[1,0].imshow(x_img)
                ax[1,1].imshow(x_mask[:,:,2])
                ax[1,2].imshow(res_seg[:,:,2])    
                plt.savefig(os.path.join(res_path, fn), bbox_inches = "tight", dpi=200)
                plt.close()
                
                err1 = err_map((x_mask[:,:,1]),res_seg[:,:,1])
                err2 = err_map((x_mask[:,:,2]),res_seg[:,:,2])
                fig, ax = plt.subplots(2,1, figsize=(20,15), tight_layout=True)
                for a in ax:
                    a.set_xticks([]) 
                    a.set_yticks([]) 
                ax[0].imshow(err1)
                ax[1].imshow(err2)      
                pop_a = mpatches.Patch(color='r', label='True')
                pop_b = mpatches.Patch(color='lime', label='False Positive')
                pop_c = mpatches.Patch(color='b', label='False Negative')
                ax[0].legend(handles=[pop_a,pop_b, pop_c], fontsize="large")
                ax[1].legend(handles=[pop_a,pop_b, pop_c], fontsize="large")
                ax[0].set_title("Dice coeff. {}".format(np.round(dice_error(x_mask[:,:,1], res_seg[:,:,1]),decimals=2)))
                ax[1].set_title("Dice coeff. {}".format(np.round(dice_error(x_mask[:,:,2], res_seg[:,:,2]),decimals=2)))
                plt.savefig(os.path.join(res_path, 'err_' + fn), bbox_inches = "tight", dpi=200)
                plt.close()
                row.append([fn, k, np.round(dice_error(x_mask[:,:,1], res_seg[:,:,1]),decimals=2), np.round(dice_error(x_mask[:,:,2], res_seg[:,:,2]),decimals=2)])
                
                res_seg = res_seg*255.
                matplotlib.image.imsave(os.path.join(cort_path, fn), res_seg[:,:,1].astype(np.uint8),  cmap="gray")
                matplotlib.image.imsave(os.path.join(trab_path, fn), res_seg[:,:,2].astype(np.uint8),  cmap="gray")
            except: print("ERROR: {}".format(fn))
            
    df = pd.DataFrame(row, columns=["filename", "k", "dice_TRAB", "dice_CORT"])
    df.to_excel(os.path.join(res_path,"DSC_L1_kfold.xlsx"), index=False)  
    
if __name__=="__main__":
    """Read command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", help='Path to dataset')
    parser.add_argument("--ktxt", help='Path to k-splits txt file')
    parser.add_argument("--weights_folder", help='Path to weights')
    parser.add_argument("--output_folder", help='Path to output')
    args = parser.parse_args()
    main(args)


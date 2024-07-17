import os
import tensorflow as tf
import numpy as np
import imageio
import matplotlib.pyplot as plt
import matplotlib.image
import matplotlib.patches as mpatches
import skimage.transform as trans
import glob
import pandas as pd
import skimage.io as io

#---------------------------------------------------------------------------

def err_map(seg, gt_seg):
    err = np.zeros(seg.shape+(3,))
    err[:,:,0] = (seg * gt_seg)
    err[:,:,1] = (seg > gt_seg)
    err[:,:,2] = (seg < gt_seg)
    return err

#---------------------------------------------------------------------------
    
def dice_error(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred)
    return np.mean( (2. * intersection) / (union))

#---------------------------------------------------------------------------

def main():

    im_width, im_height = (512, 512)
    input_size = (512, 512) #(256, 256)
    
    weights_path = './weights'
    path = './DATA/image/'
    seg_path = './DATA/label/'
    res_path = './weights/res'
    VAT_path = './weights/VAT'
    SAT_path = './weights/SAT'
    SMA_path = './weights/SMA'
    os.makedirs(res_path, exist_ok=True)
    os.makedirs(VAT_path, exist_ok=True)
    os.makedirs(SAT_path, exist_ok=True)
    os.makedirs(SMA_path, exist_ok=True)
    Folds = pd.read_csv('/home/debian/compositIA/GitHub/compositIA/slicer/k-fold-test.txt', sep=" ", header=None)

    print('Finding images ... ')
    row = []
    for k, TEST_NAME in Folds.iterrows():
        TEST_NAME = TEST_NAME.values
        tf.keras.backend.clear_session()
        MODEL_PATH = os.path.join(weights_path, 'unet_L3_k{}.hdf5'.format(k))
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
                img = io.imread(os.path.join(path, fn), as_gray = False)
                x_img = trans.resize(img, (input_size[0], input_size[1]), mode='constant',cval=0,anti_aliasing=True,preserve_range=True,order=3)
                X = np.zeros((1, im_height, im_width, 3), dtype=np.float32)
                X[0,:,:,:] = (x_img.squeeze()-x_img.min())/(x_img.max()-x_img.min())
                
                mask = io.imread(os.path.join(seg_path, fn), as_gray = True)
                x_mask = trans.resize(mask, (input_size[0], input_size[1]), mode='constant',cval=0,anti_aliasing=True,preserve_range=True,order=0)
                x_mask = np.round((x_mask.squeeze()-x_mask.min())/(x_mask.max()-x_mask.min()), decimals=2)
                
                results = model.predict(X, verbose=1)
                res_seg = np.round(results)[0,:,:,:]             
                
                c0, c1, c2, c3 = np.unique(x_mask)
                
                fig, ax = plt.subplots(3,3, figsize=(20,20), tight_layout=True)
                for a in ax:
                    for a1 in a:
                        a1.set_xticks([]) 
                        a1.set_yticks([]) 
                ax[0,0].imshow(x_img[:,:,0])
                ax[0,1].set_title("True")
                ax[0,1].imshow(x_mask==c1)
                ax[0,2].set_title("Predicted")
                ax[0,2].imshow(res_seg[:,:,1])
                ax[1,0].imshow(x_img[:,:,0])
                ax[1,1].imshow(x_mask==c2)
                ax[1,2].imshow(res_seg[:,:,2])  
                ax[2,0].imshow(x_img[:,:,0])
                ax[2,1].imshow(x_mask==c3)
                ax[2,2].imshow(res_seg[:,:,3])    
                plt.savefig(os.path.join(res_path, fn), bbox_inches = "tight", dpi=200)
                plt.close()
                
                err1 = err_map((x_mask==c1),res_seg[:,:,1])
                err2 = err_map((x_mask==c2),res_seg[:,:,2])
                err3 = err_map((x_mask==c3),res_seg[:,:,3])
                fig, ax = plt.subplots(3,1, figsize=(20,20), tight_layout=True)
                for a in ax:
                    a.set_xticks([]) 
                    a.set_yticks([]) 
                ax[0].imshow(err1)
                ax[1].imshow(err2)
                ax[2].imshow(err3)      
                pop_a = mpatches.Patch(color='r', label='True')
                pop_b = mpatches.Patch(color='lime', label='False Positive')
                pop_c = mpatches.Patch(color='b', label='False Negative')
                ax[0].legend(handles=[pop_a,pop_b, pop_c], fontsize="large")
                ax[1].legend(handles=[pop_a,pop_b, pop_c], fontsize="large")
                ax[2].legend(handles=[pop_a,pop_b, pop_c], fontsize="large")
                ax[0].set_title("Dice coeff. {}".format(np.round(dice_error(x_mask==c1, res_seg[:,:,1]),decimals=2)))
                ax[1].set_title("Dice coeff. {}".format(np.round(dice_error(x_mask==c2, res_seg[:,:,2]),decimals=2)))
                ax[2].set_title("Dice coeff. {}".format(np.round(dice_error(x_mask==c3, res_seg[:,:,3]),decimals=2)))
                plt.savefig(os.path.join(res_path, 'err_' + fn), bbox_inches = "tight", dpi=200)
                plt.close()
                row.append([fn, k, np.round(dice_error(x_mask==c1, res_seg[:,:,1]),decimals=2),np.round(dice_error(x_mask==c2, res_seg[:,:,2]),decimals=2),np.round(dice_error(x_mask==c3, res_seg[:,:,3]),decimals=2)])
                
                res_seg = res_seg*255.
                matplotlib.image.imsave(os.path.join(VAT_path, fn), res_seg[:,:,1].astype(np.uint8),  cmap="gray")
                matplotlib.image.imsave(os.path.join(SAT_path, fn), res_seg[:,:,2].astype(np.uint8),  cmap="gray")
                matplotlib.image.imsave(os.path.join(SMA_path, fn), res_seg[:,:,3].astype(np.uint8),  cmap="gray")
            except: print("ERROR: {}".format(fn))
                
    df = pd.DataFrame(row, columns=["filename", "k", "dice_VAT", "dice_SAT", "dice_SMA"])
    df.to_excel(os.path.join(res_path, "DSC_L3_kfold.xlsx"), index=False) 

if __name__=="__main__":
    main()

import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.utils import img_to_array, load_img
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from model_depth_4 import dice_coef, unet
import imageio
import matplotlib.pyplot as plt
import matplotlib.image
import matplotlib.patches as mpatches
import skimage.transform as trans
import glob

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
    input_size = (256, 256)
    
    path = '../DATA_input/test/image'
    seg_path = '../DATA_input/test/label'
    res_path = './res'
    VAT_path = './VAT'
    SAT_path = './SAT'
    SMA_path = './SMA'

    print('Loading model ... ')
    model = keras.models.load_model('./weights/unet_DB1_multi.hdf5', custom_objects={"dice_coef": dice_coef })

    print('Finding images ... ')
    for file in glob.glob(path+"/*.png"):
        fn = os.path.basename(file)
        print(fn)
        ids = []
        ids.append(fn)
        img = load_img(path + '/' + fn, grayscale=False)
        x_img = img_to_array(img)
        X = np.zeros((1, im_height, im_width, 3), dtype=np.float32)
        X[0,:,:,:] = x_img.squeeze() / 255
        X = trans.resize(X, (1, input_size[0], input_size[1], 3), mode='constant',cval=0,anti_aliasing=True,preserve_range=True,order=3)
        
        mask = load_img(seg_path + '/' + fn, grayscale=True)
        x_mask = img_to_array(mask)
        x_mask = x_mask[:,:,0] / 255
        
        results = model.predict(X, verbose=1)
        results = trans.resize(results, (1, im_height, im_width, results.shape[-1]),mode='constant',cval=0,anti_aliasing=True,preserve_range=True,order=0)
        res_seg = np.round(results)[0,:,:,:]
        
        c0, c1, c2, c3 = np.unique(x_mask)
        
        fig, ax = plt.subplots(3,3, figsize=(20,20), tight_layout=True)
        for a in ax:
            for a1 in a:
                a1.set_xticks([]) 
                a1.set_yticks([]) 
        ax[0,0].imshow(x_img[:,:,0])
        ax[0,1].imshow(x_mask==c1)
        ax[0,2].imshow(res_seg[:,:,1])
        ax[1,0].imshow(x_img[:,:,0])
        ax[1,1].imshow(x_mask==c2)
        ax[1,2].imshow(res_seg[:,:,2])  
        ax[2,0].imshow(x_img[:,:,0])
        ax[2,1].imshow(x_mask==c3)
        ax[2,2].imshow(res_seg[:,:,3])    
        plt.savefig(res_path+ '/' + fn, bbox_inches = "tight", dpi=200)
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
        plt.savefig(res_path+ '/err_' + fn, bbox_inches = "tight", dpi=200)
        plt.close()
        
        res_seg = res_seg*255
        matplotlib.image.imsave(VAT_path+ '/' + fn, res_seg[:,:,1].astype(np.uint8),  cmap="gray")
        matplotlib.image.imsave(SAT_path+ '/' + fn, res_seg[:,:,2].astype(np.uint8),  cmap="gray")
        matplotlib.image.imsave(SMA_path+ '/' + fn, res_seg[:,:,3].astype(np.uint8),  cmap="gray")

if __name__=="__main__":
    main()

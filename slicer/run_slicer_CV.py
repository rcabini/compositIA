import os, sys
sys.path.insert(0,'../')
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform as trans
import glob
import pandas as pd
import skimage.io as io
from skimage.feature import peak_local_max
from utils.windowed_utils import *

#---------------------------------------------------------------------------

def draw_center(buff_s, results, centers, GTcenters, dstFolder, base):
    plt.imshow(buff_s, cmap='gray')
    plt.imshow(results, cmap='coolwarm', alpha=0.5)
    plt.plot(centers[0][0],centers[0][1],'cx', linewidth=7.0)
    plt.plot(centers[1][0],centers[1][1],'bx', linewidth=7.0)
    plt.plot(GTcenters[0][0],GTcenters[0][1],'yx', linewidth=7.0)
    plt.plot(GTcenters[1][0],GTcenters[1][1],'rx', linewidth=7.0)
    plt.axis("off")
    plt.savefig(os.path.join(dstFolder, "{}.png".format(base)), bbox_inches='tight', dpi=200)
    plt.close()

#---------------------------------------------------------------------------

def main():

    im_height, im_width, im_ch = (512, 1024, 3)
    input_size = (128, 256, 3)
    
    path = './DATA/image/'
    seg_path = './DATA/label/'
    res_path = './weights/res'
    pred_path = './weights/pred'
    os.makedirs(res_path, exist_ok=True)
    os.makedirs(pred_path, exist_ok=True)
    WEIGHTS_PATH = "./weights/"
    nii_path = '/home/debian/compositIA/DataNIFTI/Images/'
    file1 = open('/home/debian/compositIA/compositIA/multires_2024/k-fold-test.txt', 'r')
    Lines = file1.readlines()
    
    names, kk = [], []
    GT_L1, GT_L3, pred_L1, pred_L3 = [], [], [], []
    GT_L1_ind, GT_L3_ind, pred_L1_ind, pred_L3_ind = [], [], [], []

    print('Finding images ... ')
    row = []
    for k, TEST_NAME in enumerate(Lines):
        if k>-1:
            TEST_NAME = [ff.replace("'", "").replace(" ", "").replace("[", "").replace("]", "").replace("\n", "") for ff in TEST_NAME.split(',')]
            tf.keras.backend.clear_session()
            MODEL_PATH = os.path.join(WEIGHTS_PATH, "Unet_kfold{}.hdf5".format(k))
            print('Loading model ... ')
            model = tf.keras.models.load_model(MODEL_PATH)
            print(' ... Loaded!')
            files = [path + ff + ".png" for ff in TEST_NAME]
            for file in files:
                fn = os.path.basename(file)
                try:
                    base = fn.split(".png")[0]
                    print(base)
                    ids = []
                    ids.append(fn)
                    #read nifti to get the spacing
                    _ , spacing = readData(os.path.join(nii_path,base,'data.nii.gz'), order=3)
                    print(spacing[0])
                    x_img = io.imread(os.path.join(path, fn), as_gray = False)
                    im_height, im_width = x_img.shape[0], x_img.shape[1]
                    X = np.zeros((1, im_height, im_width, 3), dtype=np.float32)
                    X[0,:,:,:] = x_img.squeeze() / 255.
                    X = trans.resize(X, (1, input_size[0], input_size[1], input_size[2]), anti_aliasing=True)
                    
                    x_mask = io.imread(os.path.join(seg_path, fn), as_gray = True)
                    x_mask = x_mask / 255.
                    [gt_L3y, gt_L3x], [gt_L1y, gt_L1x] = peak_local_max(x_mask, min_distance=10, num_peaks=2)
                    if gt_L1x<gt_L3x:
                        gt_L1x, gt_L3x = gt_L3x, gt_L1x
                        gt_L1y, gt_L3y = gt_L3y, gt_L1y
                    GT_centers = ((gt_L1x, gt_L1y), (gt_L3x, gt_L3y))
                    
                    results = model.predict(X, verbose=1)
                    results = trans.resize(results, (1, im_height, im_width, results.shape[-1]), anti_aliasing=True)
                    res = results[0,:,:,0]*255
                    
                    plt.imsave(os.path.join(pred_path,'{}.png'.format(base)),(res).astype(np.uint8))
                    
                    [pr_L3y, pr_L3x], [pr_L1y, pr_L1x] = peak_local_max(res, min_distance=10, num_peaks=2)
                    if pr_L1x<pr_L3x:
                        pr_L1x, pr_L3x = pr_L3x, pr_L1x
                        pr_L1y, pr_L3y = pr_L3y, pr_L1y
                    centers = ((pr_L1x, pr_L1y), (pr_L3x, pr_L3y))
                    
                    draw_center(x_img[:,:,0], res, centers, GT_centers, res_path, base)
                    
                    #distance
                    GT_L1.append(GT_centers[0][0]*spacing[0]) 
                    GT_L3.append(GT_centers[1][0]*spacing[0])                    
                    pred_L1.append(centers[0][0]*spacing[0]) 
                    pred_L3.append(centers[1][0]*spacing[0])
                    names.append(base)
                    kk.append(k)
                    #index
                    GT_L1_ind.append(round(GT_centers[0][0]*spacing[0]/spacing[2]))
                    GT_L3_ind.append(round(GT_centers[1][0]*spacing[0]/spacing[2]))
                    pred_L1_ind.append(round(centers[0][0]*spacing[0]/spacing[2]))
                    pred_L3_ind.append(round(centers[1][0]*spacing[0]/spacing[2]))
                    
                except: print("ERROR: {}".format(fn))
    dist_L1 = np.abs(np.array(pred_L1, dtype=float)-np.array(GT_L1, dtype=float))
    dist_L3 = np.abs(np.array(pred_L3, dtype=float)-np.array(GT_L3, dtype=float))
    df = pd.DataFrame(columns=['filename', 'k', 'GT_L1', 'GT_L3', 'pred_L1', 'pred_L3', 'dist_L1', 'dist_L3'])
    df['filename']=np.array(names)
    df['k']=np.array(kk)
    #distance
    df['GT_L1']=np.array(GT_L1)
    df['GT_L3']=np.array(GT_L3)
    df['pred_L1']=np.array(pred_L1)
    df['pred_L3']=np.array(pred_L3)
    df['dist_L1']=np.array(dist_L1)
    df['dist_L3']=np.array(dist_L3)
    #index
    dfi=pd.DataFrame(columns=['filename', 'k', 'GT_L1_ind', 'GT_L3_ind', 'pred_L1_ind', 'pred_L3_ind', 'dist_L1_ind', 'dist_L3_ind'])
    dfi['filename']=np.array(names)
    dfi['k']=np.array(kk)
    dist_L1_ind = np.abs(np.array(pred_L1_ind, dtype=int)-np.array(GT_L1_ind, dtype=int))
    dist_L3_ind = np.abs(np.array(pred_L3_ind, dtype=int)-np.array(GT_L3_ind, dtype=int))
    dfi['GT_L1_ind']=np.array(GT_L1_ind)
    dfi['GT_L3_ind']=np.array(GT_L3_ind)
    dfi['pred_L1_ind']=np.array(pred_L1_ind)
    dfi['pred_L3_ind']=np.array(pred_L3_ind)
    dfi['dist_L1_ind']=np.array(dist_L1_ind)
    dfi['dist_L3_ind']=np.array(dist_L3_ind)
    print(df)
    df.to_excel(os.path.join(res_path, "/output_kfold_s15.xlsx"), index=False)
    dfi.to_excel(os.path.join(res_path, "/output_kfold_s15_idx.xlsx"), index=False) 
    
if __name__=="__main__":
    main()


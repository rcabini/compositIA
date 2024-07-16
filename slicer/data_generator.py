import os, sys
sys.path.insert(0,'../')
import numpy as np
import cv2
import nibabel as nib
import matplotlib.pyplot as plt
import skimage
import argparse
from utils.windowed_utils import *
from utils.windower import windower
from glob import glob
import pandas as pd
from scipy import ndimage
from utils.aug import *

FIXED_SIZE = [512, 1024]

#---------------------------------------------------------------------------

def draw_GT(centers, dstFolder, prevFolder, base):
    # Intermediate step to dar the L1 and L3 centers on a preview image for verification
    buff_s = skimage.io.imread(dstFolder + os.path.sep + "{}.png".format(base))
    buff_s = skimage.color.gray2rgb(buff_s[:,:,0])
    #L1 on sagital
    rr,cc = skimage.draw.circle_perimeter(centers[0][0],centers[0][1],5,shape=buff_s.shape)
    buff_s[rr,cc,:] = [255,0,0]
    # L3 on sagital
    rr,cc = skimage.draw.circle_perimeter(centers[1][0],centers[1][1],5,shape=buff_s.shape)
    buff_s[rr,cc,:] = [0,0,255]
    #save
    skimage.io.imsave(prevFolder + os.path.sep + "{}.png".format(base), buff_s)

#---------------------------------------------------------------------------

def projection(volume, spacing):
    #extract sagital & coronal projections, scaled and save a preview in the input folder
    sagital = extract_images(volume, axis=0, spacing=spacing) #Sagital
    sagital = np.flipud(sagital)
    
    if sagital.shape[1] >= FIXED_SIZE[1]:
        sagital = sagital[:, :FIXED_SIZE[1],:]
    elif sagital.shape[1] < FIXED_SIZE[1]:
        sagital = np.pad(sagital, ((0,0),(0,FIXED_SIZE[1]-sagital.shape[1]),(0,0)), mode='edge')
    return sagital

#---------------------------------------------------------------------------

def extract_seg_slice(seg):
    try:
        L1 = ndimage.center_of_mass(seg==1)
        L1 = (int(round(L1[0])),int(round(L1[1])),int(round(L1[2])))
        L3 = ndimage.center_of_mass(seg==3)
        L3 = (int(round(L3[0])),int(round(L3[1])),int(round(L3[2])))
    except:
        L1 = [None,None,None]
        L3 = [None,None,None]
    return L1, L3

#---------------------------------------------------------------------------

def import_images(img_path, seg_path, dstFolder, gtrFolder, prevFolder, prefix = "prova", aug=False):
    
    ## Read the nifti image
    (img, spacing) = readData(img_path, order=3)
    ## Read the nifti segmentation
    (seg, _) = readData(seg_path, order=0)
    seg = np.round(seg)
    print(img.shape)
    ## Save projections: extract the 3 windowed channels and resize the image to be isotropic
    sagital = projection(img, spacing)
    print(sagital.shape)
    returns = []
    spots_L13 = np.zeros((sagital.shape[0],sagital.shape[1],2))
    if img.shape == seg.shape:
        L1, L3 = extract_seg_slice(seg)
        if np.all(L1) != None and np.all(L3) != None:
            centers = ((int(img.shape[1]-L1[1]), int(L1[2]*spacing[2]/spacing[0])), #L1 sagital
                       (int(img.shape[1]-L3[1]), int(L3[2]*spacing[2]/spacing[0]))) #L3 sagital

            #groud truth       
            spots_L13[centers[0][0],centers[0][1],0] = 255.
            spots_L13[centers[1][0],centers[1][1],1] = 255.
            spots_L13[:,:,0] = cv2.GaussianBlur(spots_L13[:,:,0], ksize=(0, 0), sigmaX=15, borderType=cv2.BORDER_ISOLATED)
            spots_L13[:,:,1] = cv2.GaussianBlur(spots_L13[:,:,1], ksize=(0, 0), sigmaX=15, borderType=cv2.BORDER_ISOLATED)
            spots_smoothed = np.max(spots_L13, axis=-1)
            spots_smoothed = windower(spots_smoothed, spots_smoothed.min(), spots_smoothed.max())
            #save groud truth
            skimage.io.imsave(gtrFolder + os.path.sep + "{}.png".format(prefix), spots_smoothed.astype(np.uint8))
            #save projection
            skimage.io.imsave(dstFolder + os.path.sep + "{}.png".format(prefix), sagital)
            returns.append([dstFolder + os.path.sep + "{}.png".format(prefix), gtrFolder + os.path.sep + "{}.png".format(prefix)])
            #save images with L1 and L3 positions
            draw_GT(centers, dstFolder, prevFolder, prefix)
            #save constrast adj for data augmentation
            if aug==True:
                #contrast adjust
                sagital_2 = sagital.copy()
                for c in range(sagital_2.shape[-1]):
                    channel = sagital_2[...,c].astype(np.float64)
                    sagital_2[...,c] = windower(channel.astype(np.float64), np.percentile(channel, 1), np.percentile(channel, 99))
                skimage.io.imsave(dstFolder + os.path.sep + "{}_aug.png".format(prefix), sagital_2.astype(np.uint8))
                skimage.io.imsave(dstFolder + os.path.sep + "{}_aug.png".format(prefix), sagital_2.astype(np.uint8))
                returns.append([dstFolder + os.path.sep + "{}_aug.png".format(prefix), gtrFolder + os.path.sep + "{}.png".format(prefix)])
                #uncomment for more augmentations
                #zooming
                #img, lab = zoom(sagital, spots_smoothed)
                #skimage.io.imsave(dstFolder + os.path.sep + "{}_aug1.png".format(prefix), img.astype(np.uint8))
                #skimage.io.imsave(gtrFolder + os.path.sep + "{}_aug1.png".format(prefix), lab.astype(np.uint8))
                #returns.append([dstFolder + os.path.sep + "{}_aug1.png".format(prefix), gtrFolder + os.path.sep + "{}_aug1.png".format(prefix)])
                #rotate
                #img, lab = rotate(sagital, spots_smoothed)
                #skimage.io.imsave(dstFolder + os.path.sep + "{}_aug2.png".format(prefix), img.astype(np.uint8))
                #skimage.io.imsave(gtrFolder + os.path.sep + "{}_aug2.png".format(prefix), lab.astype(np.uint8))
                #returns.append([dstFolder + os.path.sep + "{}_aug2.png".format(prefix), gtrFolder + os.path.sep + "{}_aug2.png".format(prefix)])
                #shift
                #img, lab = shift(sagital, spots_smoothed)
                #skimage.io.imsave(dstFolder + os.path.sep + "{}_aug3.png".format(prefix), img.astype(np.uint8))
                #skimage.io.imsave(gtrFolder + os.path.sep + "{}_aug3.png".format(prefix), lab.astype(np.uint8))
                #returns.append([dstFolder + os.path.sep + "{}_aug3.png".format(prefix), gtrFolder + os.path.sep + "{}_aug3.png".format(prefix)])
        else: print(prefix, "Segmentation not found, stopping")
    else: print(prefix, "Shape mismatch")
    
    return returns

#---------------------------------------------------------------------------

def main():

    img_path = "/home/debian/compositIA/DataNIFTI/Images/"
    seg_path = "/home/debian/compositIA/DataNIFTI/Segmentations/"
    dstFolder = "./DATA/image"
    gtrFolder = "./DATA/label"
    prevFolder = "./DATA/prev"
    os.makedirs(dstFolder, exist_ok=True)
    os.makedirs(gtrFolder, exist_ok=True)
    os.makedirs(prevFolder, exist_ok=True)
    file1 = open('/home/debian/compositIA/compositIA/multires_2024/k-fold-test.txt', 'r')
    Lines = file1.readlines()
    
    K_FOLDS = 5
    ns = [os.path.basename(f) for f in glob(img_path+'*')]
    tot = []
    i = 0
    for k, TEST_NAME in enumerate(Lines):
        if k>=0:
            print("Fold {}/{}".format(k+1, K_FOLDS))
            TEST_NAME = [ff.replace("'", "").replace(" ", "").replace("[", "").replace("]", "").replace("\n", "") for ff in TEST_NAME.split(',')]
            print(TEST_NAME)
            
            for base in TEST_NAME:
                print(base)
                try:
                    i_path = glob(os.path.join(img_path, base, "*.nii.gz"))[0]
                    s_path = glob(os.path.join(seg_path, base, "*.nii.gz"))[0]
                    lista = import_images(i_path, s_path, dstFolder, gtrFolder, prevFolder, base, aug=True)
                    for l in lista:
                        tot.append([k]+l)
                    i+=1
                except: print(base, "ERROR")
    df=pd.DataFrame(tot, columns=['k', 'image', 'label'])
    df.to_excel("./dataset.xlsx", index=False)
    print('Total number of images:', i)
if __name__=="__main__":
    main()


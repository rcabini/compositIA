import os, sys
sys.path.insert(0,'../')
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
from skimage import img_as_ubyte
from glob import glob
import argparse
from utils.windower import windower

#---------------------------------------------------------------------------

def extract_seg_slice(img, seg):
    index = []
    for k in range(seg.shape[2]): 
        if np.any(seg[:,:,k] != 0):
            index.append(k)
    return index

#---------------------------------------------------------------------------

def import_images(img_path, seg_path, img_save_path, seg_save_path, prefix = "prova"):

    nii_img = nib.load(img_path)
    nii_img = nib.as_closest_canonical(nii_img)
    nii_seg = nib.load(seg_path)
    nii_seg = nib.as_closest_canonical(nii_seg)   
    img = nii_img.get_fdata()
    seg = nii_seg.get_fdata()
    
    index = extract_seg_slice(img, seg)
    k = index[1] if len(index)==2 else None
    if k != None:
        s_img = img[:,:,k]
        s_seg = seg[:,:,k]
        if img.shape==seg.shape:
            rgb_seg = np.zeros((s_seg.shape[0], s_seg.shape[1], 3))
            rgb_seg[:,:,1] = (s_seg==1)
            rgb_seg[:,:,2] = (s_seg==2)
            rgb_seg[:,:,0] = 1-((s_seg==1) | (s_seg==2))
            rgb_seg = rgb_seg*255.

            rgb_img = windower(s_img, -1024, 500)
            
            io.imsave(os.path.join(seg_save_path,prefix+'.png'),rgb_seg.astype(np.uint8))
            io.imsave(os.path.join(img_save_path,prefix+'.png'),img_as_ubyte(rgb_img.astype(np.uint8)))
        else: print(prefix, "Shape mismatch")
    else: print(prefix, "Segmentation not found, stopping")
    
#---------------------------------------------------------------------------


def main(args):
    img_save_path = os.path.join(args.output_folder, "image/")
    seg_save_path = os.path.join(args.output_folder, "label/")
    os.makedirs(img_save_path, exist_ok=True)
    os.makedirs(seg_save_path, exist_ok=True)
    img_path = glob(os.path.join(args.data_folder, "Images/*"))
    seg_path = os.path.join(args.data_folder, "Segmentations/")

    for p in img_path:
        try:
            base = str(os.path.basename(p))
            i_path = glob(os.path.join(p, "*.nii.gz"))[0]
            s_path = glob(os.path.join(seg_path, base, "*.nii.gz"))[0]
            import_images(i_path, s_path, img_save_path, seg_save_path, base)
        except:
            print("ERROR: ", base, p)

if __name__=="__main__":

    """Read command line arguments"""
	parser = argparse.ArgumentParser()
	parser.add_argument("data_folder", help='Path to dataset')
	parser.add_argument("output_folder", help='Path to output')
	args = parser.parse_args()
	main(args)

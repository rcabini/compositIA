import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
from skimage import img_as_ubyte
import os
from glob import glob
import matplotlib.image

#---------------------------------------------------------------------------

# Windowing of input data
def windower(data,wmin,wmax):
    """
    windower function, gets in input a numpy array,
    the minimum HU value and the maximum HU value.
    It returns a windowed numpy array with the same dimension of data
    containing values between 0 and 255
    """
    dump = data.copy()
    dump[dump>=wmax] = wmax
    dump[dump<=wmin] = wmin
    dump -= wmin
    w = wmax - wmin
    return dump / w * 255

#---------------------------------------------------------------------------

def extract_seg_slice(img, seg):
    index = None # va inizializzato a None prima!
    for k in range(seg.shape[2]): 
        if np.any(seg[:,:,k] != 0):
            index=k
            break
    if index is None:
        print("Segmentation not found, stopping")
    else:
        print(f"Found segmentation at slice {index}")
    return index

#---------------------------------------------------------------------------

def import_images(img_path, seg_path, img_save_path, seg_save_path, prefix = "prova"):

    nii_img = nib.load(img_path)
    nii_img = nib.as_closest_canonical(nii_img)
    nii_seg = nib.load(seg_path)
    nii_seg = nib.as_closest_canonical(nii_seg)   
    img = nii_img.get_fdata()
    seg = nii_seg.get_fdata()
    
    k = extract_seg_slice(img, seg)
    if k !=None:
        s_img = img[:,:,k]
        s_seg = seg[:,:,k]
        if s_img.shape==s_seg.shape:
            #rgb_seg = np.zeros((s_seg.shape[0], s_seg.shape[1], 4))
            #rgb_seg[:,:,1] = (s_seg==4)
            #rgb_seg[:,:,2] = (s_seg==5)
            #rgb_seg[:,:,3] = (s_seg==6)
            #rgb_seg[:,:,0] = 1-((s_seg==4) | (s_seg==4) | (s_seg==6))
            #rgb_seg = rgb_seg*255
            #plt.imshow(rgb_seg)
            #plt.show()
            
            rgb_seg = np.zeros((s_seg.shape[0], s_seg.shape[1]))
            rgb_seg[s_seg==4] = 1
            rgb_seg[s_seg==5] = 2
            rgb_seg[s_seg==6] = 3
            rgb_seg = rgb_seg * 255 / 3

            rgb_img = np.zeros((s_img.shape[0], s_img.shape[1], 3))
            rgb_img[:,:,0] = windower(s_img, -1024, 2048)
            rgb_img[:,:,1] = windower(s_img, -190, -30)
            rgb_img[:,:,2] = windower(s_img, 40, 100)
            
            io.imsave(os.path.join(seg_save_path,prefix+'.png'),rgb_seg.astype(np.uint8))
            io.imsave(os.path.join(img_save_path,prefix+'.png'),img_as_ubyte(rgb_img.astype(np.uint8)))
        else: print("Shape mismatch")
    else: print("Segmentation not found, stopping")

#---------------------------------------------------------------------------


def main():
    img_save_path = "../DATA_input/img"
    seg_save_path = "../DATA_input/seg"
    img_path = glob("../DataNIFTI/Images/*")
    seg_path = "../DataNIFTI/Segmentations/"

    for p in img_path:
        try:
            base = str(os.path.basename(p))
            i_path = glob(os.path.join(p, "*.nii.gz"))[0]
            s_path = glob(os.path.join(seg_path, base, "*.nii.gz"))[0]
            print(base, i_path, s_path)
            import_images(i_path, s_path, img_save_path, seg_save_path, base)
        except:
            print("ERROR: ", base, i_path, s_path)

if __name__=="__main__":
    main()

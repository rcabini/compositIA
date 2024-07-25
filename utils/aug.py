# SPDX-License-Identifier: EUPL-1.2
# Copyright 2024 dp-lab Università della Svizzera Italiana

import random
import numpy as np
from scipy import ndimage
import skimage.transform as trans

FIXED_SIZE = (512, 1024, 3)

def rotate(image, label):
    # define some rotation angles
    angles = [-15, -10, -5, 5, 10, 15]
    # pick angles at random
    angle = random.choice(angles)
    # rotate volume
    image = ndimage.rotate(image, angle, cval=0, reshape=False) #mode='nearest'
    label = ndimage.rotate(label, angle, cval=0, reshape=False) #mode='nearest'
    return image, label
    
def shift(img, label):
    # define some y-displacemets (in percentage of pixels)
    list_shift = [-0.078, -0.058, -0.038, 0.038, 0.058, 0.078]
    # pick shift at random
    ty = random.choice(list_shift)*img.shape[1]
    tx = random.choice(list_shift)*img.shape[0]
    # shift volume
    img_shift = ndimage.shift(img, shift=(tx,ty,0), order=3, cval=0)
    lab_shift = ndimage.shift(label, shift=(tx,ty), order=3, cval=0)
    return img_shift, lab_shift

def zoom(img, lab):
    x,y,z = img.shape
    list_zoom = [0.05,0.04,0.03,0.02]
    m = random.choice(list_zoom)
    dx = int(m*x)
    dy = int(m*y)

    xm = dx
    xM = x - dx
    ym = dy
    yM = y - dy

    if xm < 0:
        xm=0
    if ym < 0:
        ym=0
    if xM > img.shape[0]:
        xM=img.shape[0]
    if yM > img.shape[1]:
        yM=img.shape[1]
    xm=int(xm)
    ym=int(ym)
    xM=int(xM)
    yM=int(yM)
    im = img[xm:xM,ym:yM,:]
    res = trans.resize(im, (FIXED_SIZE[0], FIXED_SIZE[1]), anti_aliasing=True)*255.
    lb = lab[xm:xM,ym:yM]
    lres = trans.resize(lb, (FIXED_SIZE[0], FIXED_SIZE[1]), anti_aliasing=True)
    return res, lres

def flip(img, lab):
    return img[:,::-1,:], lab[:,::-1]


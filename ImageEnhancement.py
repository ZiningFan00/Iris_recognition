import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def ImageEnhancement(img):

    height=img.shape[0]
    width=img.shape[1]

    means=np.zeros([math.floor(height/16),math.floor(width/16)],dtype=float)

    for i in range(0,math.floor(height/16)):
        for j in range(0,math.floor(width/16)):

            tp=img[i*16+0:i*16+16,j*16+0:j*16+16]
            means[i,j]=np.mean(tp)

    back=cv2.resize(means, (width,height), interpolation=cv2.INTER_CUBIC)
    back=np.uint8(back)

    img=np.uint8(img)
    
    img_1=cv2.subtract(img,back)
    img_1=np.uint8(img_1)

    img_new=img_1.copy()
    
    for i in range(0,math.floor(height/32)):
        for j in range(0,math.floor(width/32)):

            img_new[i*32+0:i*32+32,j*32+0:j*32+32]=cv2.equalizeHist(img_1[i*32+0:i*32+32,j*32+0:j*32+32])
    
    return(img_new)


# img=cv2.imread('enhancetest.png',cv2.IMREAD_GRAYSCALE)
# img=cv2.resize(img, (512,64), interpolation=cv2.INTER_CUBIC)
# img_new=ImageEnhancement(img)



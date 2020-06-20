
# -*- coding: utf-8 -*-
import cv2
import math
import numpy as np

def kernel(x, y, delta_x, delta_y, f = 1):
        result = 1/(2*math.pi*delta_x*delta_y)
        result *= math.e**(-0.5*(x**2/delta_x**2+y**2/delta_y**2))
        # result *= math.cos(2*math.pi*f*math.sqrt(x**2+y**2))
        result *= math.cos(2*math.pi/delta_x)
        return result

def gaborFilter(img, delta_x, delta_y, f = 1):
    '''
    modified gabor filter using a circularly symmetric sinusoidal function
    
    input:
        img: 
            a grayscale image
        delta_x, delta_y:
            space constants of the Gaussian envelope along the x and y axis
        f:
            frequency of the sinusoidal
            
    output:
        img_filtered:
            a grayscale image processed by gabor filter
    '''
    M = img.shape[0]
    N = img.shape[1]
    img_filtered = np.zeros((M, N))
    kernel_matrix = np.zeros((2*M, 2*N))
    for m in range(M):
        for n in range(N):
            kernel_matrix[m+M, n+N] = kernel(m, n, delta_x, delta_y)
    
    kernel_matrix[M:, :N] = np.flip(kernel_matrix[M:, N:], axis = 0)
    kernel_matrix[:M, N:] = np.flip(kernel_matrix[M:, N:], axis = 1)
    kernel_matrix[:M, :N] = np.flip(kernel_matrix[:M, N:], axis = 0)
        
    for m in range(M):
        for n in range(N):
            img_filtered[m, n] = np.sum(img * kernel_matrix[m:m+M, n:n+N])
    return img_filtered
    
def featureExtraction(img):
    '''
    input:
        img:
            normalized image with the size 64 * 512

    output:
        a 1-D feature vector with the length 1536
    '''
    img = img[:48, :]

    img1 = gaborFilter(img, 3, 1.5)
    img1 = np.abs(img1)
    img2 = gaborFilter(img, 4.5, 1.5)
    img2 = np.abs(img2)
    list1 = []
    list2 = []
    
    for i in range(6):
        for j in range(64):
            m = np.mean(img1[8*i:8*i+8, 8*j:8*j+8])
            list1.append(m)
            sigma = np.std(img1[8*i:8*i+8, 8*j:8*j+8])
            list1.append(sigma)
            
            m = np.average(img2[8*i:8*i+8, 8*j:8*j+8])
            list2.append(m)
            sigma = np.average(np.abs(img2[8*i:8*i+8, 8*j:8*j+8]-m))
            list2.append(sigma)
            
    return list1 + list2

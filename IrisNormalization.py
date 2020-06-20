# import main as loc
import cv2
import numpy as np

def ImageNormalization(img,circle_inner,circle_outer,theta0):
    height=64.0
    width=512.0
    img_new=np.zeros((int(height),int(width)))
    try:
        img_new=np.uint8(np.zeros((int(height),int(width))))
        for i in range(0,int(width)):
            for j in range(0,int(height)):
                theta=2.0*np.pi*float(i)/width+theta0
                # x=int(circle_inner[0]+circle_inner[0]*np.cos(theta)+float(j)*(circle_outer[0]*np.cos(theta)-circle_inner[0]*np.cos(theta))/height)  
                # y=int(circle_inner[1]+circle_inner[1]*np.sin(theta)+float(j)*(circle_outer[1]*np.sin(theta)-circle_inner[1]*np.sin(theta))/height)
                x=int(circle_inner[0]+np.sin(theta)*(circle_inner[2]+float(j)* (circle_outer[2]-circle_inner[2])/ height ))
                y=int(circle_inner[1]+np.cos(theta)*(circle_inner[2]+float(j)* (circle_outer[2]-circle_inner[2])/ height ))
                img_new[j,i]=img[y,x]
        return img_new
    except:
        return img_new

def rotateNormalization(img,circle_inner,circle_outer):
    img_norm_list=[]
    for tp in range(-9,10,3):
        theta0=tp*1.0/180.0*np.pi
        img_norm_list.append( ImageNormalization(img,circle_inner,circle_outer,theta0))
    return img_norm_list




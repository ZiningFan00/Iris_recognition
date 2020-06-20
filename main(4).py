import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import IrisNormalization as norm
import IrisLocalization as loc
import IrisMatching as match
import ImageEnhancement as enhance
import FeatureExtraction as extract
import Evaluate as eva
import csv
import pandas as pd
def combine(folder_path,x,mode):
    return  folder_path+"/"+str(x)+"/"+str(mode)+"/" 

def readFromPath(folder_path,mode=1):
    files_folder= os.listdir(folder_path) #得到文件夹下的所有文件名称
    files_folder=sorted(files_folder)
    files_folder=[ combine(folder_path,x,mode) for x in files_folder]
    image_total_path = []
    for folder_path in files_folder[1:]: #遍历文件夹
        image_path=os.listdir(folder_path)
        image_path=sorted(image_path)
        for image in image_path:
            if image[-3:]=="bmp":
                image_total_path.append( folder_path+image)
    return image_total_path

def loc_norm_extract(folder_path):
    image_total_path= readFromPath(folder_path)
    print(len(image_total_path))
    ii=0  
    num=0
    X=[]
    y=[]
    for image_path in image_total_path:
        clas=int( image_path[-11:-8])
        img_orig=cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
        img_cut=loc.roughLocalization(img_orig)
        laplacian,center=loc.binarizeEdgeDetection(img_cut)
        row_x,col_y,r,pupil=loc.innerBoundComputation(img_cut, laplacian,center)
        img_cut2,row_x1,col_y1=loc.shrink(img_cut,row_x,col_y,r)
        [row_x_o,col_y_o,r_o],dist_o,img_outer=loc.cannyEdge(img_cut2,row_x1,col_y1,r)
        img_norm_list=norm.rotateNormalization(img_cut2,[col_y1,row_x1,r],[col_y_o,row_x_o,r_o])
        # cv2.imwrite("train_pupil"+ image_path[image_path.rfind("/"):image_path.rfind(".")  ] +"_p.bmp"  ,pupil)
        # cv2.imwrite("train_outer"+ image_path[image_path.rfind("/"):image_path.rfind(".")  ] +"_o.bmp"  ,img_outer)
        for i,img_norm in enumerate(img_norm_list):
            img_enhance=enhance.ImageEnhancement(img_norm)
            FeatureList=extract.featureExtraction(img_enhance)
            X.append(FeatureList)
            y.append(clas)
        print(ii)
        ii+=1
    return X,y
   
def test_norm_extract(folder_path):
    image_total_path= readFromPath(folder_path,2)
    ii=0
    num=0
    label_test=[]
    label_test_re=[]
    f_test_list=[]
    f_test_re_list=[]
    dist_test_list=[]
    clas_list=[]
    for image_path in image_total_path:
        clas=int( image_path[-11:-8]  )
        img_orig=cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
        img_cut=loc.roughLocalization(img_orig)
        laplacian,center=loc.binarizeEdgeDetection(img_cut)
        row_x,col_y,r,pupil=loc.innerBoundComputation(img_cut, laplacian,center)
        img_cut2,row_x1,col_y1=loc.shrink(img_cut,row_x,col_y,r)
        [row_x_o,col_y_o,r_o],dist_o,img_outer=loc.cannyEdge(img_cut2,row_x1,col_y1,r)
        img_norm_list=norm.rotateNormalization(img_cut2,[col_y1,row_x1,r],[col_y_o,row_x_o,r_o])
        # cv2.imwrite("test_pupil"+ image_path[image_path.rfind("/"):image_path.rfind(".")  ] +"_p.bmp"  ,pupil)
        # cv2.imwrite("test_outer"+ image_path[image_path.rfind("/"):image_path.rfind(".")  ] +"_o.bmp"  ,img_outer)
        for i,img_norm in enumerate(img_norm_list):
            img_enhance=enhance.ImageEnhancement(img_norm)
            f_test_un=extract.featureExtraction(img_enhance)
            f_test_list.append(f_test_un)
            label_t,dist_list=match.irisMatching(f_test_un, X_train, y_train,clas)
            clas_list.append(clas)
            label_test.append(label_t)
            f_test_re=match.FisherLinearDisTest(f_test_un, LDA)
            f_test_re_list.append(f_test_re)
            label_t,dist_list=match.irisMatching(f_test_re, f_train, y_train,clas)
            label_test_re.append(label_t)
            dist_test_list.append(dist_list)
        print(ii)
        ii=ii+1
    # return f_test_list,f_test_re_list,label_test, label_test_re,dist_test_list,clas_list
    return f_test_list,clas_list

def train(X_train,y_train):
    f_train,LDA=match.FisherLinearDis(X_train,y_train)
    f_train=np.asarray(f_train)
    np.savetxt("train_feature.csv", f_train, delimiter=",")
    return f_train,LDA

def test( f_test_list,f_train,X_train, y_train,clas_list, LDA):
    label_test=[]
    f_test_re_list=[]
    label_test_re=[]
    dist_test_list=[]
    for i,f_test_un in enumerate(f_test_list):
        # if i>500:
        #     break
        label_t,dist_list=match.irisMatching(f_test_un, X_train, y_train,clas_list[i])
        label_test.append(label_t)
        f_test_re=match.FisherLinearDisTest(f_test_un, LDA)
        f_test_re_list.append(list(f_test_re))
        label_t,dist_list=match.irisMatching(f_test_re, f_train, y_train,clas_list[i])
        label_test_re.append(label_t)
        dist_test_list.append(dist_list)
        print(i)
    return f_test_re_list,label_test, label_test_re,dist_test_list


# def rightLabel(folder_path):
#     image_total_path= readFromPath(folder_path,2)
#     ii=0
#     clas_list=[]
#     for image_path in image_total_path[0:]:
#         clas=int( image_path[-11:-8]  )
#         for i in range(7):
#             # label_t,dist_list=match.irisMatching(f_test_un, X_train, y_train,clas)
#             clas_list.append(clas)
#         print(ii)
#         ii=ii+1
#     return clas_list
#     # return f_test_list,f_test_re_list,label_test, label_test_re,dist_test_list,clas_list
#     # return f_test_list,clas_list

if __name__=="__main__":
    folder_path="dataset"
    # clas_list=rightLabel(folder_path)
    # clas_list=np.asarray(clas_list)
    # np.savetxt("clas_list.csv", clas_list, delimiter=",")
    # X_train,y_train=loc_norm_extract(folder_path)
    # X_train=np.asarray(X_train)
    # y_train=np.asarray(y_train)
    # np.savetxt("X.csv", X_train, delimiter=",")
    # np.savetxt("y.csv", y_train, delimiter=",")

    X_train=np.array(pd.read_csv("X.csv")) 
    y_train=np.array(pd.read_csv("y.csv")) 
    f_train, LDA=match.FisherLinearDis(X_train,y_train)
    f_train=np.asarray(f_train)
    np.savetxt("feature.csv", f_train, delimiter=",")

    ## f_test_list,f_tests_re_list,label_test,label_test_re,dist_test_list=test(folder_path,f_train,X_train,y_train,LDA)
    # f_test_list,clas_list=test_norm_extract(folder_path)
    f_test_list=np.array(pd.read_csv("f_test_list_tp.csv")) 
    clas_list=np.array(pd.read_csv("clas_list_tp.csv")) 
    f_test_re_list,label_test, label_test_re,dist_test_list=test( f_test_list,f_train,X_train, y_train,clas_list, LDA)
    # f_test_list=np.asarray(f_test_list)
    f_test_re_list=np.asarray(f_test_re_list)
    label_test=np.asarray(label_test)
    label_test_re=np.asarray(label_test_re)
    clas_list=np.asarray(clas_list)
    # np.savetxt( "clas_list.csv",clas_list, delimiter=",")
    # np.savetxt("f_test_list.csv", f_test_list, delimiter=",")
    np.savetxt("f_test_re_list_tp.csv", f_test_re_list, delimiter=",")
    np.savetxt( "label_test_tp.csv",label_test, delimiter=",")
    np.savetxt( "label_test_re_tp.csv",label_test_re, delimiter=",")

    # np.savetxt( "dist_test_list.csv",dist_test_list, delimiter=",")

    # f_test_list=np.array(pd.read_csv("f_test_list.csv")) 
    # f_test_re_list=np.array(pd.read_csv("f_test_re_list.csv"))    
    # label_test=np.array(pd.read_csv("label_test.csv"))     
    # label_test_re=np.array(pd.read_csv("label_test_re.csv"))     

    eva.evaluateVerification(dist_test_list)
    eva.evaluateIdentification(label_test_re,label_test,clas_list)

















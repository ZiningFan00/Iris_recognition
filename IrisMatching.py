import cv2
import numpy as np
import math
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt

def FisherLinearDis(X, y):
    LDA=LinearDiscriminantAnalysis(n_components=107)
    LDA.fit(X, y)
    featureSet=LDA.transform(X)
    return featureSet, LDA # return the feature set of all datasets and the LDA object

def FisherLinearDisTest(X, LDA):
    X=np.array(X,)
    X.reshape(1,-1)
    # featureSet=LDA.transform(X)
    X_new = np.dot(X - LDA.xbar_, LDA.scalings_)
    return X_new[:LDA._max_components]# return the feature set of all datasets and the LDA object


def irisMatching(feature, featureSet, y,index):
    dist_list=[]
    dist_l=[]
    for i, f in enumerate(featureSet):
        f = np.array(f)
        d1 = np.average(np.abs(f-feature))
        d2 = np.average((f-feature)**2)
        d3 = 1 - np.sum(f*feature)/(np.sqrt(np.sum(f**2)*np.sum(np.array(feature)**2)))
        # d=mdist()
        # d.id_train=y[i]
        # d.id_test=index
        # d.dlist=[d1,d2,d3]
        dist_l.append([d1,d2,d3])
        # dist_list.append(d)
        dist_list.append([ y[i,0], index, min([d1,d2,d3]) ])
    label=[]
    dist_l=np.array(dist_l)
    for i in range(3):
        label.append(y[ list(dist_l[:,i]).index(min(list(dist_l[:,i]))),0])
    return label,dist_list











    


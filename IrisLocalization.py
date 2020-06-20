import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

def roughLocalization(image):
    _,img_binary=cv2.threshold(image,70,255,cv2.THRESH_BINARY)
    img_binary=cv2.medianBlur(img_binary,9)
    _,img_binary=cv2.threshold(img_binary,80,255,cv2.THRESH_BINARY)
    ## dilate
    kernel = np.ones((7,7), np.uint8) 
# img_erosion = cv2.erode(img, kernel, iterations=1) 
    img_dilation = cv2.dilate(img_binary, kernel, iterations=5) 
    tp=np.uint8(255*np.ones(image.shape))
    img_rev=tp-img_dilation
    _, contours,hierarchy = cv2.findContours(img_rev, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area=[cv2.contourArea(c) for c in contours]
    max_area_index=area.index(max(area)) 
    rect=cv2.boundingRect(contours[max_area_index])
    center= [rect[1]+rect[3]/2.0, rect[0]+rect[2]/2.0 ]
    img_cut=image[ int(max( center[0]-170,0)):int(min(center[0]+170, image.shape[0])) ,  int(max( center[1]-150,0)):int(min(center[1]+150, image.shape[1])) ]
    img_cut=image.copy()
    # cv2.imshow('tp',img_cut)
    # cv2.waitKey(0)
    return img_cut

            
def binarizeEdgeDetection(image):
    _,img_binary=cv2.threshold(image,70,255,cv2.THRESH_BINARY)
    img_binary=cv2.medianBlur(img_binary,9)
    _,img_binary=cv2.threshold(img_binary,80,255,cv2.THRESH_BINARY)
    laplacian = np.uint8(cv2.Laplacian(img_binary,cv2.CV_64F))
    _,contours, hierarchy = cv2.findContours(laplacian, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    tp=[]
    max_area_c=[]
    max_area=0
    for c in contours:
        bounding=cv2.boundingRect(c)
        area=cv2.contourArea(c)
        if bounding[2]<50 or bounding[3]<50:
            for i in range(bounding[1],bounding[1]+bounding[3]):
                for j in range(bounding[0],bounding[0]+bounding[2]):
                    if laplacian[i,j]>0:
                        laplacian[i,j]=0
        else:
            if max_area<area:
                max_area=area
                max_area_c=c.copy()

    bounding=cv2.boundingRect(max_area_c)
    center=[bounding[1]+bounding[3]/2.0,bounding[0]+bounding[2]/2.0]
    # tp=laplacian.copy()
    # tp = cv2.drawContours(tp, np.array(max_area_c), -1, (255,0,0), 3)
    # cv2.imshow("tp",tp)
    # cv2.waitKey(0)

    return laplacian,center




def innerBoundComputation(img_cut,image,center):
    # cv2.imshow("tp",image)
    # cv2.waitKey(0)
    ##-----------------------------------Hough----------------------------------
    ## Hough circle
    # tp_circle=image.copy()
    # circles = cv2.HoughCircles(tp_circle,cv2.HOUGH_GRADIENT,1,20,
    #                         param1=50,param2=30,minRadius=0,maxRadius=0)
    # circles = np.uint16(np.around(circles))
    # tpp=[]
    # for i in circles[0,:]:
    #     # if calDist(  np.array([i[1],i[0]]),np.array([col_y,row_x]) )<20.0:
    #         # draw the outer circle
    #     cv2.circle(tp_circle,(i[0],i[1]),i[2],(255,0,0),2)
    #     tpp=i
    #         # draw the center of the circle
    # cv2.imshow("tp",tp_circle)
    # cv2.waitKey(0)
    # return tpp[1],tpp[0],tpp[2],tp_circle

    ##--------------------------------MLS--------------------------------------
    # get points
    point_list=[]
    center=[int(center[0]),int(center[1])]
    l1=list(image[int(center[0]):,int(center[1])])
    r1=center[0]+l1.index(max(l1))
    c1=center[1]
    point_list.append([r1,c1])
    # tpr=calDist( [r1,c1],center)
    l2=list(image[int(center[0]+(r1-center[0])/4.0*3.0),int(image.shape[1]*0.2):center[1]+1])
    l3=list(image[int(center[0]+(r1-center[0])/2.0),center[1]:int(image.shape[1]*0.9)])
    point_list.append([int(center[0]+(r1-center[0])/4.0*3.0),int(image.shape[1]*0.2)+l2.index(max(l2))])
    point_list.append([int(center[0]+(r1-center[0])/2.0),center[1]+l3.index(max(l3))])
    l2=list(image[int(center[0]+(r1-center[0])/8.0*5.0),int(image.shape[1]*0.2):center[1]+1])
    l3=list(image[int(center[0]+(r1-center[0])/10.0*7.0),center[1]:int(image.shape[1]*0.8)])
    point_list.append([int(center[0]+(r1-center[0])/8.0*5.0),int(image.shape[1]*0.2)+l2.index(max(l2))])
    point_list.append([int(center[0]+(r1-center[0])/10.0*7.0),center[1]+l3.index(max(l3))])

    l2=list(image[int(center[0]+(r1-center[0])/8.0*7.0),int(image.shape[1]*0.2):center[1]+1])
    l3=list(image[int(center[0]+(r1-center[0])/10.0*9.0),center[1]:int(image.shape[1]*0.8)])
    point_list.append([int(center[0]+(r1-center[0])/8.0*7.0),int(image.shape[1]*0.2)+l2.index(max(l2))])
    point_list.append([int(center[0]+(r1-center[0])/10.0*9.0),center[1]+l3.index(max(l3))])
    theta4=85.0/180.0*np.pi
    r_tp=calDist( point_list[len(point_list)-1],center)
    l4=list(image[max(int(center[0]-r_tp*np.sin(theta4)),0):int(center[0]), int(center[1]+center[1]*np.cos(theta4))])
    point_list.append([max(int(center[0]-r_tp*np.sin(theta4)),0)+l4.index(max(l4)),int(center[1]+center[1]*np.cos(theta4))])
    row_x,col_y,r=computeCircle(point_list)
    tp=   img_cut.copy()
    cv2.circle(tp,(int(col_y),int(row_x)), int(r), (255,0,0) )
    return row_x,col_y,r,tp


def computeCircle(point_list):
    point_list=np.array(point_list)
    sum_x=np.sum(point_list[:,0])
    sum_y=np.sum(point_list[:,1])
    sum_xx= np.sum( point_list[:,0]**2 )  
    sum_yy= np.sum( point_list[:,1]**2 )  
    sum_xy=np.sum(point_list[:,0]*point_list[:,1] ) 
    sum_xxx= np.sum( point_list[:,0]**3 )  
    sum_yyy= np.sum( point_list[:,1]**3 )  
    sum_xyy=np.sum(point_list[:,0]*point_list[:,1]**2 )
    sum_xxy=np.sum(point_list[:,0]**2*point_list[:,1] )

    n=point_list.shape[0]
    C=n*sum_xx-sum_x*sum_x
    D=n*sum_xy-sum_x*sum_y

    E = n*sum_xxx + n*sum_xyy - (sum_xx+sum_yy)*sum_x
    G = n*sum_yy - sum_y*sum_y
    H = n*sum_xxy + n*sum_yyy - (sum_xx+sum_yy)*sum_y
    a = (H*D-E*G)/(C*G-D*D)
    b = (H*C-E*D)/(D*D-G*C)
    c = -(a*sum_x + b*sum_y + sum_xx + sum_yy)/n

    row_x = a/(-2.0)
    col_y = b/(-2.0)
    r = np.sqrt(a*a+b*b-4*c)/2.0
    return row_x,col_y,r

def shrink(img_cut,row_x,col_y,r):
    ind1=3.5
    ind2=3.00
    img_cut2=img_cut[int(max( row_x-ind2*r,0)):int(min(row_x+ind2*r, img_cut.shape[0])) ,  int(max(col_y-ind1*r,0)):int(min(col_y+ind1*r, img_cut.shape[1])) ]
    if max( row_x-ind2*r,0)==row_x-ind2*r:
        row_x= row_x-(int(row_x-ind2*r)) 
    if max(col_y-ind1*r,0)==col_y-ind1*r:
        col_y=col_y-(int(col_y-ind1*r)) 
    tpp=img_cut2.copy()
    tpp=cv2.circle(tpp,(int(col_y),int(row_x)),int(r),(255,0,0),1)
    return img_cut2,row_x,col_y

def calDist( vec_point1,vec_point2 ):
    return np.sqrt(np.sum(np.power(np.array(vec_point1)-np.array(vec_point2),2)))

def outerBoundComputation(img_cut,image,row_x,col_y,r):
    point_list=[]
    ## left
    left=[]
    for i in range(5):
        theta=(i+1)*5.0/180.0*np.pi
        left.append(list(image[int(row_x+3.0*r*np.sin(theta)),int(max(col_y-3.0*r*np.cos(theta),0)):int(col_y)]))
    i=0
    for l in left:
        try:
            theta=(i+1)*5.0/180.0*np.pi
            i=i+1
            point_list.append([int(row_x+3.0*r*np.sin(theta)),int(max(col_y-3.0*r*np.cos(theta),0))+int(np.where(np.array(l)==255)[0][0])])
        except:
            continue
    right=[]
    for i in range(6):
        theta=(i+1)*5.0/180.0*np.pi
        l=list(image[int(row_x+3.0*r*np.sin(theta)),int(col_y):int(min(col_y+3.0*r*np.cos(theta),img_cut.shape[1]))])
        l.reverse()
        right.append(l)
    i=0
    for l in right:
        try:
            theta=(i+1)*5.0/180.0*np.pi
            i=i+1
            point_list.append([int(row_x+3.0*r*np.sin(theta)),int(col_y)+len(l)-int(np.where(np.array(l)==255)[0][0])-1])
        except:
            continue
    point_list_filter=[]
    dist=[calDist(pt,(row_x,col_y)) for pt in point_list]
    num=[calNum(a) for a in dist]
    counts = np.bincount(num)
    num_most=np.argmax(counts)
    
    for d in enumerate(dist):
        if calDist(d[1],num_most)<15.0:
            point_list_filter.append(point_list[d[0]])     
    tp=img_cut.copy()
    for pt in point_list:
        cv2.circle(tp,(pt[1],pt[0]), 2, (255,0,0) )
    # cv2.imshow("tp",tp)
    # cv2.waitKey(0)
    row_x_o=0
    col_y_o=0
    r_o=0
    if len(point_list_filter)>=3:
        row_x_o,col_y_o,r_o=computeCircle(point_list_filter)
    dist_o=calDist((row_x_o,col_y_o),(row_x,col_y))
    cv2.circle(tp,(int(col_y_o),int(row_x_o)), int(r_o), (255,0,0) )
    # cv2.imshow("tp",tp)
    # cv2.waitKey(0)
    # print("dist=",dist_o)
    return [row_x_o,col_y_o,r_o],dist_o

def calNum(a):
    tp=int(a)//5
    if a-tp*5<2.5 and a-tp*5>-2.5 :
        return tp*5
    else:
        return tp*5+5


def cannyEdge(img_cut2,row_x,col_y,r):  
    # img_cut2_mb=cv2.medianBlur(img_cut2,5)    
    img_cut2_mb=cv2.equalizeHist(img_cut2)
    img_cut2_mb=cv2.medianBlur(img_cut2_mb,5)
    img_binary=cv2.adaptiveThreshold(img_cut2_mb,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,21,2)
    img_binary=cv2.medianBlur(img_binary,9)
    ind=1.1
    tp=img_binary.copy()
    cv2.circle(tp, (int(col_y),int(row_x)), int(r*ind), (0,0,0), -1)
    # cv2.imshow("ci",tp)
    # cv2.waitKey(0)
    img_binary=filterImage(tp)
    # cv2.imshow("bi",img_binary)
    # cv2.waitKey(0)   
    kernel = np.ones((3,3), np.uint8) 
    img_binary = cv2.erode(img_binary, kernel, iterations=1) 
    img_binary=filterImage(img_binary)
    # cv2.imshow("bi",img_binary)
    # cv2.waitKey(0)   
    res=cv2.bitwise_and(img_cut2,img_cut2,mask = img_binary)
    # cv2.imshow("res",res)
    # cv2.waitKey(0)
    img_tpp=img_cut2.copy()
    try:
        cc=[]
        tp_circle=img_binary.copy()
        im=img_cut2_mb.copy()
        circles = cv2.HoughCircles(tp_circle,cv2.HOUGH_GRADIENT,1,50,
                                param1=80,param2=30,minRadius=0,maxRadius=tp_circle.shape[0])           
        circles = np.uint16(np.around(circles))
        # print("able")
        min_dist=img_cut2.shape[0]
        hough_im=img_cut2.copy()
        for i in circles[0,:]:
            dist=calDist(  np.array([i[0],i[1]]),np.array([col_y,row_x]) )
            cv2.circle(hough_im, (int(i[0]),int(i[1])),int(i[3]),(255,0,0),2)
            if dist<50.0 and dist<min_dist:
                min_dist=dist
                cc=i
                cc=i
        [row_x_o,col_y_o,r_o],dist_o=outerBoundComputation(img_cut2,img_binary,row_x,col_y,r)
        # cv2.imshow("hough",hough_im)
        # cv2.waitKey(0)

        if min_dist<dist_o :
            if cc!=[]:
                cv2.circle(img_tpp,(cc[0],cc[1]),cc[2],(255,0,0),2)
                # cv2.imshow("tp",img_tpp)
                # cv2.waitKey(0)
                return cc,min_dist,img_tpp
            else:
                return [0,0,0],0,img_tpp
        else:
            cv2.circle(img_tpp,(int(col_y_o),int(row_x_o)), int(r_o), (255,0,0) ,2)
            # cv2.imshow("tp",img_tpp)
            # cv2.waitKey(0)
            return [row_x_o,col_y_o,r_o],dist_o,img_tpp
    except:
        [row_x_o,col_y_o,r_o],dist_o=outerBoundComputation(img_cut2,img_binary,row_x,col_y,r)
        cv2.circle(img_tpp,(int(col_y_o),int(row_x_o)), int(r_o), (255,0,0) ,2)
        # cv2.imshow("tp",img_tpp)
        # cv2.waitKey(0)
        # print(dist_o)
        if dist_o>15:
            return [0,0,0],0,img_tpp    
        else:
            return [row_x_o,col_y_o,r_o],dist_o,img_tpp


def fillImage(image):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # image_tp=cv2.drawContours(image, contours, -1, (255,0,0), 3)
    img_tp=image.copy()
    for c in contours:
        bounding=cv2.boundingRect(c)
        for i in range(bounding[1],bounding[1]+bounding[3]+1):
            for j in range(bounding[0],bounding[0]+bounding[2]+1):
                if cv2.pointPolygonTest(c,(j,i),False)>=0:
                    img_tp[i,j]=255
    # cv2.imshow("tp",img_tp)
    # cv2.waitKey(0)
    return img_tp


##1/3 of maximum area
def filterImage(image):
    _, contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area_list=[cv2.contourArea(c) for c in contours]
    max_area=max(area_list)
    # max_length_index=max_length.index(max(max_length))
    for c in contours:
        if cv2.contourArea(c)<max_area/4.0:
            bounding=cv2.boundingRect(c)
            for i in range(bounding[1],bounding[1]+bounding[3]+1):
                for j in range(bounding[0],bounding[0]+bounding[2]+1):
                    if cv2.pointPolygonTest(c,(j,i),False)>=0:
                        image[i,j]=0
    # cv2.imshow("tp",image)
    # cv2.waitKey(0)
    return image

# def showImage(image,name,mode=1):
#     cv2.imshow(name,image)
#     if mode==0:
#         cv2.waitKey(0)
# def showImage(image):
#     cv2.imshow("tp",image)
#     cv2.waitKey(0)
    
def extractIris(circle_inner,circle_outer,img_cut):
    tp=np.uint8(np.zeros(img_cut.shape))
    iris=[]
    try:
        tp=cv2.circle(tp, (int(circle_outer[0]),int(circle_outer[1])),int(circle_outer[2]),(255,0,0), -1)
        # cv2.circle(img_binary, (int(col_y),int(row_x)), int(r*ind), (0,0,0), -1)
        tp=cv2.circle(tp, (int(circle_inner[0]),int(circle_inner[1])),int(circle_inner[2]),(0,0,0), -1)
        iris=cv2.bitwise_and(img_cut,img_cut,mask=tp)
        return iris
    except:
        return iris



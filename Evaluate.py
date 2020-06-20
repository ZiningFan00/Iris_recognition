import numpy as np

def match(a,b):
    if a==b:
        return 1
    else:
        return 0

def evaluateCRR(label_list,label_right):
    right_match=[]
    for i,label in enumerate(label_list):
        right_match.append(match(int(label), int(label_right[i])))
    return float( sum(right_match)*1.0/len(right_match))


def evaluateFR(dist_thres,dist_list):
    FMR=0
    FNMR=0
    sum=0
    for dist in dist_list:
        for di in dist:
            sum+=1.0
            if int(di[0])==int(di[1]):
                # match train_id=a and test_id=a
                if di[2]>dist_thres:
                    FNMR+=1
            else:
                # match train_id=a and test_id=b
                if di[2]<dist_thres:
                    FMR+=1
    FMR= FMR*1.0/sum
    FNMR=FNMR*1.0/sum
    return FMR,FNMR

def evaluateIdentification(label_list_reduced,label_list_unreduced,label_right):
    CRR_reduced=[]
    CRR_unreduced=[]
    print("Similarity Measure"+'\t'+"Original FeatureSet"+'\t'+"Reduced FeatureSet")
    for i in range(3):
        CRR_reduced=evaluateCRR(label_list_reduced[:,i],label_right)  
        CRR_unreduced=evaluateCRR(label_list_unreduced[:,i],label_right)
        # CRR_reduced=evaluateCRR(label_list_reduced[i],label_right)  
        # CRR_unreduced=evaluateCRR(label_list_unreduced[i],label_right)
        print(str(i+1)+'\t\t'+ str(round(CRR_unreduced,4)*100.0)+"%"+'\t\t'+str(round(CRR_reduced,4)*100.0)+"%")


def evaluateVerification(dist_list):
    FNMR=list(range(0,3))
    FMR=list(range(0,3))
    print("Threshhold"+'\t'+ "False Match Rate(%)"+'\t'+  "False Non-Match Rate(%)")
    thresh=[0.446,0.472,0.502]
    for i in range(len(thresh)):
        FMR[i],FNMR[i]=evaluateFR(thresh[i],dist_list)
    print(str(thresh[i])+'\t\t'+ str(round(FMR[i],3))+'\t\t'+str(round(FNMR[i])))
    

    



    







    
    

    
    


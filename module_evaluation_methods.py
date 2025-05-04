# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 17:55:58 2024

@author: vikir
"""

#module_evaluation_methods.py
import numpy as np
import module_classification_methods as cl

def K_FOLD_Evaluation (X,y,clChoice,nFolds):
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=nFolds,shuffle=True)
    Z=kf.split(X)
    nClasses=np.max(y)+1
    TT=np.zeros((nClasses,nClasses),int)
     
    for i in Z:
        X_train=X[i[0]];y_train=y[i[0]]
        X_test=X[i[1]];y_test=y[i[1]];
        K=cl.classifierChoice(X_train, X_test, y_train, y_test,clChoice)
        
        for k in range(len(K)):
            ii=y_test[k];jj=K[k];
            TT[ii,jj]=TT[ii,jj]+1
    return(TT)
#--------------------------------------------------------------------  
def HoldOutEvaluation(X,y,clChoice,epochs):
    nClasses=np.max(y)+1
    TT=np.zeros((nClasses,nClasses),int) #for maxL+1 classes
    from sklearn.model_selection import train_test_split
    for iEpochs in range(epochs):
        (X_train, X_test, y_train, y_test) = train_test_split(X,y,test_size=0.4)
        K=cl.classifierChoice(X_train, X_test, y_train, y_test,clChoice)
        for k in range(len(K)):
            ii=y_test[k];
            jj=K[k];
            TT[ii,jj]=TT[ii,jj]+1
    return(TT)
  
 #--------------------------------------------------------------------
def LOOEvaluation(X,y,clChoice):
     from sklearn.model_selection import LeaveOneOut
     loo=LeaveOneOut()
     Z=loo.split(X)
     nClasses=np.max(y)+1
     TT=np.zeros((nClasses,nClasses),int);
     y=np.asarray(y);
     for i in Z:
         X_train=X[i[0]];y_train=y[i[0]]
     
         X_test=X[i[1]];y_test=y[i[1]];
         K=cl.classifierChoice(X_train, X_test, y_train, y_test,clChoice)
         for k in range(len(K)):
             ii=y_test[k];jj=K[k];
             TT[ii,jj]=TT[ii,jj]+1
     return(TT)
#--------------------------------------------------------------------
def BootStrap_Evaluation (X,y,clChoice,epochs):
    nClasses=np.max(y)+1
    TT=np.zeros((nClasses,nClasses),int)
    nPatterns=np.size(X,0)
    for iEpochs in range(epochs):
        percent=0.8
        idx=np.random.choice(nPatterns,np.int(nPatterns*percent))
        x=X[idx,:]
        yy=y[idx]
    #        Z=HoldOutEvaluation(x,yy,clChoice,epochs)
        Z=K_FOLD_Evaluation(x,yy,clChoice,3)
        TT=TT+Z
    #        print("Z: ");print(Z)
    return(TT)
 
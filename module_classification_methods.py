# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 19:12:46 2021

@author: medisp-2
"""
# module_classification_methods.py
import numpy as np
import scipy as sc


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
        K=classifierChoice(X_train, X_test, y_train, y_test,clChoice)
        for k in range(len(K)):
            ii=y_test[k];jj=K[k];
            TT[ii,jj]=TT[ii,jj]+1
    return(TT)

#--------------------------------------------------------------------
def K_FOLD_Evaluation (X,y,clChoice,nFolds):
    from sklearn.model_selection import KFold
    
    kf = KFold(n_splits=nFolds,shuffle=True)
    Z=kf.split(X)
    nClasses=np.max(y)+1
    TT=np.zeros((nClasses,nClasses),int)
 
    for i in Z:
        X_train=X[i[0]];y_train=y[i[0]]
        X_test=X[i[1]];y_test=y[i[1]];
        K=classifierChoice(X_train, X_test, y_train, y_test,clChoice)
        
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
        idx=np.random.choice(nPatterns,int(nPatterns*percent))
        x=X[idx,:]
        yy=y[idx]
#        Z=HoldOutEvaluation(x,yy,clChoice,epochs)
        Z=K_FOLD_Evaluation(x,yy,clChoice,3)
        TT=TT+Z
#        print("Z: ");print(Z)
    return(TT)
        
#--------------------------------------------------------------------  
def HoldOutEvaluation(X,y,clChoice,epochs):
   nClasses=np.max(y)+1
   TT=np.zeros((nClasses,nClasses),int) #for maxL+1 classes
   from sklearn.model_selection import train_test_split
   for iEpochs in range(epochs):
       (X_train, X_test, y_train, y_test) = train_test_split(X,y,test_size=0.4)
       K=classifierChoice(X_train, X_test, y_train, y_test,clChoice)
       for k in range(len(K)):
           ii=y_test[k];
           jj=K[k];
           TT[ii,jj]=TT[ii,jj]+1
   return(TT)
# ---------------------------------------------------------------
def my_nchoosek(nFeats, nCombs):
    from itertools import combinations
    x = list(range(0, nFeats))
    x = np.asarray(x, int)
    #    L=combs(x)
    L = [c for i in range(nCombs + 1) for c in combinations(x, i)]
    return (L)
# ----------------------------------------------------------------
def printAccuracies(TT, fNames, classifierName,evalChoice):
    print("for features: ", end='')
    print(fNames)
    if(evalChoice == 0 or evalChoice== 2 ):
        print("Truth Table:")
        print(TT)
    sens = 100 * (TT[0, 0] / (TT[0, 0] + TT[0, 1]))
    spec = 100 * (TT[1, 1] / (TT[1, 0] + TT[1, 1]))
    accuracy = 100 * (TT[0, 0] + TT[1, 1]) / np.sum(TT)

    print(" % ACCURACIES: ")
    print("sensitivity: %4.2f " % sens)
    print("specificity: %4.2f " % spec)
    print("accuracy: %4.2f " % accuracy)
# -------------------------------------------------------------------
def classifierChoice(X_train, X_test, y_train, y_test, clChoice):
    if (clChoice == 0):
        from sklearn.neighbors import NearestCentroid
        clf = NearestCentroid()
        clf.fit(X_train, y_train)
        K = clf.predict(X_test)

    elif (clChoice == 1):
        C = 3
        from sklearn.neighbors import KNeighborsClassifier
        clf = KNeighborsClassifier(n_neighbors=C)
        clf.fit(X_train, y_train)
        K = clf.predict(X_test)

    elif (clChoice == 2):
        from sklearn.naive_bayes import GaussianNB
        clf = GaussianNB()
        clf.fit(X_train, y_train)
        K = clf.predict(X_test)

    elif (clChoice == 3):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        clf = LinearDiscriminantAnalysis()
        clf.fit(X_train, y_train)
        K = clf.predict(X_test)

    elif (clChoice == 4):
        from sklearn.linear_model import LogisticRegression
        clf = clf = LogisticRegression()
        clf.fit(X_train, y_train)
        K = clf.predict(X_test)

    elif (clChoice == 5):
        from sklearn.linear_model import Perceptron
        clf = Perceptron(tol=1e-3, random_state=0)
        clf.fit(X_train, y_train)
        K = clf.predict(X_test)

    elif (clChoice == 6):
        nFeats=len(X_train)
        from sklearn.neural_network import MLPClassifier
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                            hidden_layer_sizes=(nFeats * 2, 2),
                            random_state=1)

        clf.fit(X_train, y_train)
        K = clf.predict(X_test)

    elif (clChoice == 7):
        from sklearn.svm import LinearSVC
        clf = LinearSVC()
        clf.fit(X_train, y_train)
        K = clf.predict(X_test)

    elif (clChoice == 8):
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=10)
        clf.fit(X_train, y_train)
        K = clf.predict(X_test)

    elif (clChoice == 9):
        from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
        clf = DecisionTreeClassifier()
        clf.fit(X_train, y_train)
        K = clf.predict(X_test)
    return (K)


 
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 15:46:34 2021

@author: medisp-2
"""

#=============   MODULE features_reduction_methods.py
# featureReduction.py
import numpy as np
import scipy as sc
# ------------------------------------------------------------
def signTestRanking(class1, class2, fNames, Crit, pValue):
    # 1. Apply significance tests
    from scipy import stats
    nFeats = np.size(class1, 1)
    P = np.zeros(nFeats, float)
    for iFeats in range(nFeats):
        X = class1[:, iFeats]
        Y = class2[:, iFeats]
        if (Crit == 0):
            (Criterion, p) = sc.stats.ttest_ind(X, Y, equal_var=False)
        elif (Crit == 1):
            (Criterion, p) = sc.stats.mannwhitneyu(X, Y, alternative='two-sided')  # wilcoxon for unequal vectors
        P[iFeats] = p
    # 1. ===========Sort features by P significance ====================
    RANK = P
    z1 = np.argsort(RANK)  # find indices of ascending ranking
    z1 = (np.asarray(z1, int))  # turn to np array

    # 2. retain features with desired p<=pValue
    ic = 0
    for x in range(nFeats):
        if (P[z1[x]] <= pValue):
            ic = ic + 1
    z1 = z1[0:ic]
    print(z1)
    class1 = class1[:, z1]
    #    print(class1)
    class2 = class2[:, z1]
    #    print(class2)
    fNames = np.asarray(fNames)
    fNames = fNames[z1]
    print("rankedFeatIndices_pvalue: ", end="")
    print(P[z1])
    print("rankedFeatIndices: ", end="")
    print(z1)
    print("rankedFeatNames: ", end="")
    print(fNames)

    return (class1, class2, fNames)


# ---------------------------------------------------------------------------
def corr_Ranking(class1, class2, fNames, corrValue):
    import pandas as pd
    nFeats = np.size(class1, 1)
    combinedClasses = np.concatenate((class1, class2), axis=0)
    df = pd.DataFrame(data=combinedClasses)
    # 1.-----------Get crosscorrelation coeffs -----------------------------
    z = df.corr()
    z = np.asarray(z, float)
    corrArray = np.zeros(nFeats, float)
    for j in range(nFeats):
        for i in range(nFeats):
            if (np.abs(z[j, i]) != 1):
                corrArray[j] = corrArray[j] + np.abs(z[j, i])
    corrArray = corrArray / (nFeats - 1)
    print("===========crossCorrelation ====================")
    print(corrArray)
    corr1 = np.argsort(corrArray)
    print("ranked correlation Indices", end='')
    print(corr1)

    # 2. retain features with desired correlation <=corrValue
    ic = 0
    for x in range(nFeats):
        if (corrArray[corr1[x]] <= corrValue):
            ic = ic + 1
    print("ic: ", end='')
    print(ic)
    z1 = corr1[0:ic]
    print("feature indices with correlation <=corrValue:  ", end='')
    print(z1)
    class1 = class1[:, z1]
    #    print(class1)
    class2 = class2[:, z1]
    fNames = np.asarray(fNames)
    fNames = fNames[z1]
    print("rankedFeatures_corrValues: ", end="")
    print(corrArray[z1])
    print("rankedFeatNames: ", end="")
    print(fNames)
    return (class1, class2, fNames)


# --------------------------------------------------------------------------
def PCA_Ranking(class1, class2, desVar):
    from sklearn.decomposition import PCA
    nPattClass1 = np.size(class1, 0)
    nPattClass2 = np.size(class2, 0)
    nFeats = np.size(class1, 1)
    combinedClasses = np.concatenate((class1, class2), axis=0)
    t=min(np.size(combinedClasses,0),np.size(combinedClasses,1))
    combinedClasses=combinedClasses[:,0:t]
    nFeats=np.size(combinedClasses,1)
    
    # feature extraction
    n_components = nFeats
    pca = PCA(n_components)
    fit = pca.fit(combinedClasses)
    Z = fit.explained_variance_ratio_
    print("Explained Variance: %s" % Z)
    Z = np.asarray(Z, float)
    sum = 0
    ic = 0
    for i in range(len(Z)):
        sum = sum + Z[i]
        ic = ic + 1
        if (sum > desVar):
            break
    print(ic)
    sortedFeatureData = pca.transform(combinedClasses)
    class1 = sortedFeatureData[0:nPattClass1, 0:ic]
    class2 = sortedFeatureData[nPattClass1:nPattClass1 + nPattClass2, 0:ic]
    #    print("\n")
    #    print(class1)
    #    print(class2)

    fNames = ["" for x in range(ic)]
    
    for i in range(ic):
        fNames[i] = "PCA" + str(i)
    #    print(fNames)
    fNames = np.asarray(fNames)
    return (class1, class2, fNames)


def mixedCriterion_Ranking(class1, class2, fNames, sigTest):
    import pandas as pd
    nFeats = np.size(class1, 1)
    nPattClass1 = np.size(class1, 0)
    nPattClass2 = np.size(class2, 0)
    combinedClasses = np.concatenate((class1, class2), axis=0)
    df = pd.DataFrame(data=combinedClasses)
    # -----------Get crosscorrelation coeffs -----------------------------
    z = df.corr()
    z = np.asarray(z, float)
    corrArray = np.zeros(nFeats, float)
    for j in range(nFeats):
        for i in range(nFeats):
            if (np.abs(z[j, i]) != 1):
                corrArray[j] = corrArray[j] + np.abs(z[j, i])
    corrArray = corrArray / (nFeats - 1)

    # ----------------------  get significance test ranking -------------
    import scipy as sc
    from scipy import stats
    Wp = np.zeros(nFeats, float)
    P = np.zeros(nFeats, float)
    for iFeats in range(nFeats):
        X = combinedClasses[0:nPattClass1, iFeats]
        Y = combinedClasses[nPattClass1:nPattClass1 + nPattClass2, iFeats]
        if (sigTest == 0):
            (Wtest, p) = sc.stats.ttest_ind(X, Y, equal_var=False)
        elif (sigTest == 1):
            (Wtest, p) = sc.stats.mannwhitneyu(X, Y, alternative='two-sided')  # wilcoxon for unequal vectors
        Wp[iFeats] = np.abs(Wtest)
        P[iFeats] = p
    RANK = np.zeros(nFeats, float)
    ALPHA = 0.5
    for iFeats in range(nFeats):
        RANK[iFeats] = Wp[iFeats] * (1 - ALPHA * corrArray[iFeats])

    #    print("===========Criterion====================")
    z1 = np.argsort(RANK)
    sortedFeatureData = combinedClasses[:, z1[::-1]]  # reverse order to descending order
    fNames = fNames[z1[::-1]]
    class1 = sortedFeatureData[0:nPattClass1, :]
    class2 = sortedFeatureData[nPattClass1:nPattClass1 + nPattClass2, :]

    return (class1, class2, fNames)


# -------------------------------------------------------------------------
def RFE_wrapper(class1, class2, fNames, classif):
    # RFE with LogReg
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LogisticRegression

    nFeats = np.size(class1, 1)
    nPattClass1 = np.size(class1, 0)
    nPattClass2 = np.size(class2, 0)
    combinedClasses = np.concatenate((class1, class2), axis=0)
    combinedClassesLabels = np.concatenate((np.zeros(nPattClass1, int),
                                                np.ones(nPattClass2, int)), axis=0)
    
    # feature extraction

    if (classif == 0):
        model = LogisticRegression(solver='lbfgs')
    elif (classif == 1):
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor()
    elif (classif == 2):
        from sklearn.ensemble import ExtraTreesClassifier
        model = ExtraTreesClassifier(n_estimators=20)

    elif (classif == 3):
        from sklearn.tree import DecisionTreeRegressor
        model = DecisionTreeRegressor(max_leaf_nodes=20)
    

    rfe = RFE(model)# have to update the sklearn library
    fit = rfe.fit(combinedClasses, combinedClassesLabels)
    print("Num Features: %d" % fit.n_features_)
    print("Selected Features: %s" % fit.support_)
    print("Feature Ranking: %s" % fit.ranking_)
    z = fit.ranking_

    indx=[]
    for i in range(nFeats):
        if (int(z[i]) == 1):
            indx.append(i)
    indx=np.asarray(indx,int)        
        
    print("selected-feature indices: ", end='')
    print(indx, end='')
    print("\n. Selected feature names:  ", end='')
    print(fNames[indx])
    sortedFeatureData = combinedClasses[:, indx]
    #    print(sortedFeatureData)
    class1 = sortedFeatureData[0:nPattClass1, :]
    class2 = sortedFeatureData[nPattClass1:nPattClass1 + nPattClass2, :]
    fNames = fNames[indx]
    return (class1, class2, fNames)

 
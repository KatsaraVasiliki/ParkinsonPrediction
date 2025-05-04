
#Statistical Analysis of Parkisons data 
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score
#https://www.kaggle.com/code/prasannashreecse/parkinson-s-diseases-detection-project

def load_data():
    parkinson_data = pd.read_csv("parkinsons.csv")
    df=parkinson_data
    y=df['status'].values
    y=np.asarray(y,int)
    print(df.shape)
    df=df.drop(columns=['name','status'])
    fNames=df.columns;fNames=np.asarray(fNames)
    X=df.values
    X=np.asarray(X)
    y=np.asarray(y,int)
    class1=X[y==0]
    class2=X[y==1]
    X= np.concatenate((class1, class2), axis=0)
    y1=y[y==0]
    y2=y[y==1]
    y= np.concatenate((y1, y2), axis=0)
    X=np.asarray(X,float)
    y=np.asarray(y,int)
   
    return X,y,fNames


   # ------------------------------------------------
def scatterDiagrams2d(class1, class2, fNames, maxAcc, classifierName ):
    ##pl=ot

    fz = 8
    plt.figure(figsize=(fz, fz))
    plt.plot(class1[:, 0], class1[:, 1], 'o', color='c')
    plt.plot(class2[:, 0], class2[:, 1], 'd', color='m')
    plt.grid()
    title_font = {'fontname':'Arial', 'size':'16', 'color':'black', 'weight':'normal',
                  'verticalalignment':'bottom'} # Bottom vertical alignment for more space
    axis_font = {'fontname':'Arial', 'size':'16'}
    plt.legend(('class1', 'class2', 'misClassified'), loc='best')
    plt.xlabel(fNames[0],**axis_font)
    plt.ylabel(fNames[1],**axis_font)
    plt.title('Scatter diagram of : ' + fNames[0] + ' Vs ' + fNames[1] +"\n"+
              'classifier: ' + classifierName + ' accuracy: ' +
              str("%4.2f %%" % maxAcc),**title_font)
    plt.show()
   
   # ----------------------------------------------------------------
def scatterDiagrams3d(class1, class2, fNames, maxAcc, classifierName):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    fz = 8
    plt.figure(figsize=(fz, fz))

    x = class1[:, 0]
    y = class1[:, 1]
    z = class1[:, 2]

    ax = plt.axes(projection='3d')
    ax.scatter(x, y, z, c='m', marker='d', s=50, alpha=1)

    x = class2[:, 0]
    y = class2[:, 1]
    z = class2[:, 2]
    ax.scatter(x, y, z, c='c', marker='o', s=50, alpha=1)
    ax.set_xlabel(fNames[0],fontsize=16)
    ax.set_ylabel(fNames[1],fontsize=16)
    ax.set_zlabel(fNames[2],fontsize=16)
    ax.set_title(fNames[0] + ' Vs ' + fNames[1] + ' Vs ' + fNames[2] +
                 '\n' + ' classifier: ' + classifierName + ' accuracy: ' +
                 str("%4.2f %%" % maxAcc))
    ax.title.set_size(15)
    ax.legend(('Healthy', 'PD', 'misClassified'), loc=1)
    plt.show()


#-----------------------------------------------------------
#-------------------- MAIN PROGRAM -------------------------
#-----------------------------------------------------------
# import numpy as np
import moduleUtils as U
# import scipy as sc
# from scipy import stats
# import matplotlib.pyplot as plt
# import ROC_calc as RC

U.cls()
X,y,fNames=load_data()

import moduleUtils as U
import module_features_reduction_methods as fr 
import module_classification_methods as cl
import module_evaluation_methods as ev
import random
U.cls()

# Reset the random number generator to its initial state
random.seed(42)

namesOfClassifiers = ['0:MDC classifier', '1:KNN classifier', '2:Bayesian classifier',
                      '3:LDA classifier', '4:LogReg classifier', 
                      '5:PERCEPTRON_classifier', '6:MLP_classifier', '7:SVM_classifier',
                      '8:RandomForest_classifier', '9:CART_classifier']
EVALUATION_METHOD = ['LOO_evaluation', 'BOOTSTRAP_evaluation', 'K_FOLD_evaluation',
                      'HoldOut_Evaluation']

'''
Feature Reduction Methods 
0:Significance Test ranking only,
1:correlation ranking,
2:Mixed criterion (correlation+signif. test),
3: PCA ranking,
4:RFE wrapper

'''
clChoice=1;#choose classifier
evaluation_method=1
chooseFeatReductionMethod = 4#choose feature reduction method


from sklearn import preprocessing
X=preprocessing.normalize(X,axis=0)

def split_classes(X,y):
    class1=X[y==0]
    class2=X[y==1]
    return(class1,class2)    
    
(class1,class2)=split_classes(X,y)
print(np.shape(class1))
print(np.shape(class2))
class1 = np.asarray(class1, dtype=float)
class2 = np.asarray(class2, dtype=float)
nFeats = np.size(class1, 1)
fNames = np.asarray(fNames)

print("class1 ")
print(np.shape(class1))
print("class2 ")
print(np.shape(class2))
print("features")
print(np.shape(fNames))





if (chooseFeatReductionMethod == 0):  # Significance Test ranking only
    chooseSignTest = 0  # 0: ttest, 1:wilkcoxon test
    pValue = 0.001  # reduce features by smallest p
    (class1, class2, fNames) = fr.signTestRanking(class1, class2, fNames, chooseSignTest, pValue)
        
elif (chooseFeatReductionMethod == 1):  # correlation ranking
    corrValue = 0.2  # select features by smallest correlation
    (class1, class2, fNames) = fr.corr_Ranking(class1, class2, fNames, corrValue)
    print("correlation Ranking")

elif (chooseFeatReductionMethod == 2):  # Mixed criterion (correlation+signif. test)
    significanceTest = 1  # 0: ttest, 1:wilkcoxon test : rank all features by larger mixed criterion
    (class1, class2, fNames) = fr.mixedCriterion_Ranking(class1, class2, fNames, significanceTest)
    
    print("Mixed Criterion Ranking")

elif (chooseFeatReductionMethod == 3):  # PCA ranking
    desiredVariancePercentage = 0.8  # reduce features by retaining those with
    # sum of variances up to desiredVariancePercentage
    (class1, class2, fNames) = fr.PCA_Ranking(class1, class2, desiredVariancePercentage)
    print("PCA Ranking")

elif (chooseFeatReductionMethod == 4):  # RFE wrapper
    classif = 3  # 0: for logistic regressor 1:for RF regressor 2:ExtraTreesClassifier
    # 3.DecisionTreeRegressor
    (class1, class2, fNames) = fr.RFE_wrapper(class1, class2, fNames, classif)
    print("RFE_wrapper")



desiredFeatures=len(fNames)-1
if(np.size(class1,1)>desiredFeatures):
    class1=class1[:,0:desiredFeatures]
    class2=class2[:,0:desiredFeatures]
    fNames=fNames[0:desiredFeatures]
    nFeats=desiredFeatures


print(np.shape(class1))
print(np.shape(class2))

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
nPattClass1 = np.size(class1, 0)
nPattClass2 = np.size(class2, 0)
nCombs = 3  # nCombs between 2 and nFeats
L = cl.my_nchoosek(nFeats, nCombs)####???????

L1 =np.asarray( L[nFeats + 1:np.size(L)],dtype=object)###?????????
# U.RETURN()
t1 = U.tic()
bestFeatures = []
maxAccuracy = 0
bestTT = np.zeros((2, 2), int)  # for 2 classes
combinedClassesLabels = np.concatenate((np.zeros(nPattClass1, int),
                                        np.ones(nPattClass2, int)), axis=0)
##einai to y


for iCombs in range(np.size(L1, 0)):
    cl1 = class1[:, L1[iCombs]]
    cl2 = class2[:, L1[iCombs]]
    feats = L1[iCombs]
    feats = np.asarray(feats)

    combinedClasses = np.concatenate((cl1, cl2), axis=0)
    nPattCombinedClasses = np.size(combinedClasses, 0)
    epochs=10
    if (evaluation_method==0):
        truthTable = ev.LOOEvaluation(combinedClasses, combinedClassesLabels, clChoice)
    elif (evaluation_method==1):
        truthTable=ev.BootStrap_Evaluation (combinedClasses, combinedClassesLabels, clChoice,epochs)
    elif (evaluation_method==2):
        Kfolds=10
        truthTable=ev.K_FOLD_Evaluation(combinedClasses, combinedClassesLabels, clChoice,Kfolds)  
    elif (evaluation_method==3): 
        truthTable=ev.HoldOutEvaluation(combinedClasses, combinedClassesLabels,clChoice,epochs)
    
    accuracy = 100 * (truthTable[0, 0] + truthTable[1, 1]) / np.sum(truthTable)
    
    
    if (accuracy >= maxAccuracy):
        maxAccuracy = accuracy
        bestFeatures = feats
        bestTT = truthTable
        classifierName = namesOfClassifiers[clChoice]
        evaluation_name= EVALUATION_METHOD[evaluation_method]
        cl.printAccuracies(truthTable, fNames[feats], classifierName,evaluation_method)
        print('classifier used: %s' % namesOfClassifiers[clChoice])
        print("evaluation method: ",EVALUATION_METHOD[evaluation_method])
        if (len(feats) == 2):
            scatterDiagrams2d(cl1, cl2, fNames[feats], accuracy, classifierName)
        elif (len(feats) == 3):
            scatterDiagrams3d(cl1, cl2, fNames[feats], accuracy, classifierName)
        print("=================================================")            
        

print("FINAL: BEST RESULTS")
cl1 = class1[:, bestFeatures]
cl2 = class2[:, bestFeatures]
fN = fNames[bestFeatures]
nClassif = namesOfClassifiers[clChoice]
cl.printAccuracies(bestTT, fN, nClassif,evaluation_method)
print('classifier used: %s' % namesOfClassifiers[clChoice])
print("evaluation method: ",EVALUATION_METHOD[evaluation_method])
if (len(bestFeatures) == 2):
    scatterDiagrams2d(cl1, cl2, fN, maxAccuracy, nClassif)
elif (len(bestFeatures) == 3):
    scatterDiagrams3d(cl1, cl2, fN, maxAccuracy, nClassif)
plt.show()
U.toc(t1, namesOfClassifiers[clChoice])
   






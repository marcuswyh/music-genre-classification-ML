import models
import pandas as pd
import numpy as np

performHyperOptimize = False
performFeatureSelection = False
performTest = False

data = pd.read_csv("musicfeatures//data.csv", delimiter=",")

data = data.drop(["filename"], axis=1)

# to run all 8 initial models
def runModels(testdata, testlabel):
    res = models.knn(testdata, testlabel)
    print ("knn: ", res.mean())
    
    res = models.svm(testdata, testlabel)
    print ("svm: ",res.mean())
    
    res = models.decisionTree(testdata, testlabel)
    print ("decision tree: ",res.mean())
    
    res = models.naiveBayes(testdata, testlabel)
    print ("Gaussian NB: ",res.mean())
    
    res = models.randomForest(testdata, testlabel)
    print ("random forest: ",res.mean())
    
    res = models.nearestCentroid(testdata, testlabel)
    print ("nearest centroid: ",res.mean())
    
    res = models.extraTree(testdata, testlabel)
    print ("extra tree: ",res.mean())
    
    res = models.extraTrees(testdata, testlabel)
    print ("extra trees: ",res.mean())
    

#============= CATEGORICAL MAPPING ==================

# convert all 10 categorical genre labels to numerical values
genre_map = {'blues': 6, 'rock': 3, 
             'metal': 4, 'jazz': 5, 
             'pop': 1, 'reggae': 7, 
             'hiphop': 8, 'disco': 9, 
             'country': 10, 'classical': 2}

# apply map to label column
data["label"] = data["label"].map(genre_map)



#============= SCALING ==================

from sklearn import preprocessing

# scale columns except labels using minMaxScaler
scalingObj = preprocessing.MinMaxScaler()
data[data.columns[:28]] = scalingObj.fit_transform(data[data.columns[:28]])



#============= OUTLIER DETECTION ==================

import seaborn as sns
import matplotlib.pyplot as plt

# calculate inter-quartile range
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1

# get all rows that are within 1.5 times the upper and lower inter-quartile range
outlier_rm = data[~((data < (Q1 - 1.5 * IQR)) |(data > (Q3 + 1.5 * IQR))).any(axis=1)]
display = (data < (Q1 - 1.5 * IQR)) |(data > (Q3 + 1.5 * IQR))

temp = data.drop(['label'], axis=1).copy()

print (display)
print (outlier_rm.shape)
sns.boxplot(data = temp)
plt.show()


##============= FEATURE SELECTION ==================

t, l = data.drop(["label"], axis=1).copy(), data["label"]

deleteFeatures = 11

if (performFeatureSelection):
    from sklearn.feature_selection import SelectPercentile
    from sklearn.feature_selection import f_regression
    
    selector = SelectPercentile(f_regression, percentile = 25)
    selector.fit(t,l)
    
    # get all feature importance scores
    res = np.array(selector.scores_)
    
    # sort by least important
    seq = res.argsort()
    col = t.columns.values.tolist()
    
    # append feature names into list
    test = []
    for i in range (len(col)):
        for j in range (len(seq)):
            if seq[i] == j:
                test.append(col[j])
    
    # remove 11 least important features from dataset
    names = test[:deleteFeatures] 
    t = t.drop(names, axis=1)





#================= TEST RUN =======================


testlabel = data["label"].reset_index(drop=True)
testdata = data.drop(["label"], axis=1).copy()

if (performTest):
    # before feature extraction
    print("")
    print("before feature extraction")
    print ("==========================")
    runModels(testdata,testlabel)


    #after feature extraction
    if (performFeatureSelection):
        print ("")
        print("after feature extraction ")
        print ("========================")
        runModels(t,l)



#================= HYPER PARAMETER OPTIMIZATION =======================

if (performHyperOptimize):
    depth = []
    depth.append(None)
    depth.extend(range(1,21))
    
    etc_params = {
            'n_estimators': [10,100,200,300,400,500],
            'max_features': ['sqrt', 'log2'],
            'max_depth' : depth,
            'criterion' :['gini', 'entropy'],
            'random_state': [0]
            }
    rfc_params = {
            'n_estimators': [10,100,200,300,400,500],
            'max_features': ['sqrt', 'log2'],
            'max_depth' : depth,
            'criterion' :['gini', 'entropy'],
            'random_state': [0]
            }
    knn_params = {
            'n_neighbors': range(1,51),
            'weights' : ['uniform', 'distance'],
            'p': [1,2,3],
            'algorithm' :['ball_tree', 'kd_tree', 'brute']
            }
    
    
    
    from sklearn.ensemble.forest import RandomForestClassifier
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import GridSearchCV
    
    rfc_grid = GridSearchCV(RandomForestClassifier(), rfc_params, cv=10, n_jobs=-1, verbose=0)
    rfc_grid.fit(testdata, testlabel)
    
    etc_grid = GridSearchCV(ExtraTreesClassifier(), etc_params, cv=10, n_jobs=-1, verbose=0)
    etc_grid.fit(testdata, testlabel)
    
    knn_grid = GridSearchCV(KNeighborsClassifier(), knn_params, cv=10, n_jobs=-1, verbose=0)
    knn_grid.fit(testdata, testlabel)
    
    print ("")
    print("Random forest best params: ",rfc_grid.best_params_ , "with a score of ", rfc_grid.best_score_)
    print("Extra trees best params: ",etc_grid.best_params_ , "with a score of ", etc_grid.best_score_)
    print("knn best params: ",knn_grid.best_params_ , "with a score of ", knn_grid.best_score_)
    
    

#================ RESEARCH: FEATURE SELECTION =========================
    
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

t,l = testdata, testlabel

                #-----------------------------
                # FILTER METHOD - CORRELATION
                #-----------------------------
import seaborn as sns
import matplotlib.pyplot as plt

#Using Pearson Correlation
plt.figure(figsize=(12,10))
cor = data.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.savefig("corr.png", dpi=300)

#Correlation with output variable
cor_target = abs(cor["label"])
#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.1]
relevant_features = list(relevant_features.index)
# delete label feature
del relevant_features[-1]

# results
temp = t[relevant_features]
print ("Number of selected features: ", len(relevant_features))
print ("Features: ",relevant_features)

print("FILTER METHOD - CORRELATION")
print("===========================")
print("knn: ",models.feat_ex_KNN(temp,l))
print("random forest: ",models.feat_ex_RFC(temp,l))
print("extra trees: ",models.feat_ex_ETC(temp,l))


                #-----------------------------
                # BACKWARD ELIMINATION METHOD
                #-----------------------------
import statsmodels.api as sm
#Adding constant column of ones, mandatory for sm.OLS model
X_1 = sm.add_constant(testdata)
#Fitting sm.OLS model
model = sm.OLS(testlabel,X_1).fit()

#Backward Elimination
cols = list(testdata.columns)
pmax = 1
# iterate and remove feature with highest p-value for each iteration
while (len(cols)>0):
    p= []
    X_1 = testdata[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(testlabel,X_1).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols) 
    # max p value     
    pmax = max(p)
    # max p index
    feature_with_p_max = p.idxmax()
    # if p-value is more than 0.05, remove
    # else, stop the loop
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break

# results
selected_feat = cols
temp = t[selected_feat]
print ("Number of selected features: ", len(selected_feat))
print ("Features: ",selected_feat)

print("BACKWARD ELIMINATION METHOD")
print("===========================")
print("knn: ",models.feat_ex_KNN(temp,l))
print("random forest: ",models.feat_ex_RFC(temp,l))
print("extra trees: ",models.feat_ex_ETC(temp,l))


                #-----------------------------
                # GREEDY FEATURE SELECTION
                #-----------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV

estimator = LogisticRegression(multi_class='auto', solver ='lbfgs')
rfecv= RFECV(estimator, cv=10)
rfecv.fit(t,l)

selected_feat = testdata.columns[(rfecv.support_)]
temp = t[selected_feat]

print ("Number of selected features: ", len(selected_feat))
print ("Features: ",list(selected_feat))

print("GREEDY FEATURE SELECTION")
print("===========================")
print("knn: ",models.feat_ex_KNN(temp,l))
print("random forest: ",models.feat_ex_RFC(temp,l))
print("extra trees: ",models.feat_ex_ETC(temp,l))
        
       
        
                #-----------------------------------
                # L1 BASED (LASSO) FEATURE SELECTION
                #-----------------------------------

from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel

sel_ = SelectFromModel(LinearSVC(C=0.05, penalty="l1", dual=False))
sel_.fit(testdata, testlabel)

selected_feat = testdata.columns[(sel_.get_support())]
temp = t[list(selected_feat)]
print("L1 BASED (LASSO) FEATURE SELECTION")
print("==================================")
print("knn: ",models.feat_ex_KNN(temp,l))
print("random forest: ",models.feat_ex_RFC(temp,l))
print("extra trees: ",models.feat_ex_ETC(temp,l))
print('total features: {}'.format((testdata.shape[1])))

print('Number of selected features: {}'.format(len(selected_feat)))
print('Features: {}'.format(list(selected_feat)))
print('features with coefficients shrank to zero: {}'.format(np.sum(sel_.estimator_.coef_ == 0)))




                #-----------------------------------
                # L2 BASED (RIDGE) FEATURE SELECTION
                #-----------------------------------

sel_ = SelectFromModel(LinearSVC(C=0.05, penalty="l2", dual=False))
sel_.fit(testdata, testlabel)

selected_feat = testdata.columns[(sel_.get_support())]
temp = t[list(selected_feat)]
print("L2 BASED (RIDGE) FEATURE SELECTION")
print("==================================")
print("knn: ",models.feat_ex_KNN(temp,l))
print("random forest: ",models.feat_ex_RFC(temp,l))
print("extra trees: ",models.feat_ex_ETC(temp,l))

print('Number of selected features: {}'.format(len(selected_feat)))
print('Features: {}'.format(list(selected_feat)))
print('features with coefficients shrank to zero: {}'.format(np.sum(sel_.estimator_.coef_ == 0)))


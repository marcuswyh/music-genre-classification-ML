from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.neighbors import NearestCentroid

from sklearn.model_selection import cross_val_score

folds = 10


def knn(test, label):
    knn = KNeighborsClassifier()
    scores = cross_val_score(knn, test, label, cv=folds)
    return scores

def nearestCentroid(test, label):
    nc = NearestCentroid()
    scores = cross_val_score(nc, test, label, cv=folds)
    return scores

def svm(test, label):
    svm = SVC(gamma='auto')
    scores = cross_val_score(svm, test, label, cv=folds)
    return scores

def decisionTree(test, label):
    clf = DecisionTreeClassifier(random_state=0)
    scores = cross_val_score(clf, test, label, cv=folds)
    return scores
    
def naiveBayes(test, label):
    clf = GaussianNB()
    scores = cross_val_score(clf, test, label, cv=folds)
    return scores

def randomForest(test, label):
    clf = RandomForestClassifier(n_estimators=100, random_state=0)
    scores = cross_val_score(clf, test, label, cv=folds)
    return scores

def extraTrees(test, label):
    clf = ExtraTreesClassifier(n_estimators=100, random_state=0)
    scores = cross_val_score(clf, test, label, cv=folds)
    return scores

def extraTree(test, label):
    clf = ExtraTreeClassifier(random_state=0)
    scores = cross_val_score(clf, test, label, cv=folds)
    return scores

#===========================================
#  3 best performing models with best params
#===========================================
def feat_ex_KNN(test,label):
    knn = KNeighborsClassifier( algorithm='ball_tree', n_neighbors= 7, p= 2, weights= 'distance')
    scores = cross_val_score(knn, test, label, cv=folds)
    return scores.mean()

def feat_ex_RFC(test,label):
    rfc = RandomForestClassifier(criterion= 'entropy', 
                			 max_depth= 10, 
                			 max_features= 'log2', 
                			 n_estimators= 200,
                			 random_state= 0)
    scores = cross_val_score(rfc, test, label, cv=folds)
    return scores.mean()

def feat_ex_ETC(test,label):
    etc = ExtraTreesClassifier(criterion= 'gini', 
                    			 max_depth= 17, 
                    			 max_features= 'sqrt', 
                    			 n_estimators= 300,
                    			 random_state= 0)
    scores = cross_val_score(etc, test, label, cv=folds)
    return scores.mean()


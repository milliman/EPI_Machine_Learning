# -*- coding: utf-8 -*-
"""
Created on Sat Jan 07 13:18:10 2017

@author: feng.han


SMOTE resampling then train RF model

"""



import scipy as sp
from sklearn.svm import SVC
import pandas as pd
import numpy as np
from scipy.stats import randint as sp_randint
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import RandomizedSearchCV
from time import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
#NEED INSTALL IMBLANCE PACKAGE
#code: conda install -c glemaitre imbalanced-learn

from imblearn.over_sampling import SMOTE
#read tables
train=pd.read_table('H:/0026ABV55-02_AbbVie Creon/Work Papers/SAS_Code/Machine Learning/20161227/training_self1.txt')
test=pd.read_table('H:/0026ABV55-02_AbbVie Creon/Work Papers/SAS_Code/Machine Learning/20161227/testing_self1.txt')



train_Ycat=train['Ycat']
train_PERT=train['PERT_Y']
train=train.drop('Ycat',1)
train=train.drop('MemberID',1)
train=train.drop('PERT_Y',1)

test_Ycat=test['Ycat']
test_PERT=test['PERT_Y']
test=test.drop('Ycat',1)
test=test.drop('PERT_Y',1)
test=test.drop('MemberID',1)

scaler = MinMaxScaler()
train = scaler.fit_transform(train)
test =  scaler.transform(test)

train_std=pd.DataFrame(train)
test_std=pd.DataFrame(test)

res=[1 if x =='Y' else 0 for x in train_PERT]


sm = SMOTE(random_state=1234)
X_res, y_res = sm.fit_sample(train_std, res)

#model1
rf=RandomForestClassifier(n_estimators=100)
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

#hyper parameter options, could also apply to other model. 
param_dist = {"max_depth": [3, None],
              "max_features": sp_randint(1, 20),
              "min_samples_split": sp_randint(2, 11),
              "min_samples_leaf": sp_randint(1, 11),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

#serrch 50 times            
n_iter_search = 20
random_search = RandomizedSearchCV(rf, param_distributions=param_dist,
                                   n_iter=n_iter_search,scoring ='f1')

start = time()
random_search.fit(X_res,y_res )
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))

report(random_search.cv_results_)

#RandomizedSearchCV took 13241.21 seconds for 20 candidates parameter settings.
#Model with rank: 1
#Mean validation score: 0.998 (std: 0.002)
#Parameters: {'bootstrap': True, 'min_samples_leaf': 1, 'min_samples_split': 6, 'criterion': 'entropy', 'max_features': 3, 'max_depth': None}

#Model with rank: 2
#Mean validation score: 0.998 (std: 0.002)
#Parameters: {'bootstrap': True, 'min_samples_leaf': 3, 'min_samples_split': 2, 'criterion': 'entropy', 'max_features': 1, 'max_depth': None}

#Model with rank: 3
#Mean validation score: 0.998 (std: 0.002)
#Parameters: {'bootstrap': False, 'min_samples_leaf': 3, 'min_samples_split': 4, 'criterion': 'entropy', 'max_features': 10, 'max_depth': None}

rf=RandomForestClassifier(n_estimators=500)
rf=RandomForestClassifier(criterion='entropy', max_depth=None, min_samples_split=6, 
                       min_samples_leaf=1,max_features=3,bootstrap=True,random_state=1234)

#fit the model
rf.fit(X_res,y_res)

predicted_probs = rf.predict_proba(train_std)
a= predicted_probs[:,1]
np.mean(a)
b= predicted_probs[:,0]
np.mean(b)
a1=sorted(a,reverse=True)
a2=pd.DataFrame(a1)
a2.iloc[6679,:]


test_prob = rf.predict_proba(test_std)
test_prob1=test_prob[:,1]
res=['Y' if x >= 0.116667 else 'N' for x in test_prob1]
res1=pd.DataFrame(res)
columns=["res"]
res1.columns=columns
test_prob1=pd.DataFrame(test_prob1)
columns=["prob"]
test_prob1.columns=columns
frame=[test_std,test_prob1,res1,test_PERT,test_Ycat]
test1 = pd.concat(frame,axis=1)

cm=confusion_matrix(test1.PERT_Y, test1.res,labels=["Y","N"])
a=cm[0,0]+0.0
b=cm[0,0]+cm[0,1]+0.0
c=cm[1,0]+cm[0,0]+0.0
d=cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1]+0.0
PU_score=(a/b)**2/(c/d)

print PU_score
print c/d
#score for labeled test data
test2=test1[test1.Ycat > -1]
cm=confusion_matrix(test2.PERT_Y, test2.res,labels=["Y","N"])
report = classification_report(test2.PERT_Y, test2.res,labels=["Y","N"])
brier=brier_score_loss(test2.PERT_Y, test2.prob,pos_label="Y")

#ROC
t=[1 if x =='Y' else 0 for x in test2.PERT_Y]
roc=roc_auc_score(t,test2.prob)

print PU_score
print cm
print report
print roc
print brier


#PU_score 9.07205403433
#f1 score 0.71 recall .59 roc .779195463406 brier .440107933464




rf=RandomForestClassifier(n_estimators=100)
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

#hyper parameter options, could also apply to other model. 
param_dist = {"max_depth": [3, None],
              "max_features": sp_randint(1, 20),
              "min_samples_split": sp_randint(2, 11),
              "min_samples_leaf": sp_randint(1, 11),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

#serrch 50 times            
n_iter_search = 20
random_search = RandomizedSearchCV(rf, param_distributions=param_dist,
                                   n_iter=n_iter_search)

start = time()
random_search.fit(X_res,y_res )
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))

report(random_search.cv_results_)


#
#RandomizedSearchCV took 11434.88 seconds for 20 candidates parameter settings.
#Model with rank: 1
#Mean validation score: 0.998 (std: 0.002)
#Parameters: {'bootstrap': True, 'min_samples_leaf': 3, 'min_samples_split': 9, 'criterion': 'entropy', 'max_features': 2, 'max_depth': None}

#Model with rank: 2
#Mean validation score: 0.998 (std: 0.002)
#Parameters: {'bootstrap': False, 'min_samples_leaf': 4, 'min_samples_split': 9, 'criterion': 'entropy', 'max_features': 17, 'max_depth': None}

#Model with rank: 3
#Mean validation score: 0.997 (std: 0.002)
#Parameters: {'bootstrap': False, 'min_samples_leaf': 8, 'min_samples_split': 3, 'criterion': 'entropy', 'max_features': 13, 'max_depth': None}



rf=RandomForestClassifier(n_estimators=500)
rf=RandomForestClassifier(criterion='entropy', max_depth=None, min_samples_split=9, 
                       min_samples_leaf=3,max_features=2,bootstrap=True,random_state=1234)

#fit the model
rf.fit(X_res,y_res)

predicted_probs = rf.predict_proba(train_std)
a= predicted_probs[:,1]
np.mean(a)
b= predicted_probs[:,0]
np.mean(b)
a1=sorted(a,reverse=True)
a2=pd.DataFrame(a1)
a2.iloc[6679,:]


test_prob = rf.predict_proba(test_std)
test_prob1=test_prob[:,1]
res=['Y' if x >= 0.203235 else 'N' for x in test_prob1]
res1=pd.DataFrame(res)
columns=["res"]
res1.columns=columns
test_prob1=pd.DataFrame(test_prob1)
columns=["prob"]
test_prob1.columns=columns
frame=[test_std,test_prob1,res1,test_PERT,test_Ycat]
test1 = pd.concat(frame,axis=1)

cm=confusion_matrix(test1.PERT_Y, test1.res,labels=["Y","N"])
a=cm[0,0]+0.0
b=cm[0,0]+cm[0,1]+0.0
c=cm[1,0]+cm[0,0]+0.0
d=cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1]+0.0
PU_score=(a/b)**2/(c/d)

print PU_score
print c/d
#score for labeled test data
test2=test1[test1.Ycat > -1]
cm=confusion_matrix(test2.PERT_Y, test2.res,labels=["Y","N"])
report = classification_report(test2.PERT_Y, test2.res,labels=["Y","N"])
brier=brier_score_loss(test2.PERT_Y, test2.prob,pos_label="Y")

#ROC
t=[1 if x =='Y' else 0 for x in test2.PERT_Y]
roc=roc_auc_score(t,test2.prob)

print PU_score
print cm
print report
print roc
print brier


#PU_score 8.02059061168
#f1 score .63 recall .49 roc 0.756274366472 brier .416605913696


rf=RandomForestClassifier(n_estimators=100)
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

#hyper parameter options, could also apply to other model. 
param_dist = {"max_depth": [3, None],
              "max_features": sp_randint(1, 20),
              "min_samples_split": sp_randint(2, 11),
              "min_samples_leaf": sp_randint(1, 11),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

#serrch 50 times            
n_iter_search = 20
random_search = RandomizedSearchCV(rf, param_distributions=param_dist,
                                   n_iter=n_iter_search,scoring ='recall')

start = time()
random_search.fit(X_res,y_res )
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))

report(random_search.cv_results_)

#Model with rank: 1
#Mean validation score: 0.983 (std: 0.022)
#Parameters: {'bootstrap': False, 'min_samples_leaf': 3, 'min_samples_split': 4, 'criterion': 'gini', 'max_features': 13, 'max_depth': None}

#Model with rank: 2
#Mean validation score: 0.983 (std: 0.022)
#Parameters: {'bootstrap': False, 'min_samples_leaf': 3, 'min_samples_split': 7, 'criterion': 'gini', 'max_features': 4, 'max_depth': None}

#Model with rank: 3
#Mean validation score: 0.983 (std: 0.021)
#Parameters: {'bootstrap': False, 'min_samples_leaf': 7, 'min_samples_split': 10, 'criterion': 'entropy', 'max_features': 6, 'max_depth': None}



rf=RandomForestClassifier(n_estimators=500)
rf=RandomForestClassifier(criterion='entropy', max_depth=None, min_samples_split=10, 
                       min_samples_leaf=7,max_features=6,bootstrap=False,random_state=1234)

#fit the model
rf.fit(X_res,y_res)

predicted_probs = rf.predict_proba(train_std)
a= predicted_probs[:,1]
np.mean(a)
b= predicted_probs[:,0]
np.mean(b)
a1=sorted(a,reverse=True)
a2=pd.DataFrame(a1)
a2.iloc[6679,:]


test_prob = rf.predict_proba(test_std)
test_prob1=test_prob[:,1]
res=['Y' if x >= 0.135714 else 'N' for x in test_prob1]
res1=pd.DataFrame(res)
columns=["res"]
res1.columns=columns
test_prob1=pd.DataFrame(test_prob1)
columns=["prob"]
test_prob1.columns=columns
frame=[test_std,test_prob1,res1,test_PERT,test_Ycat]
test1 = pd.concat(frame,axis=1)

cm=confusion_matrix(test1.PERT_Y, test1.res,labels=["Y","N"])
a=cm[0,0]+0.0
b=cm[0,0]+cm[0,1]+0.0
c=cm[1,0]+cm[0,0]+0.0
d=cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1]+0.0
PU_score=(a/b)**2/(c/d)

print PU_score
print c/d
#score for labeled test data
test2=test1[test1.Ycat > -1]
cm=confusion_matrix(test2.PERT_Y, test2.res,labels=["Y","N"])
report = classification_report(test2.PERT_Y, test2.res,labels=["Y","N"])
brier=brier_score_loss(test2.PERT_Y, test2.prob,pos_label="Y")

#ROC
t=[1 if x =='Y' else 0 for x in test2.PERT_Y]
roc=roc_auc_score(t,test2.prob)

print PU_score
print cm
print report
print roc
print brier

#PU_score 12.0295781462
#f1 score .72 recall .60 roc 0.791594674818 brier 0.426523017981
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 07 13:18:10 2017

@author: feng.han


SMOTE resampling then train RF model

"""


from sklearn.ensemble import GradientBoostingClassifier
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


sm = SMOTE(random_state=1234,ratio=.2)
X_res, y_res = sm.fit_sample(train_std, res)

#model1


gbc=GradientBoostingClassifier(random_state=1234)
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
param_dist = {"loss": ['deviance', 'exponential'],
              "learning_rate": sp.stats.expon(scale=.1),
              "min_samples_split": sp_randint(2, 11),
              "min_samples_leaf": sp_randint(1, 11),
              "max_depth": sp_randint(1, 5),
              "max_features": ['auto', 'sqrt','log2']}

#serrch 50 times            
n_iter_search = 20
random_search = RandomizedSearchCV(gbc, param_distributions=param_dist,
                                   n_iter=n_iter_search,scoring ='recall')

start = time()
random_search.fit(X_res, y_res)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))

report(random_search.cv_results_)


#Model with rank: 1
#Mean validation score: 0.974 (std: 0.033)
#Parameters: {'loss': 'deviance', 'learning_rate': 0.31124683564481409, 'min_samples_leaf': 3, 'min_samples_split': 10, 'max_features': 'auto', 'max_depth': 4}

#Model with rank: 2
#Mean validation score: 0.961 (std: 0.030)
#Parameters: {'loss': 'exponential', 'learning_rate': 0.23321819950083111, 'min_samples_leaf': 3, 'min_samples_split': 5, 'max_features': 'sqrt', 'max_depth': 4}

#Model with rank: 3
#Mean validation score: 0.951 (std: 0.028)
#Parameters: {'loss': 'deviance', 'learning_rate': 0.53130556264074225, 'min_samples_leaf': 4, 'min_samples_split': 2, 'max_features': 'sqrt', 'max_depth': 2}


gbc=GradientBoostingClassifier(random_state=1234,loss='deviance',learning_rate=
                                0.31124683564481409,min_samples_leaf=3,
                                min_samples_split=10,max_features='auto',max_depth=4)

#fit the model
gbc.fit(X_res,y_res)

predicted_probs = gbc.predict_proba(train_std)
a= predicted_probs[:,1]
np.mean(a)
b= predicted_probs[:,0]
np.mean(b)
a1=sorted(a,reverse=True)
a2=pd.DataFrame(a1)
a2.iloc[6679,:]


test_prob = gbc.predict_proba(test_std)
test_prob1=test_prob[:,1]
res=['Y' if x >= 0.077149 else 'N' for x in test_prob1]
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


#PU_score 14.8706779629
#f1 score 0.68  recall 0.51 roc 0.843655856814 brier 0.494591183748



#model 2


gbc=GradientBoostingClassifier(random_state=1234)
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
param_dist = {"loss": ['deviance', 'exponential'],
              "learning_rate": sp.stats.expon(scale=.1),
              "min_samples_split": sp_randint(2, 11),
              "min_samples_leaf": sp_randint(1, 11),
              "max_depth": sp_randint(1, 5),
              "max_features": ['auto', 'sqrt','log2']}

#serrch 50 times            
n_iter_search = 20
random_search = RandomizedSearchCV(gbc, param_distributions=param_dist,
                                   n_iter=n_iter_search,scoring ='f1')

start = time()
random_search.fit(X_res, y_res)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))

report(random_search.cv_results_)


#RandomizedSearchCV took 6659.39 seconds for 20 candidates parameter settings.
#Model with rank: 1
#Mean validation score: 0.982 (std: 0.014)
#Parameters: {'loss': 'exponential', 'learning_rate': 0.35102594485401917, 'min_samples_leaf': 2, 'min_samples_split': 5, 'max_features': 'auto', 'max_depth': 4}

#Model with rank: 2
#Mean validation score: 0.972 (std: 0.013)
#Parameters: {'loss': 'deviance', 'learning_rate': 0.26701919308395144, 'min_samples_leaf': 8, 'min_samples_split': 10, 'max_features': 'auto', 'max_depth': 2}

#Model with rank: 3
#Mean validation score: 0.967 (std: 0.013)
#Parameters: {'loss': 'exponential', 'learning_rate': 0.15010593637125785, 'min_samples_leaf': 4, 'min_samples_split': 10, 'max_features': 'auto', 'max_depth': 3}

gbc=GradientBoostingClassifier(random_state=1234,loss='exponential',learning_rate=
                                0.35102594485401917,min_samples_leaf=2,
                                min_samples_split=5,max_features='auto',max_depth=4)

#fit the model
gbc.fit(X_res,y_res)

predicted_probs = gbc.predict_proba(train_std)
a= predicted_probs[:,1]
np.mean(a)
b= predicted_probs[:,0]
np.mean(b)
a1=sorted(a,reverse=True)
a2=pd.DataFrame(a1)
a2.iloc[6679,:]


test_prob = gbc.predict_proba(test_std)
test_prob1=test_prob[:,1]
res=['Y' if x >= 0.070143 else 'N' for x in test_prob1]
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


#PU_score 13.968855493
#f1 score 0.66  recall .52 roc 0.8439105972 brier 0.474499987757 


from sklearn.metrics import precision_recall_fscore_support,recall_score, make_scorer,precision_score

def pu(actual,prediction):
    precision = precision_score(actual, prediction)
    recall = recall_score(actual, prediction)
    pu= precision*recall
    return pu
pu_score=make_scorer(pu,greater_is_better=True)


gbc=GradientBoostingClassifier(random_state=1234)
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
param_dist = {"loss": ['deviance', 'exponential'],
              "learning_rate": sp.stats.expon(scale=.1),
              "min_samples_split": sp_randint(2, 11),
              "min_samples_leaf": sp_randint(1, 11),
              "max_depth": sp_randint(1, 5),
              "max_features": ['auto', 'sqrt','log2']}

#serrch 50 times            
n_iter_search = 20
random_search = RandomizedSearchCV(gbc, param_distributions=param_dist,
                                   n_iter=n_iter_search,scoring =pu_score)

start = time()
random_search.fit(X_res, y_res)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))

report(random_search.cv_results_)

#Model with rank: 1
#Mean validation score: 0.918 (std: 0.019)
#Parameters: {'loss': 'deviance', 'learning_rate': 0.14236732939674285, 'min_samples_leaf': 8, 'min_samples_split': 5, 'max_features': 'log2', 'max_depth': 4}

#Model with rank: 2
#Mean validation score: 0.909 (std: 0.021)
#Parameters: {'loss': 'deviance', 'learning_rate': 0.1037442652897735, 'min_samples_leaf': 6, 'min_samples_split': 10, 'max_features': 'sqrt', 'max_depth': 4}

#Model with rank: 3
#Mean validation score: 0.867 (std: 0.011)
#Parameters: {'loss': 'deviance', 'learning_rate': 0.050528133567381978, 'min_samples_leaf': 3, 'min_samples_split': 10, 'max_features': 'auto', 'max_depth': 3}

gbc=GradientBoostingClassifier(random_state=1234,loss='deviance',learning_rate=
                                0.14236732939674285,min_samples_leaf=8,
                                min_samples_split=5,max_features='log2',max_depth=4)

#fit the model
gbc.fit(X_res,y_res)

predicted_probs = gbc.predict_proba(train_std)
a= predicted_probs[:,1]
np.mean(a)
b= predicted_probs[:,0]
np.mean(b)
a1=sorted(a,reverse=True)
a2=pd.DataFrame(a1)
a2.iloc[6679,:]


test_prob = gbc.predict_proba(test_std)
test_prob1=test_prob[:,1]
res=['Y' if x >= 0.259176 else 'N' for x in test_prob1]
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


#PU_score 11.3279136071
#f1 score 0.61  recall .46 roc 0.821194400142 brier 0.379540702591 

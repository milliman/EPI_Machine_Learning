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


sm = SMOTE(random_state=1234)
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
#Mean validation score: 0.993 (std: 0.005)
#Parameters: {'loss': 'deviance', 'learning_rate': 0.19316695662272315, 'min_samples_leaf': 2, 'min_samples_split': 2, 'max_features': 'sqrt', 'max_depth': 4}

#Model with rank: 2
#Mean validation score: 0.991 (std: 0.005)
#Parameters: {'loss': 'exponential', 'learning_rate': 0.15480248319791112, 'min_samples_leaf': 2, 'min_samples_split': 10, 'max_features': 'auto', 'max_depth': 3}

#Model with rank: 3
#Mean validation score: 0.981 (std: 0.004)
#Parameters: {'loss': 'exponential', 'learning_rate': 0.16802528178707174, 'min_samples_leaf': 7, 'min_samples_split': 4, 'max_features': 'sqrt', 'max_depth': 3}

gbc=GradientBoostingClassifier(random_state=1234,loss='deviance',learning_rate=
                                0.19316695662272315,min_samples_leaf=2,
                                min_samples_split=2,max_features='sqrt',max_depth=4)

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
res=['Y' if x >= 0.326596 else 'N' for x in test_prob1]
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


#PU_score 11.6531584916
#f1 score 0.62  recall 0.47 roc 0.820153287259 brier 0.35434336598



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


#RandomizedSearchCV took 4363.86 seconds for 20 candidates parameter settings.
#Model with rank: 1
#Mean validation score: 0.979 (std: 0.001)
#Parameters: {'loss': 'exponential', 'learning_rate': 0.10515520115364881, 'min_samples_leaf': 5, 'min_samples_split': 10, 'max_features': 'sqrt', 'max_depth': 4}

#Model with rank: 2
#Mean validation score: 0.964 (std: 0.001)
#Parameters: {'loss': 'exponential', 'learning_rate': 0.048742041855919765, 'min_samples_leaf': 8, 'min_samples_split': 7, 'max_features': 'log2', 'max_depth': 4}

#Model with rank: 3
#Mean validation score: 0.963 (std: 0.001)
#Parameters: {'loss': 'deviance', 'learning_rate': 0.12395644775280323, 'min_samples_leaf': 1, 'min_samples_split': 8, 'max_features': 'sqrt', 'max_depth': 2}



gbc=GradientBoostingClassifier(random_state=1234,loss='exponential',learning_rate=
                                0.10515520115364881,min_samples_leaf=5,
                                min_samples_split=10,max_features='sqrt',max_depth=4)

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
res=['Y' if x >= 0.615038 else 'N' for x in test_prob1]
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


#PU_score 10.3382604311
#f1 score 0.59  recall .44 roc 0.816863813574 brier 0.262999118643 




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
                                   n_iter=n_iter_search)

start = time()
random_search.fit(X_res, y_res)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))

report(random_search.cv_results_)

#RandomizedSearchCV took 6998.14 seconds for 20 candidates parameter settings.
#Model with rank: 1
#Mean validation score: 0.996 (std: 0.003)
#Parameters: {'loss': 'deviance', 'learning_rate': 0.33962763970174581, 'min_samples_leaf': 1, 'min_samples_split': 10, 'max_features': 'auto', 'max_depth': 3}

#Model with rank: 2
#Mean validation score: 0.996 (std: 0.003)
#Parameters: {'loss': 'deviance', 'learning_rate': 0.32345211319150868, 'min_samples_leaf': 7, 'min_samples_split': 8, 'max_features': 'auto', 'max_depth': 3}

#Model with rank: 3
#Mean validation score: 0.996 (std: 0.002)
#Parameters: {'loss': 'deviance', 'learning_rate': 0.43169680002123806, 'min_samples_leaf': 10, 'min_samples_split': 10, 'max_features': 'sqrt', 'max_depth': 4}



gbc=GradientBoostingClassifier(random_state=1234,loss='deviance',learning_rate=
                                0.33962763970174581,min_samples_leaf=1,
                                min_samples_split=10,max_features='auto',max_depth=3)

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
res=['Y' if x >= 0.131685 else 'N' for x in test_prob1]
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


#PU_score 13.6229263213
#f1 score 0.66  recall .51 roc 0.846513379408 brier 0.46211109245 

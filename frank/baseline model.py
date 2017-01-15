
"""
Spyder Editor

"""
#import module
import pandas as pd
import numpy as np
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import RandomizedSearchCV
from time import time

#read table: train as total training dataset, train_re as resampled training
#dataset for modeling,test as test dataset

train=pd.read_table('H:/0026ABV55-02_AbbVie Creon/Work Papers/SAS_Code/Machine Learning/20161227/training.txt')
train_re=pd.read_table('H:/0026ABV55-02_AbbVie Creon/Work Papers/SAS_Code/Machine Learning/20161227/training_re.txt')
test=pd.read_table('H:/0026ABV55-02_AbbVie Creon/Work Papers/SAS_Code/Machine Learning/20161227/testing.txt')

#extract rsponse column
target = train_re.iloc[:,[293]]

#extract train matrix
tr = train_re.iloc[:,1:293]

#extract test matrix
tt = test.iloc[:,2:294]

#extract response variable for model, strange data type for python 
target=target['PERT_Y']

#set random forest tree as 500
rf=RandomForestClassifier(n_estimators=500)
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
n_iter_search = 50
random_search = RandomizedSearchCV(rf, param_distributions=param_dist,
                                   n_iter=n_iter_search)

start = time()
random_search.fit(tr,target )
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))

report(random_search.cv_results_)

""""""""

""""""""
#Model with rank: 1
#Mean validation score: 0.903 (std: 0.013)
#Parameters: {'bootstrap': False, 'min_samples_leaf': 1, 'min_samples_split': 8, 'criterion': 'entropy', 'max_features': 19, 'max_depth': None}
#build random forest model after random search, everythime result will be change.
rf=RandomForestClassifier(criterion='entropy', max_depth=None, min_samples_split=8, 
                       min_samples_leaf=1,max_features=19,bootstrap=False,random_state=1234)

#fit the model
rf.fit(tr, target)

#apply to original train data from resample train data model result
tr1 = train.iloc[:,1:293]
predicted_probs1 = rf.predict_proba(tr1)
a= predicted_probs1[:,1]
np.mean(a)
b= predicted_probs1[:,0]
np.mean(b)
a1=sorted(a, reverse=True)
a2=pd.DataFrame(a1)

#get 1670*4=6680
pd.crosstab(index=train["PERT_Y"],columns="count")
a2.iloc[6679,:]
cr=a2.iloc[6679,:]
print cr


#score test data and get PU score
predicted_probs_test = rf.predict_proba(tt)
test_prob= predicted_probs_test[:,1]
res=['Y' if x >= 0.664286 else 'N' for x in test_prob]
res1=pd.DataFrame(res)
columns=["res"]
res1.columns=columns

test_prob1=pd.DataFrame(test_prob)
columns=["prob"]
test_prob1.columns=columns
frame=[test,test_prob1,res1]
test1 = pd.concat(frame,axis=1)

cm=confusion_matrix(test1.PERT_Y, test1.res,labels=["Y","N"])
a=cm[0,0]+0.0
b=cm[0,0]+cm[0,1]+0.0
c=cm[1,0]+cm[0,0]+0.0
d=cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1]+0.0
PU_score=(a/b)**2/(c/d)

pd.crosstab(test1.Ycat,test1.res)

c/d

#score for labeled test data
test2=test1[test1.Ycat > -1]
cm=confusion_matrix(test2.PERT_Y, test2.res,labels=["Y","N"])
report = classification_report(test2.PERT_Y, test2.res,labels=["Y","N"])
brier=brier_score_loss(test2.PERT_Y, test2.prob,pos_label="Y")

#ROC
t=[1 if x =='Y' else 0 for x in test2.PERT_Y]
roc=roc_auc_score(t,test2.prob)

#output
#PU score 10.3781152768 f1 .58 roc .904 brier .1889 
print PU_score
print cm
print report
print roc
print brier



#Output Variable importance
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
tr1=tr.iloc[:,indices]
output=pd.DataFrame({'name':tr1.columns.values,'value':importances[indices]})
output.to_excel('H:/0026ABV55-02_AbbVie Creon/Work Papers/SAS_Code/Machine Learning/20161227/varimp0105.xlsx', sheet_name='sheet1', index=False)

#future question: should we set all cut off point to max PU score?
#we might get different ratio than 25%.
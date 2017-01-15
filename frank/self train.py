
"""
Spyder Editor






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
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import  SMOTE
from sklearn.naive_bayes import GaussianNB
#NEED INSTALL IMBLANCE PACKAGE
#code: conda install -c glemaitre imbalanced-learn


#read tables
train=pd.read_table('H:/0026ABV55-02_AbbVie Creon/Work Papers/SAS_Code/Machine Learning/20161227/training_self1.txt')
test=pd.read_table('H:/0026ABV55-02_AbbVie Creon/Work Papers/SAS_Code/Machine Learning/20161227/testing_self1.txt')

#drop columns and for standize
train_Ycat=train['Ycat']
train_PERT=train['PERT_Y']
train=train.drop('Ycat',1)
train=train.drop('PERT_Y',1)
train=train.drop('MemberID',1)

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

#split label and unlabel
frame=[train_std,train_Ycat,train_PERT]
train1 = pd.concat(frame,axis=1)
train_label=train1[train1.Ycat > -1]
train_unlabel=train1[train1.Ycat == -1]

#train matrix
target = train_label.iloc[:,[52]]
Y=target['PERT_Y']
tr = train_label.iloc[:,0:51]

sm = SMOTE(random_state=1234)
X_res, y_res = sm.fit_sample(tr, Y)

gnb = GaussianNB()
gnb.fit(tr, Y)

tr_un = train_unlabel.iloc[:,0:51]
predicted_unlabel = gnb.predict(tr_un)


predicted_unlabel=pd.DataFrame(predicted_unlabel)
columns=["PERT_Y"]
predicted_unlabel.columns=columns
frame=[tr_un,predicted_unlabel]
tr_un1 = pd.concat(frame,axis=1)



train_label=train_label.drop('Ycat',1)
train_unlabel1=tr_un1[tr_un1.PERT_Y == "Y"]
train_label1=tr_un1[tr_un1.PERT_Y == "N"]
frames=[train_label,train_label1]
train_label2 = pd.concat(frames)


target = train_label2.iloc[:,[51]]
Y=target["PERT_Y"]
tr=train_label2.iloc[:,0:51]


#iteration 2


gnb1 = GaussianNB()
gnb1.fit(tr, Y)

tr_un = train_unlabel1.iloc[:,0:51]
predicted_unlabel = gnb1.predict(tr_un)
predicted_unlabel=pd.DataFrame(predicted_unlabel)
columns=["PERT_Y"]
predicted_unlabel.columns=columns
tr_un=tr_un.reset_index(drop=True)
frame=[tr_un,predicted_unlabel]
tr_un1 = pd.concat(frame,axis=1)

a=predicted_unlabel[predicted_unlabel.PERT_Y=='Y']
b=predicted_unlabel[predicted_unlabel.PERT_Y=='N']



train_unlabel1=tr_un1[tr_un1.PERT_Y == "Y"]
train_label3=tr_un1[tr_un1.PERT_Y == "N"]
frames=[train_label2,train_label3]
train_label4 = pd.concat(frames)


target = train_label4.iloc[:,[51]]
Y=target["PERT_Y"]
tr=train_label4.iloc[:,0:51]


#iteration 3



gnb2 = GaussianNB()
gnb2.fit(tr, Y)

tr_un = train_unlabel1.iloc[:,0:51]
predicted_unlabel = gnb2.predict(tr_un)
predicted_unlabel=pd.DataFrame(predicted_unlabel)
columns=["PERT_Y"]
predicted_unlabel.columns=columns
tr_un=tr_un.reset_index(drop=True)
frame=[tr_un,predicted_unlabel]
tr_un1 = pd.concat(frame,axis=1)

a=predicted_unlabel[predicted_unlabel.PERT_Y=='Y']
b=predicted_unlabel[predicted_unlabel.PERT_Y=='N']



train_unlabel1=tr_un1[tr_un1.PERT_Y == "Y"]
train_label5=tr_un1[tr_un1.PERT_Y == "N"]
frames=[train_label4,train_label5]
train_label6 = pd.concat(frames)


target = train_label6.iloc[:,[51]]
Y=target["PERT_Y"]
tr=train_label6.iloc[:,0:51]



#iteration 4



gnb3 = GaussianNB()
gnb3.fit(tr, Y)

tr_un = train_unlabel1.iloc[:,0:51]
predicted_unlabel = gnb3.predict(tr_un)
predicted_unlabel=pd.DataFrame(predicted_unlabel)
columns=["PERT_Y"]
predicted_unlabel.columns=columns
tr_un=tr_un.reset_index(drop=True)
frame=[tr_un,predicted_unlabel]
tr_un1 = pd.concat(frame,axis=1)

a=predicted_unlabel[predicted_unlabel.PERT_Y=='Y']
b=predicted_unlabel[predicted_unlabel.PERT_Y=='N']



train_unlabel1=tr_un1[tr_un1.PERT_Y == "Y"]
train_label7=tr_un1[tr_un1.PERT_Y == "N"]
frames=[train_label6,train_label7]
train_label8 = pd.concat(frames)


target = train_label8.iloc[:,[51]]
Y=target["PERT_Y"]
tr=train_label8.iloc[:,0:51]




#iteration 5



gnb4 = GaussianNB()
gnb4.fit(tr, Y)

tr_un = train_unlabel1.iloc[:,0:51]
predicted_unlabel = gnb4.predict(tr_un)
predicted_unlabel=pd.DataFrame(predicted_unlabel)
columns=["PERT_Y"]
predicted_unlabel.columns=columns
tr_un=tr_un.reset_index(drop=True)
frame=[tr_un,predicted_unlabel]
tr_un1 = pd.concat(frame,axis=1)

a=predicted_unlabel[predicted_unlabel.PERT_Y=='Y']
b=predicted_unlabel[predicted_unlabel.PERT_Y=='N']



train_unlabel1=tr_un1[tr_un1.PERT_Y == "Y"]
train_label9=tr_un1[tr_un1.PERT_Y == "N"]
frames=[train_label8,train_label9]
train_label10 = pd.concat(frames)


target = train_label10.iloc[:,[51]]
Y=target["PERT_Y"]
tr=train_label10.iloc[:,0:51]


#iteration 6



gnb5 = GaussianNB()
gnb5.fit(tr, Y)

tr_un = train_unlabel1.iloc[:,0:51]
predicted_unlabel = gnb5.predict(tr_un)
predicted_unlabel=pd.DataFrame(predicted_unlabel)
columns=["PERT_Y"]
predicted_unlabel.columns=columns
tr_un=tr_un.reset_index(drop=True)
frame=[tr_un,predicted_unlabel]
tr_un1 = pd.concat(frame,axis=1)

a=predicted_unlabel[predicted_unlabel.PERT_Y=='Y']
b=predicted_unlabel[predicted_unlabel.PERT_Y=='N']



train_unlabel1=tr_un1[tr_un1.PERT_Y == "Y"]
train_label11=tr_un1[tr_un1.PERT_Y == "N"]
frames=[train_label10,train_label11]
train_label12 = pd.concat(frames)


target = train_label12.iloc[:,[51]]
Y=target["PERT_Y"]
tr=train_label12.iloc[:,0:51]


#iteration 7



gnb6 = GaussianNB()
gnb6.fit(tr, Y)

tr_un = train_unlabel1.iloc[:,0:51]
predicted_unlabel = gnb6.predict(tr_un)
predicted_unlabel=pd.DataFrame(predicted_unlabel)
columns=["PERT_Y"]
predicted_unlabel.columns=columns
tr_un=tr_un.reset_index(drop=True)
frame=[tr_un,predicted_unlabel]
tr_un1 = pd.concat(frame,axis=1)

a=predicted_unlabel[predicted_unlabel.PERT_Y=='Y']
b=predicted_unlabel[predicted_unlabel.PERT_Y=='N']



train_unlabel1=tr_un1[tr_un1.PERT_Y == "Y"]
train_label13=tr_un1[tr_un1.PERT_Y == "N"]
frames=[train_label13,train_label12]
train_label14 = pd.concat(frames)


target = train_label14.iloc[:,[51]]
Y=target["PERT_Y"]
tr=train_label14.iloc[:,0:51]




#iteration 8



gnb7 = GaussianNB()
gnb7.fit(tr, Y)

tr_un = train_unlabel1.iloc[:,0:51]
predicted_unlabel = gnb7.predict(tr_un)
predicted_unlabel=pd.DataFrame(predicted_unlabel)
columns=["PERT_Y"]
predicted_unlabel.columns=columns
tr_un=tr_un.reset_index(drop=True)
frame=[tr_un,predicted_unlabel]
tr_un1 = pd.concat(frame,axis=1)

a=predicted_unlabel[predicted_unlabel.PERT_Y=='Y']
b=predicted_unlabel[predicted_unlabel.PERT_Y=='N']



train_unlabel1=tr_un1[tr_un1.PERT_Y == "Y"]
train_label15=tr_un1[tr_un1.PERT_Y == "N"]
frames=[train_label14,train_label15]
train_label16 = pd.concat(frames)


target = train_label16.iloc[:,[51]]
Y=target["PERT_Y"]
tr=train_label16.iloc[:,0:51]



#iteration 9



gnb8 = GaussianNB()
gnb8.fit(tr, Y)

tr_un = train_unlabel1.iloc[:,0:51]
predicted_unlabel = gnb8.predict(tr_un)
predicted_unlabel=pd.DataFrame(predicted_unlabel)
columns=["PERT_Y"]
predicted_unlabel.columns=columns
tr_un=tr_un.reset_index(drop=True)
frame=[tr_un,predicted_unlabel]
tr_un1 = pd.concat(frame,axis=1)

a=predicted_unlabel[predicted_unlabel.PERT_Y=='Y']
b=predicted_unlabel[predicted_unlabel.PERT_Y=='N']



train_unlabel1=tr_un1[tr_un1.PERT_Y == "Y"]
train_label17=tr_un1[tr_un1.PERT_Y == "N"]
frames=[train_label16,train_label17]
train_label18 = pd.concat(frames)


target = train_label18.iloc[:,[51]]
Y=target["PERT_Y"]
tr=train_label18.iloc[:,0:51]


#iteration 10



gnb9 = GaussianNB()
gnb9.fit(tr, Y)

tr_un = train_unlabel1.iloc[:,0:51]
predicted_unlabel = gnb9.predict(tr_un)
predicted_unlabel=pd.DataFrame(predicted_unlabel)
columns=["PERT_Y"]
predicted_unlabel.columns=columns
tr_un=tr_un.reset_index(drop=True)
frame=[tr_un,predicted_unlabel]
tr_un1 = pd.concat(frame,axis=1)

a=predicted_unlabel[predicted_unlabel.PERT_Y=='Y']
b=predicted_unlabel[predicted_unlabel.PERT_Y=='N']



train_unlabel1=tr_un1[tr_un1.PERT_Y == "Y"]
train_label19=tr_un1[tr_un1.PERT_Y == "N"]
frames=[train_label18,train_label19]
train_label20 = pd.concat(frames)


target = train_label20.iloc[:,[51]]
Y=target["PERT_Y"]
tr=train_label20.iloc[:,0:51]


#iteration 11



gnb10 = GaussianNB()
gnb10.fit(tr, Y)

tr_un = train_unlabel1.iloc[:,0:51]
predicted_unlabel = gnb10.predict(tr_un)
predicted_unlabel=pd.DataFrame(predicted_unlabel)
columns=["PERT_Y"]
predicted_unlabel.columns=columns
tr_un=tr_un.reset_index(drop=True)
frame=[tr_un,predicted_unlabel]
tr_un1 = pd.concat(frame,axis=1)

a=predicted_unlabel[predicted_unlabel.PERT_Y=='Y']
b=predicted_unlabel[predicted_unlabel.PERT_Y=='N']



train_unlabel1=tr_un1[tr_un1.PERT_Y == "Y"]
train_label21=tr_un1[tr_un1.PERT_Y == "N"]
frames=[train_label20,train_label21]
train_label22 = pd.concat(frames)


target = train_label22.iloc[:,[51]]
Y=target["PERT_Y"]
tr=train_label22.iloc[:,0:51]




#iteration 12



gnb11 = GaussianNB()
gnb11.fit(tr, Y)

tr_un = train_unlabel1.iloc[:,0:51]
predicted_unlabel = gnb11.predict(tr_un)
predicted_unlabel=pd.DataFrame(predicted_unlabel)
columns=["PERT_Y"]
predicted_unlabel.columns=columns
tr_un=tr_un.reset_index(drop=True)
frame=[tr_un,predicted_unlabel]
tr_un1 = pd.concat(frame,axis=1)

a=predicted_unlabel[predicted_unlabel.PERT_Y=='Y']
b=predicted_unlabel[predicted_unlabel.PERT_Y=='N']



train_unlabel1=tr_un1[tr_un1.PERT_Y == "Y"]
train_label23=tr_un1[tr_un1.PERT_Y == "N"]
frames=[train_label22,train_label23,train_unlabel1]
train_label24 = pd.concat(frames)


target = train_label24.iloc[:,[51]]
Y=target["PERT_Y"]
tr=train_label24.iloc[:,0:51]
"""""""""""""""""""""""""""""""""""""""""""""""""""""""
#final dataset

"""""""""""""""""""""""""""""""""""""""""""""""""""""""

train_label24.to_csv('C:/Users/feng.han/Desktop/self_train_data.txt') 
train_label24_re=pd.read_table('H:/0026ABV55-02_AbbVie Creon/Work Papers/SAS_Code/Machine Learning/20161227/self_train_data.txt',sep=',')
train_label24_re=pd.read_table('C:/Users/feng.han/Desktop/self_train_data.txt',sep=',')
train_label24=train_label24_re.drop(train_label24_re.columns[0],1)
""""""""""""""
""""""""""""""
#model 1: SVM

rf=RandomForestClassifier(n_estimators=200)
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
random_search.fit(tr,Y )
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))

report(random_search.cv_results_)


#Model with rank: 1
#Mean validation score: 0.930 (std: 0.004)
#Parameters: {'bootstrap': True, 'min_samples_leaf': 7, 'min_samples_split': 2, 'criterion': 'gini', 'max_features': 18, 'max_depth': 3}

#Model with rank: 2
#Mean validation score: 0.930 (std: 0.004)
#Parameters: {'bootstrap': True, 'min_samples_leaf': 10, 'min_samples_split': 9, 'criterion': 'gini', 'max_features': 19, 'max_depth': 3}

#Model with rank: 3
#Mean validation score: 0.928 (std: 0.003)
#Parameters: {'bootstrap': True, 'min_samples_leaf': 4, 'min_samples_split': 7, 'criterion': 'entropy', 'max_features': 13, 'max_depth': 3}



rf=RandomForestClassifier(bootstrap=True, min_samples_leaf=7, min_samples_split=2,
      criterion='gini',max_features=18,max_depth=3,random_state=1234,n_estimators=200)
rf.fit(tr, Y)


predicted_probs1 = rf.predict_proba(tr)
a= predicted_probs1[:,1]
np.mean(a)
b= predicted_probs1[:,0]
np.mean(b)
a1=sorted(a, reverse=True)
a2=pd.DataFrame(a1)
a2.iloc[44691,:]
cr=a2.iloc[44691,:]
print cr

predicted_probs_test = rf.predict_proba(test_std)
test_prob= predicted_probs_test[:,1]

res=['Y' if x >= 0.327948 else 'N' for x in test_prob]
res1=pd.DataFrame(res)
columns=["res"]
res1.columns=columns

test_prob1=pd.DataFrame(test_prob)
columns=["prob"]
test_prob1.columns=columns
frame=[test_std,test_prob1,test_PERT,test_Ycat,res1]
test1 = pd.concat(frame,axis=1)

cm=confusion_matrix(test1.PERT_Y, test1.res,labels=["Y","N"])
a=cm[0,0]+0.0
b=cm[0,0]+cm[0,1]+0.0
c=cm[1,0]+cm[0,0]+0.0
d=cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1]+0.0
PU_score=(a/b)**2/(c/d)

PU_score
cm
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

print PU_score
print cm
print report
print roc
print brier




#output
#PU score 0.72319772899 f1 .43 roc .527633794081 brier .460859623338 


#model2 Naive Bayes
gnb12 = GaussianNB()
gnb12.fit(tr, Y)


predicted_probs1 = gnb12.predict_proba(train_std)
a= predicted_probs1[:,1]
np.mean(a)
b= predicted_probs1[:,0]
np.mean(b)
a1=sorted(a, reverse=True)
a2=pd.DataFrame(a1)
a2.iloc[6679,:]
cr=a2.iloc[6679,:]
print cr

predicted_probs_test = gnb12.predict_proba(test_std)
test_prob= predicted_probs_test[:,1]

res=['Y' if x >= 1 else 'N' for x in test_prob]
res1=pd.DataFrame(res)
columns=["res"]
res1.columns=columns

test_prob1=pd.DataFrame(test_prob)
columns=["prob"]
test_prob1.columns=columns
frame=[test_std,test_prob1,test_PERT,test_Ycat,res1]
test1 = pd.concat(frame,axis=1)

cm=confusion_matrix(test1.PERT_Y, test1.res,labels=["Y","N"])
a=cm[0,0]+0.0
b=cm[0,0]+cm[0,1]+0.0
c=cm[1,0]+cm[0,0]+0.0
d=cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1]+0.0
PU_score=(a/b)**2/(c/d)

PU_score
cm
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

print PU_score
print cm
print report
print roc
print brier
print c/d

#output
#PU score 1.12496276249 f1 .43 roc .904416976785 brier .404455452298 


#model 3


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
random_search.fit(tr,Y )
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))

report(random_search.cv_results_)



#Model with rank: 1
#Mean validation score: 0.949 (std: 0.013)
#Parameters: {'loss': 'deviance', 'learning_rate': 0.041804975120693123, 'min_samples_leaf': 8, 'min_samples_split': 10, 'max_features': 'auto', 'max_depth': 4}

#Model with rank: 2
#Mean validation score: 0.947 (std: 0.008)
#Parameters: {'loss': 'exponential', 'learning_rate': 0.10896708540306849, 'min_samples_leaf': 9, 'min_samples_split': 10, 'max_features': 'sqrt', 'max_depth': 2}

#Model with rank: 3
#Mean validation score: 0.947 (std: 0.021)
#Parameters: {'loss': 'exponential', 'learning_rate': 0.14270134261083042, 'min_samples_leaf': 10, 'min_samples_split': 3, 'max_features': 'log2', 'max_depth': 3}


gbc1=GradientBoostingClassifier(random_state=1234,loss='deviance',learning_rate=
                                0.041804975120693123,min_samples_leaf=9,
                                min_samples_split=10,max_features='auto',max_depth=4)
gbc1.fit(tr, Y)


predicted_probs1 = gbc1.predict_proba(train_std)
a= predicted_probs1[:,1]
np.mean(a)
b= predicted_probs1[:,0]
np.mean(b)
a1=sorted(a, reverse=True)
a2=pd.DataFrame(a1)
a2.iloc[6679,:]
cr=a2.iloc[6679,:]
print cr

predicted_probs_test = gbc1.predict_proba(test_std)
test_prob= predicted_probs_test[:,1]

res=['Y' if x >= 0.96566 else 'N' for x in test_prob]
res1=pd.DataFrame(res)
columns=["res"]
res1.columns=columns

test_prob1=pd.DataFrame(test_prob)
columns=["prob"]
test_prob1.columns=columns
frame=[test_std,test_prob1,test_PERT,test_Ycat,res1]
test1 = pd.concat(frame,axis=1)

cm=confusion_matrix(test1.PERT_Y, test1.res,labels=["Y","N"])
a=cm[0,0]+0.0
b=cm[0,0]+cm[0,1]+0.0
c=cm[1,0]+cm[0,0]+0.0
d=cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1]+0.0
PU_score=(a/b)**2/(c/d)

PU_score
cm
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

print PU_score
print cm
print report
print roc
print brier
print c/d


#PU score 0.637647545552 f1 .2 roc 0.648646553252 brier 0.415111013784 







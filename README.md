# Applying Machine Learning Techniques to Identify Undiagnosed Patients with Exocrine Pancreatic Insufficiency

This material is part of the Appendix to **_[manuscript]_**. Please refer to that article for important information about this material and its application. 

### Disclaimer
This repository is prepared for educational or research purposes only. You may not use this repository for any illegal or unethical purpose; including activities which would give rise to criminal or civil liability. 

### License
Copyright (c) 2018 Milliman, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. 

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


### What is this repository for?

 This repository is used to train, evaluate, and use models to predict prevalence of EPI
 in a medical claims database.  It also contains additions to the Scikit-Learn Python library
 to experiment with oversampling / undersampling techniques for large class imbalances,
 nested cross validation, metrics for use with unlabeled data, and random search for hyper-parameters
 using multiple metrics at the same time.


### Requirements

The `environment.yml` file describes the environment needed to run the code.  Please use:
```
conda env create -f environment.yml
```
from Anaconda to create a compatible runtime environment for this project.

Please see [input data format](#Input-data-format) for a description of the data used to train models in this project.

### Main entry-point to use best models

Clone the repository and start using `epiml.epimlnmain.py` with data generated into a tab separated value file.
CSV files would also suffice as long as a comma (_,_) is passed into the `sep` parameter of `generate_trained_model`.

Please see `Epimlmain Example.ipynb` for an example of how to start using the library.

### Python File Descriptions
File | Description
---|---
`epimlmain.py` | Main entrypoint to train, save, load, and use models
`loadepiml.py` | Used to load data and normalize / clean it for training
`modeldeepdive.py` | Code to generate graphs and analysis / descriptions of models generated in the project
`bestmodels.py` | Hard coded model(s) with parameters found to be good during various searches for hyper-parameters
`semisuperhelper.py` | Helper code to handle unlabeled data (labeled as `-1`)
`epimlsklearn.epimlmetrics.py` | Custom scoring metrics for models using unlabled data
`epimlsklearn.frankenscorer.py` | An Sklearn scorer object that can score multiple metrics at once
`epimlsklearn.jsearch.py` | A random search of hyper-parameters using a Frankenscorer
`epimlsklearn.nestedcross.py` | A class that runs a nested cross validation search
`epimlsklearn.pnuwrapper.py` | Wraps classifiers to be used with unlabeled data PNU = *P*ositive *N*egative *U*nlabled and has mechanism for random undersampling of unlabeled data
`epimlsklearn.repeatedsampling.py` | Wraps classifiers to be used with massively unbalanced data using repeated oversampling
`epimlsklearn.rfsubsample.py` | A modified Random Forest algorithm where every bootstrapped sample used adheres to a _target imbalance ratio_, uses oversampling

### Notebook File Descriptions
File | Description
--- | ---
`1 - Epimlmain Example` | Example code using epimlmain.py
`2 - ModelDeepDive with Best Model` | Code to deep dive the _Model 3_ from the manuscript
`3.1 - LASSO - PN and PNU` | Code to generate the LASSO baseline model using only PN (Positive and Negative) data and then a search for a model using PNU (PN + Unlabeled) data (pnuwrapper.py) which is undersampled
`3.2 - SVC - PN and PNU` | Code to generate a Support Vector Classifier (SVC) using only PN data and then a search for an SVC using PNU data which is understampled using pnuwrapper.py
`3.3 - RF - PNU All Data` | This file uses a good performing RF model found in previous searches and test with undersampling turned off, then generate a validation curve for an undersampling parameter in pnuwrapper.py
`3.4 - RF - PNU Random Search` | A nested cross validated 3x3 100 iteration search for a Random Forest (RF) classifier wrapped with PNU data which is undersampled using pnuwrapper.py.  2 different metrics are used to optimize results seperately.
`3.5 - RF - PNU Repeated Random Subsampling Random Search` | A nested cross validated 3x3 60 iteration search for a RF using repeated random subsampling (oversampling) using all PNU data.  This notebook also contains a lot of exploration and graphs around the best model found.


### Input data format

Note that the code will run as long as there is a `Gender` column in the data along with some set of the columns below:

The input dataset used in the study had this format:

SAS Flag Name | Long Name | Description
--- | --- | ---
`MemberID`| ID of patient | anonymous patient
`age` | Age of patient | How old patient is
`Gender` | Gender of patient
`true_pos_flag` | Flag if patient has been labeled as a true positive | Patients who have taken at least 3+ scripts of a PERT are labeled as true positive
`true_neg_flag` |  Flag if patient has been labeled as true negative | Patients who have taken the fecal elastase test and not taken a PERT are labeled as true negative
`unlabel_flag` | Flag if patient has not been labeled
`DIAG_FLAG1_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 DIAGNOSIS GROUP OTHER PANCREATIC NEOPLASMS | OTHER PANCREATIC NEOPLASMS
`DIAG_FLAG2_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 DIAGNOSIS GROUP CYSTIC FIBROSIS | CYSTIC FIBROSIS
`DIAG_FLAG3_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 DIAGNOSIS GROUP PANCREATIC CANCER | PANCREATIC CANCER
`DIAG_FLAG4_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 DIAGNOSIS GROUP INFLAMMATORY CONDITIONS OF PANCREAS | INFLAMMATORY CONDITIONS OF PANCREAS
`DIAG_FLAG5_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 DIAGNOSIS GROUP OTHER PANCREATIC CONDITIONS | OTHER PANCREATIC CONDITIONS
`DIAG_FLAG6_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 DIAGNOSIS GROUP MALABSORPTION SYNDROMES | MALABSORPTION SYNDROMES
`DIAG_FLAG7_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 DIAGNOSIS GROUP DIABETES | DIABETES
`DIAG_FLAG8_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 DIAGNOSIS GROUP HIV | HIV
`DIAG_FLAG9_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 DIAGNOSIS GROUP IRRITABLE BOWEL SYNDROME | IRRITABLE BOWEL SYNDROME
`DIAG_FLAG10_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 DIAGNOSIS GROUP ULCERATION COLITIS | ULCERATION COLITIS
`DIAG_FLAG11_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 DIAGNOSIS GROUP CROHN'S DISEASE | CROHN'S DISEASE
`DIAG_FLAG12_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 DIAGNOSIS GROUP CONGENITAL ANOMALIES | CONGENITAL ANOMALIES
`DIAG_FLAG15_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 DIAGNOSIS GROUP EXPOSURE/PROPHYLATIC | EXPOSURE/PROPHYLATIC
`DIAG_FLAG16_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 DIAGNOSIS GROUP EXTERNAL CAUSES | EXTERNAL CAUSES
`DIAG_FLAG17_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 DIAGNOSIS GROUP HS ENCOUNTERS | HS ENCOUNTERS
`DIAG_FLAG21_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 DIAGNOSIS GROUP MUSCULOSKELETAL/CONNECTIVE | MUSCULOSKELETAL/CONNECTIVE
`DIAG_FLAG22_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 DIAGNOSIS GROUP NEWBORN CLASSIFICATION | NEWBORN CLASSIFICATION
`DIAG_FLAG23_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 DIAGNOSIS GROUP PERINATAL CONDITIONS | PERINATAL CONDITIONS
`DIAG_FLAG24_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 DIAGNOSIS GROUP SPECIAL ENCOUNTERS | SPECIAL ENCOUNTERS
`DIAG_FLAG25_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 DIAGNOSIS GROUP SPECIAL PROCEDURES | SPECIAL PROCEDURES
`DIAG_FLAG26_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 DIAGNOSIS GROUP ACCIDENTAL POISONING | ACCIDENTAL POISONING
`DIAG_FLAG27_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 DIAGNOSIS GROUP ACUTE RHEUMATIC FEVER | ACUTE RHEUMATIC FEVER
`DIAG_FLAG28_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 DIAGNOSIS GROUP ALCOHOL SA | ALCOHOL SA
`DIAG_FLAG29_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 DIAGNOSIS GROUP APPENDICITIS | APPENDICITIS
`DIAG_FLAG30_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 DIAGNOSIS GROUP ARTHROPATHIES | ARTHROPATHIES
`DIAG_FLAG31_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 DIAGNOSIS GROUP BLOOD | BLOOD
`DIAG_FLAG32_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 DIAGNOSIS GROUP CEREBROVASCULAR | CEREBROVASCULAR
`DIAG_FLAG33_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 DIAGNOSIS GROUP CHRONIC RHEUM HEART | CHRONIC RHEUM HEART
`DIAG_FLAG34_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 DIAGNOSIS GROUP CIRCULATORY | CIRCULATORY
`DIAG_FLAG35_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 DIAGNOSIS GROUP CNS  | CNS 
`DIAG_FLAG36_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 DIAGNOSIS GROUP CNS INFLAMMATORY | CNS INFLAMMATORY
`DIAG_FLAG37_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 DIAGNOSIS GROUP CONGENITAL | CONGENITAL
`DIAG_FLAG38_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 DIAGNOSIS GROUP DIGESTIVE | DIGESTIVE
`DIAG_FLAG39_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 DIAGNOSIS GROUP DRUGS ADVERSE | DRUGS ADVERSE
`DIAG_FLAG40_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 DIAGNOSIS GROUP EAR | EAR
`DIAG_FLAG41_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 DIAGNOSIS GROUP ENDOCRINE | ENDOCRINE
`DIAG_FLAG42_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 DIAGNOSIS GROUP ESOPHAGUS | ESOPHAGUS
`DIAG_FLAG45_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 DIAGNOSIS GROUP GENITOURINARY | GENITOURINARY
`DIAG_FLAG46_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 DIAGNOSIS GROUP HEADACHE | HEADACHE
`DIAG_FLAG47_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 DIAGNOSIS GROUP HEREDITARY CNS | HEREDITARY CNS
`DIAG_FLAG48_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 DIAGNOSIS GROUP HERNIA | HERNIA
`DIAG_FLAG49_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 DIAGNOSIS GROUP HYPERTENSIVE | HYPERTENSIVE
`DIAG_FLAG50_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 DIAGNOSIS GROUP INFLAMMATORY SKIN | INFLAMMATORY SKIN
`DIAG_FLAG51_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 DIAGNOSIS GROUP INJ POISONING | INJ POISONING
`DIAG_FLAG52_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 DIAGNOSIS GROUP ISCHEMIC HEART | ISCHEMIC HEART
`DIAG_FLAG53_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 DIAGNOSIS GROUP MENTAL | MENTAL
`DIAG_FLAG54_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 DIAGNOSIS GROUP METABOLIC | METABOLIC
`DIAG_FLAG55_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 DIAGNOSIS GROUP MUSCULOSKELETAL | MUSCULOSKELETAL
`DIAG_FLAG60_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 DIAGNOSIS GROUP NUTRITIONAL | NUTRITIONAL
`DIAG_FLAG61_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 DIAGNOSIS GROUP ORAL  | ORAL 
`DIAG_FLAG62_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 DIAGNOSIS GROUP PERINATAL | PERINATAL
`DIAG_FLAG63_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 DIAGNOSIS GROUP PERIPHERAL | PERIPHERAL
`DIAG_FLAG66_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 DIAGNOSIS GROUP PULMONARY | PULMONARY
`DIAG_FLAG67_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 DIAGNOSIS GROUP RESPIRATORY | RESPIRATORY
`DIAG_FLAG68_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 DIAGNOSIS GROUP SKIN | SKIN
`DIAG_FLAG69_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 DIAGNOSIS GROUP SKIN INFECTIONS | SKIN INFECTIONS
`DIAG_FLAG70_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 DIAGNOSIS GROUP SLEEP | SLEEP
`DIAG_FLAG71_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 DIAGNOSIS GROUP SPECIAL ENCOUNTERS | SPECIAL ENCOUNTERS
`DIAG_FLAG72_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 DIAGNOSIS GROUP OTHER SYMPTOMS | OTHER SYMPTOMS
`DIAG_FLAG73_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 DIAGNOSIS GROUP THYROID | THYROID
`DIAG_FLAG74_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 DIAGNOSIS GROUP ENDOCRINE, BASAL METABOLISM, LIVER SYMPTOMS | ENDOCRINE, BASAL METABOLISM, LIVER SYMPTOMS
`DIAG_FLAG75_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 DIAGNOSIS GROUP SYMPTOMS (ABDOMINAL AND PELVIS) | SYMPTOMS (ABDOMINAL AND PELVIS)
`DIAG_FLAG76_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 DIAGNOSIS GROUP SYMPTOMS (ABDOMINAL FINDINGS GASTROINTESTINAL) | SYMPTOMS (ABDOMINAL FINDINGS GASTROINTESTINAL)
`DIAG_FLAG77_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 DIAGNOSIS GROUP SYMPTOMS (DIGESTIVE SYSTEM) | SYMPTOMS (DIGESTIVE SYSTEM)
`DIAG_FLAG78_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 DIAGNOSIS GROUP SYMPTOMS (EARLY SATIETY) | SYMPTOMS (EARLY SATIETY)
`DIAG_FLAG79_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 DIAGNOSIS GROUP SYMPTOMS (MALAISE AND FATIGUE) | SYMPTOMS (MALAISE AND FATIGUE)
`DIAG_FLAG80_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 DIAGNOSIS GROUP SYMPTOMS (NUTRITION) | SYMPTOMS (NUTRITION)
`DIAG_FLAG81_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 DIAGNOSIS GROUP SYMPTOMS (STOOL CONTENTS) | SYMPTOMS (STOOL CONTENTS)
`DIAG_FLAG82_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 DIAGNOSIS GROUP E-CODES | E-CODES
`DIAG_FLAG83_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 DIAGNOSIS GROUP INJURY AND POISONING | INJURY AND POISONING
`DIAG_FLAG84_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 DIAGNOSIS GROUP EYE ADNEXA | EYE ADNEXA
`DIAG_FLAG85_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 DIAGNOSIS GROUP NEOPLASMS | NEOPLASMS
`DIAG_FLAG86_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 DIAGNOSIS GROUP NONHIV INFECTIOUS | NONHIV INFECTIOUS
`DIAG_FLAG87_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 DIAGNOSIS GROUP PREGNANCY COMP | PREGNANCY COMP
`PROC_FLAG1_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 PROCEDURE GROUP TOTAL PANCREATECTOMY | TOTAL PANCREATECTOMY
`PROC_FLAG2_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 PROCEDURE GROUP RADICAL PANCREATODUODENECTOMY | RADICAL PANCREATODUODENECTOMY
`PROC_FLAG3_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 PROCEDURE GROUP RADICAL SUBTOTAL PANCREATECTOMY | RADICAL SUBTOTAL PANCREATECTOMY
`PROC_FLAG4_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 PROCEDURE GROUP PARTIAL PANCREATECTOMY | PARTIAL PANCREATECTOMY
`PROC_FLAG5_SUM` | COUNT OF CLAIMS WITH CONDITION ICD-9 PROCEDURE GROUP OTHER PANCREATIC SURGERY | OTHER PANCREATIC SURGERY
`CPT_FLAG1_SUM` | COUNT OF CLAIMS WITH CONDITION PROCEDURE CODE GROUP BARIATRIC SURGERY | BARIATRIC SURGERY
`CPT_FLAG2_SUM` | COUNT OF CLAIMS WITH CONDITION PROCEDURE CODE GROUP OTHER PANCREATIC SURGERY | OTHER PANCREATIC SURGERY
`CPT_FLAG3_SUM` | COUNT OF CLAIMS WITH CONDITION PROCEDURE CODE GROUP LABS CAPSULE ENDOSCOPY | LABS CAPSULE ENDOSCOPY
`CPT_FLAG4_SUM` | COUNT OF CLAIMS WITH CONDITION PROCEDURE CODE GROUP LABS FECAL TEST | LABS FECAL TEST
`CPT_FLAG5_SUM` | COUNT OF CLAIMS WITH CONDITION PROCEDURE CODE GROUP LABS MALABSORPTION TEST | LABS MALABSORPTION TEST
`CPT_FLAG6_SUM` | COUNT OF CLAIMS WITH CONDITION PROCEDURE CODE GROUP LABS VITAMIN LEVELS TEST | LABS VITAMIN LEVELS TEST
`CPT_FLAG7_SUM` | COUNT OF CLAIMS WITH CONDITION PROCEDURE CODE GROUP RADIOLOGY GI SYSTEM | RADIOLOGY GI SYSTEM
`CPT_FLAG8_SUM` | COUNT OF CLAIMS WITH CONDITION PROCEDURE CODE GROUP OTHER RADIOLOGY BONE DENSITY | OTHER RADIOLOGY BONE DENSITY
`CPT_FLAG9_SUM` | COUNT OF CLAIMS WITH CONDITION PROCEDURE CODE GROUP RADIOLOGY ABDOMINAL | RADIOLOGY ABDOMINAL
`CPT_FLAG10_SUM` | COUNT OF CLAIMS WITH CONDITION PROCEDURE CODE GROUP RADIOLOGY GI TRACT | RADIOLOGY GI TRACT
`CPT_FLAG11_SUM` | COUNT OF CLAIMS WITH CONDITION PROCEDURE CODE GROUP SURGERY BARIATRIC | SURGERY BARIATRIC
`CPT_FLAG12_SUM` | COUNT OF CLAIMS WITH CONDITION PROCEDURE CODE GROUP SURGERY BILIARY TRACT | SURGERY BILIARY TRACT
`CPT_FLAG13_SUM` | COUNT OF CLAIMS WITH CONDITION PROCEDURE CODE GROUP SURGERY COLONOSCOPY | SURGERY COLONOSCOPY
`CPT_FLAG14_SUM` | COUNT OF CLAIMS WITH CONDITION PROCEDURE CODE GROUP SURGERY DUODENAL ASPIRATION | SURGERY DUODENAL ASPIRATION
`CPT_FLAG15_SUM` | COUNT OF CLAIMS WITH CONDITION PROCEDURE CODE GROUP SURGERY LYSIS ADHESIONS | SURGERY LYSIS ADHESIONS
`CPT_FLAG16_SUM` | COUNT OF CLAIMS WITH CONDITION PROCEDURE CODE GROUP SURGERY PANCREAS | SURGERY PANCREAS
`CPT_FLAG17_SUM` | COUNT OF CLAIMS WITH CONDITION PROCEDURE CODE GROUP SURGERY PANCREAS TRANSPLANT | SURGERY PANCREAS TRANSPLANT
`CPT_FLAG18_SUM` | COUNT OF CLAIMS WITH CONDITION PROCEDURE CODE GROUP ULTRASOUND ABDOMINAL | ULTRASOUND ABDOMINAL
`CPT_FLAG19_SUM` | COUNT OF CLAIMS WITH CONDITION PROCEDURE CODE GROUP ULTRASOUND PELVIS | ULTRASOUND PELVIS
`CPT_FLAG20_SUM` | COUNT OF CLAIMS WITH CONDITION PROCEDURE CODE GROUP RADIOLOGY | RADIOLOGY
`CPT_FLAG21_SUM` | COUNT OF CLAIMS WITH CONDITION PROCEDURE CODE GROUP ANESTHESIA | ANESTHESIA
`CPT_FLAG22_SUM` | COUNT OF CLAIMS WITH CONDITION PROCEDURE CODE GROUP INTEGUMENTARY SURGERY | INTEGUMENTARY SURGERY
`CPT_FLAG26_SUM` | COUNT OF CLAIMS WITH CONDITION PROCEDURE CODE GROUP RESPIRATORY SURGERY | RESPIRATORY SURGERY
`CPT_FLAG29_SUM` | COUNT OF CLAIMS WITH CONDITION PROCEDURE CODE GROUP HEM LYMPH SURGERY | HEM LYMPH SURGERY
`CPT_FLAG32_SUM` | COUNT OF CLAIMS WITH CONDITION PROCEDURE CODE GROUP URINARY SURGERY | URINARY SURGERY
`CPT_FLAG33_SUM` | COUNT OF CLAIMS WITH CONDITION PROCEDURE CODE GROUP M GENITAL SURGERY | M GENITAL SURGERY
`CPT_FLAG34_SUM` | COUNT OF CLAIMS WITH CONDITION PROCEDURE CODE GROUP F GENITAL SURGERY | F GENITAL SURGERY
`CPT_FLAG35_SUM` | COUNT OF CLAIMS WITH CONDITION PROCEDURE CODE GROUP NERVOUS SYSTEM SURGERY | NERVOUS SYSTEM SURGERY
`CPT_FLAG36_SUM` | COUNT OF CLAIMS WITH CONDITION PROCEDURE CODE GROUP EYE SURGERY | EYE SURGERY
`CPT_FLAG37_SUM` | COUNT OF CLAIMS WITH CONDITION PROCEDURE CODE GROUP OTHER RADIOLOGY | OTHER RADIOLOGY
`CPT_FLAG43_SUM` | COUNT OF CLAIMS WITH CONDITION PROCEDURE CODE GROUP EM | EM
`CPT_FLAG44_SUM` | COUNT OF CLAIMS WITH CONDITION PROCEDURE CODE GROUP ELASTASE TEST | ELASTASE TEST
`CPT_FLAG45_SUM` | COUNT OF CLAIMS WITH CONDITION PROCEDURE CODE GROUP MUSCULOSKELETAL SURGERY | MUSCULOSKELETAL SURGERY
`CPT_FLAG46_SUM` | COUNT OF CLAIMS WITH CONDITION PROCEDURE CODE GROUP CV | CV
`CPT_FLAG47_SUM` | COUNT OF CLAIMS WITH CONDITION PROCEDURE CODE GROUP DIGESTIVE SURGERY | DIGESTIVE SURGERY
`CPT_FLAG48_SUM` | COUNT OF CLAIMS WITH CONDITION PROCEDURE CODE GROUP PATHOLOGY | PATHOLOGY
`CPT_FLAG49_SUM` | COUNT OF CLAIMS WITH CONDITION PROCEDURE CODE GROUP MEDICINE | MEDICINE
`REVCODE_FLAG1_SUM` | COUNT OF CLAIMS WITH CONDITION REVENUE CODE GROUP ADDL BENEFITS OTHER - GENERAL | ADDL BENEFITS OTHER - GENERAL
`REVCODE_FLAG2_SUM` | COUNT OF CLAIMS WITH CONDITION REVENUE CODE GROUP AMBULATORY SURGICAL CARE | AMBULATORY SURGICAL CARE
`REVCODE_FLAG3_SUM` | COUNT OF CLAIMS WITH CONDITION REVENUE CODE GROUP ANESTHESIA | ANESTHESIA
`REVCODE_FLAG4_SUM` | COUNT OF CLAIMS WITH CONDITION REVENUE CODE GROUP AUDIOLOGY | AUDIOLOGY
`REVCODE_FLAG5_SUM` | COUNT OF CLAIMS WITH CONDITION REVENUE CODE GROUP BLOOD PRODUCTS | BLOOD PRODUCTS
`REVCODE_FLAG6_SUM` | COUNT OF CLAIMS WITH CONDITION REVENUE CODE GROUP BLOOD STORAGE AND PROCESSING | BLOOD STORAGE AND PROCESSING
`REVCODE_FLAG7_SUM` | COUNT OF CLAIMS WITH CONDITION REVENUE CODE GROUP CARDIAC CATHETER LAB | CARDIAC CATHETER LAB
`REVCODE_FLAG8_SUM` | COUNT OF CLAIMS WITH CONDITION REVENUE CODE GROUP CARDIOLOGY | CARDIOLOGY
`REVCODE_FLAG9_SUM` | COUNT OF CLAIMS WITH CONDITION REVENUE CODE GROUP CAST ROOM | CAST ROOM
`REVCODE_FLAG10_SUM` | COUNT OF CLAIMS WITH CONDITION REVENUE CODE GROUP CLINIC | CLINIC
`REVCODE_FLAG11_SUM` | COUNT OF CLAIMS WITH CONDITION REVENUE CODE GROUP CT SCAN | CT SCAN
`REVCODE_FLAG12_SUM` | COUNT OF CLAIMS WITH CONDITION REVENUE CODE GROUP DRUGS REQUIRING SPECIFIC IDENTIFICATION | DRUGS REQUIRING SPECIFIC IDENTIFICATION
`REVCODE_FLAG13_SUM` | COUNT OF CLAIMS WITH CONDITION REVENUE CODE GROUP EEG | EEG
`REVCODE_FLAG14_SUM` | COUNT OF CLAIMS WITH CONDITION REVENUE CODE GROUP EKG/ECG | EKG/ECG
`REVCODE_FLAG15_SUM` | COUNT OF CLAIMS WITH CONDITION REVENUE CODE GROUP EMERGENCY ROOM | EMERGENCY ROOM
`REVCODE_FLAG16_SUM` | COUNT OF CLAIMS WITH CONDITION REVENUE CODE GROUP GASTROINTESTINAL SERVICES | GASTROINTESTINAL SERVICES
`REVCODE_FLAG17_SUM` | COUNT OF CLAIMS WITH CONDITION REVENUE CODE GROUP IV THERAPY | IV THERAPY
`REVCODE_FLAG18_SUM` | COUNT OF CLAIMS WITH CONDITION REVENUE CODE GROUP LABOR & DELIVERY | LABOR & DELIVERY
`REVCODE_FLAG19_SUM` | COUNT OF CLAIMS WITH CONDITION REVENUE CODE GROUP LABORATORY | LABORATORY
`REVCODE_FLAG20_SUM` | COUNT OF CLAIMS WITH CONDITION REVENUE CODE GROUP LABORATORY - PATHOLOGICAL | LABORATORY - PATHOLOGICAL
`REVCODE_FLAG21_SUM` | COUNT OF CLAIMS WITH CONDITION REVENUE CODE GROUP LITHOTRIPSY | LITHOTRIPSY
`REVCODE_FLAG22_SUM` | COUNT OF CLAIMS WITH CONDITION REVENUE CODE GROUP MAGNETIC RESONANCE TECHNOLOGY (MRT) | MAGNETIC RESONANCE TECHNOLOGY (MRT)
`REVCODE_FLAG23_SUM` | COUNT OF CLAIMS WITH CONDITION REVENUE CODE GROUP MEDICAL SURGICAL SUPPLIES | MEDICAL SURGICAL SUPPLIES
`REVCODE_FLAG24_SUM` | COUNT OF CLAIMS WITH CONDITION REVENUE CODE GROUP NUCLEAR MEDICINE | NUCLEAR MEDICINE
`REVCODE_FLAG25_SUM` | COUNT OF CLAIMS WITH CONDITION REVENUE CODE GROUP OBSERVATION ROOM | OBSERVATION ROOM
`REVCODE_FLAG26_SUM` | COUNT OF CLAIMS WITH CONDITION REVENUE CODE GROUP OCCUPATIONAL THERAPY | OCCUPATIONAL THERAPY
`REVCODE_FLAG27_SUM` | COUNT OF CLAIMS WITH CONDITION REVENUE CODE GROUP ONCOLOGY | ONCOLOGY
`REVCODE_FLAG28_SUM` | COUNT OF CLAIMS WITH CONDITION REVENUE CODE GROUP OPERATING ROOM SERVICES | OPERATING ROOM SERVICES
`REVCODE_FLAG29_SUM` | COUNT OF CLAIMS WITH CONDITION REVENUE CODE GROUP OTH AMBULANCE | OTH AMBULANCE
`REVCODE_FLAG30_SUM` | COUNT OF CLAIMS WITH CONDITION REVENUE CODE GROUP OTH DME AND SUPPLIES | OTH DME AND SUPPLIES
`REVCODE_FLAG31_SUM` | COUNT OF CLAIMS WITH CONDITION REVENUE CODE GROUP OTH PRIVATE DUTY NURSING/HOME HEALTH - HH | OTH PRIVATE DUTY NURSING/HOME HEALTH - HH
`REVCODE_FLAG32_SUM` | COUNT OF CLAIMS WITH CONDITION REVENUE CODE GROUP OTH PRIVATE DUTY NURSING/HOME HEALTH - HOSPICE | OTH PRIVATE DUTY NURSING/HOME HEALTH - HOSPICE
`REVCODE_FLAG33_SUM` | COUNT OF CLAIMS WITH CONDITION REVENUE CODE GROUP OTHER DIAGNOSTIC SERVICES | OTHER DIAGNOSTIC SERVICES
`REVCODE_FLAG34_SUM` | COUNT OF CLAIMS WITH CONDITION REVENUE CODE GROUP OTHER IMAGING SERVICES | OTHER IMAGING SERVICES
`REVCODE_FLAG35_SUM` | COUNT OF CLAIMS WITH CONDITION REVENUE CODE GROUP OTHER THERAPEUTIC - CARDIAC REHAB | OTHER THERAPEUTIC - CARDIAC REHAB
`REVCODE_FLAG36_SUM` | COUNT OF CLAIMS WITH CONDITION REVENUE CODE GROUP OTHER THERAPEUTIC - EDUCATION / TRAINING | OTHER THERAPEUTIC - EDUCATION / TRAINING
`REVCODE_FLAG37_SUM` | COUNT OF CLAIMS WITH CONDITION REVENUE CODE GROUP OTHER THERAPEUTIC SERVICES | OTHER THERAPEUTIC SERVICES
`REVCODE_FLAG38_SUM` | COUNT OF CLAIMS WITH CONDITION REVENUE CODE GROUP PHARMACY | PHARMACY
`REVCODE_FLAG39_SUM` | COUNT OF CLAIMS WITH CONDITION REVENUE CODE GROUP PHYSICAL THERAPY | PHYSICAL THERAPY
`REVCODE_FLAG40_SUM` | COUNT OF CLAIMS WITH CONDITION REVENUE CODE GROUP PROFESSIONAL CARDIOVASCULAR | PROFESSIONAL CARDIOVASCULAR
`REVCODE_FLAG41_SUM` | COUNT OF CLAIMS WITH CONDITION REVENUE CODE GROUP PROFESSIONAL ER VISITS AND OBSERVATION CARE | PROFESSIONAL ER VISITS AND OBSERVATION CARE
`REVCODE_FLAG42_SUM` | COUNT OF CLAIMS WITH CONDITION REVENUE CODE GROUP PROFESSIONAL INPATIENT VISITS - MEDICAL | PROFESSIONAL INPATIENT VISITS - MEDICAL
`REVCODE_FLAG43_SUM` | COUNT OF CLAIMS WITH CONDITION REVENUE CODE GROUP PROFESSIONAL MISCELLANEOUS MEDICAL - DIALYSIS | PROFESSIONAL MISCELLANEOUS MEDICAL - DIALYSIS
`REVCODE_FLAG44_SUM` | COUNT OF CLAIMS WITH CONDITION REVENUE CODE GROUP PROFESSIONAL MISCELLANEOUS MEDICAL - GENERAL | PROFESSIONAL MISCELLANEOUS MEDICAL - GENERAL
`REVCODE_FLAG45_SUM` | COUNT OF CLAIMS WITH CONDITION REVENUE CODE GROUP PROFESSIONAL MISCELLANEOUS MEDICAL - NEUROLOGY | PROFESSIONAL MISCELLANEOUS MEDICAL - NEUROLOGY
`REVCODE_FLAG46_SUM` | COUNT OF CLAIMS WITH CONDITION REVENUE CODE GROUP PROFESSIONAL MISCELLANEOUS MEDICAL - OPHTHALMOLOGY | PROFESSIONAL MISCELLANEOUS MEDICAL - OPHTHALMOLOGY
`REVCODE_FLAG47_SUM` | COUNT OF CLAIMS WITH CONDITION REVENUE CODE GROUP PROFESSIONAL MISCELLANEOUS MEDICAL - PULMONOLOGY | PROFESSIONAL MISCELLANEOUS MEDICAL - PULMONOLOGY
`REVCODE_FLAG48_SUM` | COUNT OF CLAIMS WITH CONDITION REVENUE CODE GROUP PROFESSIONAL OFFICE/HOME VISITS - PCP | PROFESSIONAL OFFICE/HOME VISITS - PCP
`REVCODE_FLAG49_SUM` | COUNT OF CLAIMS WITH CONDITION REVENUE CODE GROUP PROFESSIONAL OUTPATIENT ALCOHOL & DRUG ABUSE | PROFESSIONAL OUTPATIENT ALCOHOL & DRUG ABUSE
`REVCODE_FLAG50_SUM` | COUNT OF CLAIMS WITH CONDITION REVENUE CODE GROUP PROFESSIONAL OUTPATIENT ANESTHESIA | PROFESSIONAL OUTPATIENT ANESTHESIA
`REVCODE_FLAG51_SUM` | COUNT OF CLAIMS WITH CONDITION REVENUE CODE GROUP PROFESSIONAL OUTPATIENT PSYCHIATRIC | PROFESSIONAL OUTPATIENT PSYCHIATRIC
`REVCODE_FLAG52_SUM` | COUNT OF CLAIMS WITH CONDITION REVENUE CODE GROUP PROFESSIONAL OUTPATIENT SURGERY | PROFESSIONAL OUTPATIENT SURGERY
`REVCODE_FLAG53_SUM` | COUNT OF CLAIMS WITH CONDITION REVENUE CODE GROUP PROFESSIONAL PATHOLOGY/LAB | PROFESSIONAL PATHOLOGY/LAB
`REVCODE_FLAG54_SUM` | COUNT OF CLAIMS WITH CONDITION REVENUE CODE GROUP PROFESSIONAL PHYSICAL THERAPY | PROFESSIONAL PHYSICAL THERAPY
`REVCODE_FLAG55_SUM` | COUNT OF CLAIMS WITH CONDITION REVENUE CODE GROUP PROFESSIONAL PREVENTIVE IMMUNIZATIONS | PROFESSIONAL PREVENTIVE IMMUNIZATIONS
`REVCODE_FLAG56_SUM` | COUNT OF CLAIMS WITH CONDITION REVENUE CODE GROUP PROFESSIONAL PREVENTIVE OTHER - GENERAL | PROFESSIONAL PREVENTIVE OTHER - GENERAL
`REVCODE_FLAG57_SUM` | COUNT OF CLAIMS WITH CONDITION REVENUE CODE GROUP PROFESSIONAL RADIOLOGY OP - GENERAL - DIAGNOSTIC | PROFESSIONAL RADIOLOGY OP - GENERAL - DIAGNOSTIC
`REVCODE_FLAG58_SUM` | COUNT OF CLAIMS WITH CONDITION REVENUE CODE GROUP PROFESSIONAL RADIOLOGY OP - GENERAL - THERAPEUTIC | PROFESSIONAL RADIOLOGY OP - GENERAL - THERAPEUTIC
`REVCODE_FLAG59_SUM` | COUNT OF CLAIMS WITH CONDITION REVENUE CODE GROUP PROFESSIONAL URGENT CARE VISITS | PROFESSIONAL URGENT CARE VISITS
`REVCODE_FLAG60_SUM` | COUNT OF CLAIMS WITH CONDITION REVENUE CODE GROUP PSYCHIATRIC / PSYCHOLOGICAL SERVICES | PSYCHIATRIC / PSYCHOLOGICAL SERVICES
`REVCODE_FLAG61_SUM` | COUNT OF CLAIMS WITH CONDITION REVENUE CODE GROUP PSYCHIATRIC / PSYCHOLOGICAL TREATMENTS | PSYCHIATRIC / PSYCHOLOGICAL TREATMENTS
`REVCODE_FLAG62_SUM` | COUNT OF CLAIMS WITH CONDITION REVENUE CODE GROUP PULMONARY FUNCTION | PULMONARY FUNCTION
`REVCODE_FLAG63_SUM` | COUNT OF CLAIMS WITH CONDITION REVENUE CODE GROUP RADIOLOGY - DIAGNOSTIC | RADIOLOGY - DIAGNOSTIC
`REVCODE_FLAG64_SUM` | COUNT OF CLAIMS WITH CONDITION REVENUE CODE GROUP RADIOLOGY - THERAPEUTIC | RADIOLOGY - THERAPEUTIC
`REVCODE_FLAG65_SUM` | COUNT OF CLAIMS WITH CONDITION REVENUE CODE GROUP RECOVERY ROOM | RECOVERY ROOM
`REVCODE_FLAG66_SUM` | COUNT OF CLAIMS WITH CONDITION REVENUE CODE GROUP RESPIRATORY SERVICES | RESPIRATORY SERVICES
`REVCODE_FLAG67_SUM` | COUNT OF CLAIMS WITH CONDITION REVENUE CODE GROUP ROOM AND BOARD | ROOM AND BOARD
`REVCODE_FLAG68_SUM` | COUNT OF CLAIMS WITH CONDITION REVENUE CODE GROUP SPEECH-LANGUAGE PATHOLOGY | SPEECH-LANGUAGE PATHOLOGY
`REVCODE_FLAG69_SUM` | COUNT OF CLAIMS WITH CONDITION REVENUE CODE GROUP TOTAL CHARGES | TOTAL CHARGES
`REVCODE_FLAG70_SUM` | COUNT OF CLAIMS WITH CONDITION REVENUE CODE GROUP TREATMENT ROOM | TREATMENT ROOM
`NDC_CAT1_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP ADHD/ANTI-NARCOLEPSY/ANTI-OBESITY/ANOREXIANTS | ADHD/ANTI-NARCOLEPSY/ANTI-OBESITY/ANOREXIANTS
`NDC_CAT2_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP ALTERNATIVE MEDICINES | ALTERNATIVE MEDICINES
`NDC_CAT3_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP AMEBICIDES | AMEBICIDES
`NDC_CAT4_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP AMINOGLYCOSIDES | AMINOGLYCOSIDES
`NDC_CAT5_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP ANALGESICS - ANTI-INFLAMMATORY | ANALGESICS - ANTI-INFLAMMATORY
`NDC_CAT6_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP ANDROGENS-ANABOLIC | ANDROGENS-ANABOLIC
`NDC_CAT7_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP ANORECTAL AGENTS | ANORECTAL AGENTS
`NDC_CAT8_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP ANTACIDS | ANTACIDS
`NDC_CAT9_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP ANTHELMINTICS | ANTHELMINTICS
`NDC_CAT10_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP ANTIANGINAL AGENTS | ANTIANGINAL AGENTS
`NDC_CAT11_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP ANTIANXIETY AGENTS | ANTIANXIETY AGENTS
`NDC_CAT12_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP ANTIARRHYTHMICS | ANTIARRHYTHMICS
`NDC_CAT13_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP ANTIASTHMATIC AND BRONCHODILATOR AGENTS | ANTIASTHMATIC AND BRONCHODILATOR AGENTS
`NDC_CAT14_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP ANTICOAGULANTS | ANTICOAGULANTS
`NDC_CAT15_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP ANTICONVULSANTS | ANTICONVULSANTS
`NDC_CAT16_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP ANTIDEPRESSANTS | ANTIDEPRESSANTS
`NDC_CAT17_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP ANTIDOTES | ANTIDOTES
`NDC_CAT18_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP ANTIEMETICS | ANTIEMETICS
`NDC_CAT19_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP ANTIFUNGALS | ANTIFUNGALS
`NDC_CAT20_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP ANTIHISTAMINES | ANTIHISTAMINES
`NDC_CAT21_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP ANTIHYPERLIPIDEMICS | ANTIHYPERLIPIDEMICS
`NDC_CAT22_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP ANTIHYPERTENSIVES | ANTIHYPERTENSIVES
`NDC_CAT23_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP ANTI-INFECTIVE AGENTS - MISC. | ANTI-INFECTIVE AGENTS - MISC.
`NDC_CAT24_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP ANTIMALARIALS | ANTIMALARIALS
`NDC_CAT25_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP ANTIMYASTHENIC AGENTS | ANTIMYASTHENIC AGENTS
`NDC_CAT26_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP ANTINEOPLASTICS AND ADJUNCTIVE THERAPIES | ANTINEOPLASTICS AND ADJUNCTIVE THERAPIES
`NDC_CAT27_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP ANTIPARKINSON AGENTS | ANTIPARKINSON AGENTS
`NDC_CAT28_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP ANTIPSYCHOTICS/ANTIMANIC AGENTS | ANTIPSYCHOTICS/ANTIMANIC AGENTS
`NDC_CAT29_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP ANTIVIRALS | ANTIVIRALS
`NDC_CAT30_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP ASSORTED CLASSES | ASSORTED CLASSES
`NDC_CAT31_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP BETA BLOCKERS | BETA BLOCKERS
`NDC_CAT32_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP BIOLOGICALS MISC | BIOLOGICALS MISC
`NDC_CAT33_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP CALCIUM CHANNEL BLOCKERS | CALCIUM CHANNEL BLOCKERS
`NDC_CAT34_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP CARDIOTONICS | CARDIOTONICS
`NDC_CAT35_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP CARDIOVASCULAR AGENTS - MISC. | CARDIOVASCULAR AGENTS - MISC.
`NDC_CAT36_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP CHEMICALS | CHEMICALS
`NDC_CAT37_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP CONTRACEPTIVES | CONTRACEPTIVES
`NDC_CAT38_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP COUGH/COLD/ALLERGY | COUGH/COLD/ALLERGY
`NDC_CAT39_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP DERMATOLOGICALS | DERMATOLOGICALS
`NDC_CAT40_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP DIAGNOSTIC PRODUCTS | DIAGNOSTIC PRODUCTS
`NDC_CAT41_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP DIETARY PRODUCTS/DIETARY MANAGEMENT PRODUCTS | DIETARY PRODUCTS/DIETARY MANAGEMENT PRODUCTS
`NDC_CAT42_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP DIURETICS | DIURETICS
`NDC_CAT43_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP ENDOCRINE AND METABOLIC AGENTS - MISC. | ENDOCRINE AND METABOLIC AGENTS - MISC.
`NDC_CAT44_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP ANTIBIOTICS | ANTIBIOTICS
`NDC_CAT45_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP ANTIDIARRHEALS | ANTIDIARRHEALS
`NDC_CAT46_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP ANTIFLATULENTS | ANTIFLATULENTS
`NDC_CAT47_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP ANTINEOPLASTICS AND ADJUNCTIVE THERAPIES (EPI-ASSOCIATED) | ANTINEOPLASTICS AND ADJUNCTIVE THERAPIES (EPI-ASSOCIATED)
`NDC_CAT48_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP ASSORTED CLASSES (EPI-ASSOCIATED) | ASSORTED CLASSES (EPI-ASSOCIATED)
`NDC_CAT49_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP CORTICOSTEROIDS | CORTICOSTEROIDS
`NDC_CAT50_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP DERMATOLOGICALS (EPI-ASSOCIATED) | DERMATOLOGICALS (EPI-ASSOCIATED)
`NDC_CAT51_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP DIAGNOSTIC PRODUCTS (EPI-ASSOCIATED) | DIAGNOSTIC PRODUCTS (EPI-ASSOCIATED)
`NDC_CAT52_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP DIGESTIVE AIDS | DIGESTIVE AIDS
`NDC_CAT53_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP HEMATOPOIETIC AGENTS (EPI-ASSOCIATED) | HEMATOPOIETIC AGENTS (EPI-ASSOCIATED)
`NDC_CAT54_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP INSULIN | INSULIN
`NDC_CAT55_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP MULTIVITAMINS | MULTIVITAMINS
`NDC_CAT56_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP NUTRIENTS | NUTRIENTS
`NDC_CAT57_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP PSYCHOTHERAPEUTIC AND NEUROLOGICAL AGENTS - MISC. (EPI-ASSOCIATED) | PSYCHOTHERAPEUTIC AND NEUROLOGICAL AGENTS - MISC. (EPI-ASSOCIATED)
`NDC_CAT58_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP ULCER DRUGS | ULCER DRUGS
`NDC_CAT59_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP VITAMINS (EPI-ASSOCIATED)  | VITAMINS (EPI-ASSOCIATED) 
`NDC_CAT60_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP ESTROGENS | ESTROGENS
`NDC_CAT61_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP GALLSTONE SOLUBILIZING AGENTS | GALLSTONE SOLUBILIZING AGENTS
`NDC_CAT62_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP GENERAL ANESTHETICS | GENERAL ANESTHETICS
`NDC_CAT63_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP GENITOURINARY AGENTS - MISCELLANEOUS | GENITOURINARY AGENTS - MISCELLANEOUS
`NDC_CAT64_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP GASTROINTESTINAL ANTIALLERGY AGENTS | GASTROINTESTINAL ANTIALLERGY AGENTS
`NDC_CAT65_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP GASTROINTESTINAL CHLORIDE CHANNEL ACTIVATORS | GASTROINTESTINAL CHLORIDE CHANNEL ACTIVATORS
`NDC_CAT66_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP GASTROINTESTINAL STIMULANTS | GASTROINTESTINAL STIMULANTS
`NDC_CAT67_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP GOUT AGENTS | GOUT AGENTS
`NDC_CAT68_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP HEMATOPOIETIC AGENTS | HEMATOPOIETIC AGENTS
`NDC_CAT69_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP HEMOSTATICS | HEMOSTATICS
`NDC_CAT70_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP HYPNOTICS | HYPNOTICS
`NDC_CAT71_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP IRRITABLE BOWEL SYNDROME (IBS) AGENTS | IRRITABLE BOWEL SYNDROME (IBS) AGENTS
`NDC_CAT72_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP INFLAMMATORY BOWEL AGENTS | INFLAMMATORY BOWEL AGENTS
`NDC_CAT73_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP INTESTINAL ACIDIFIERS | INTESTINAL ACIDIFIERS
`NDC_CAT74_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP LAXATIVES | LAXATIVES
`NDC_CAT75_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP LOCAL ANESTHETICS-PARENTERAL | LOCAL ANESTHETICS-PARENTERAL
`NDC_CAT76_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP MACROLIDES | MACROLIDES
`NDC_CAT77_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP MEDICAL DEVICES | MEDICAL DEVICES
`NDC_CAT78_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP MIGRAINE PRODUCTS | MIGRAINE PRODUCTS
`NDC_CAT79_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP MINERALS & ELECTROLYTES | MINERALS & ELECTROLYTES
`NDC_CAT80_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP MOUTH/THROAT/DENTAL AGENTS | MOUTH/THROAT/DENTAL AGENTS
`NDC_CAT82_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP MUSCULOSKELETAL THERAPY AGENTS | MUSCULOSKELETAL THERAPY AGENTS
`NDC_CAT83_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP NASAL AGENTS - SYSTEMIC AND TOPICAL | NASAL AGENTS - SYSTEMIC AND TOPICAL
`NDC_CAT84_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP NEUROMUSCULAR AGENTS | NEUROMUSCULAR AGENTS
`NDC_CAT85_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP NON-INSULIN ANTIDIABETICS | NON-INSULIN ANTIDIABETICS
`NDC_CAT86_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP OPHTHALMIC AGENTS | OPHTHALMIC AGENTS
`NDC_CAT87_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP OTHER ANALGESICS | OTHER ANALGESICS
`NDC_CAT88_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP OTIC AGENTS | OTIC AGENTS
`NDC_CAT89_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP OXYTOCICS | OXYTOCICS
`NDC_CAT90_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP PASSIVE IMMUNIZING AGENTS | PASSIVE IMMUNIZING AGENTS
`NDC_CAT91_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP PERIPHERAL OPIOID RECEPTOR ANTAGONISTS | PERIPHERAL OPIOID RECEPTOR ANTAGONISTS
`NDC_CAT94_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP PHARMACEUTICAL ADJUVANTS | PHARMACEUTICAL ADJUVANTS
`NDC_CAT95_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP PHOSPHATE BINDER AGENTS | PHOSPHATE BINDER AGENTS
`NDC_CAT96_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP PROGESTINS | PROGESTINS
`NDC_CAT97_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP PSYCHOTHERAPEUTIC AND NEUROLOGICAL AGENTS - MISC. | PSYCHOTHERAPEUTIC AND NEUROLOGICAL AGENTS - MISC.
`NDC_CAT98_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP RESPIRATORY AGENTS - MISC. | RESPIRATORY AGENTS - MISC.
`NDC_CAT99_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP SHORT BOWEL SYNDROME (SBS) AGENTS | SHORT BOWEL SYNDROME (SBS) AGENTS
`NDC_CAT100_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP THYROID AGENTS | THYROID AGENTS
`NDC_CAT101_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP TOXOIDS | TOXOIDS
`NDC_CAT102_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP URINARY ANTI-INFECTIVES | URINARY ANTI-INFECTIVES
`NDC_CAT103_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP URINARY ANTISPASMODICS | URINARY ANTISPASMODICS
`NDC_CAT104_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP VACCINES | VACCINES
`NDC_CAT105_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP VAGINAL PRODUCTS | VAGINAL PRODUCTS
`NDC_CAT106_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP VASOPRESSORS | VASOPRESSORS
`NDC_CAT107_SUM` | COUNT OF CLAIMS WITH CONDITION NDC DRUG GROUP VITAMINS | VITAMINS

NOTE: 
The code categories NDC_CAT81 (Non-EPI related Multivitamins), NDC_CAT92 (Digestive Aids - Creon) and NDC_CAT93 (Digestive Aids - Non-Creon) were excluded from the list of predictive features of the model. NDC_CAT92 and NDC_CAT93 were excluded because they were the features used to identify a patient as a true positive and NDC_CAT81 was excluded because it included drugs that were commonly over-the-counter.

### Milliman Contact

Maggie Alston    
maggie.alston@milliman.com

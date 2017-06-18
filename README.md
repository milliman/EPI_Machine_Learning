# Creon

This project is referenced in the memo:
_**reference memo here**_

### What is this repository for?

 This repository is used to train, evaluate, and use models to predict prevalence of EPI
 in a medical claims database.  It also contains additions to the Scikit-Learn Python library
 to experiment with oversampling / undersampling techniques for large class imbalances,
 nested cross validation, metrics for use with unlabeled data, and random search for hyper-parameters
 using multiple metrics at the same time.

### **Version 0.1**

### Requirements

The `environment.yml` file describes the environment needed to run the code.  Please use:
```
conda env create -f environment.yml
```
from Anaconda to create a compatible runtime environment for this project.

Please see [input data format](#Input-data-format) for a description of the data used to train models in this project.

### Main entry-point to use best models

Clone the repository and start using `creon.creonmain.py` with data generated into a tab separated value file.
CSV files would also suffice as long as a comma (_,_) is passed into the `sep` parameter of `generate_trained_model`.

Please see `Creonmain Example.ipynb` for an example of how to start using the library.

### Python File Descriptions
File | Description
---|---
`creonmain.py` | Main entrypoint to train, save, load, and use models
`loadcreon.py` | Used to load data and normalize / clean it for training
`modeldeepdive.py` | Code to generate graphs and analysis / descriptions of models generated in the project
`brestmodels.py` | Hard coded model(s) with parameters found to be good during various searches for hyper-parameters
`semisuperhelper.py` | Helper code to handle unlabeled data (labeled as `-1`)
`creonsklearn.creonmetrics.py` | Custom scoring metrics for models using unlabled data
`creonsklearn.frankenscorer.py` | An Sklearn scorer object that can score multiple metrics at once
`creonsklearn.jeffsearch.py` | A random search of hyper-parameters using a Frankenscorer
`creonsklearn.nestedcross.py` | A class that runs a nested cross validation search
`creonsklearn.pnuwrapper.py` | Wraps classifiers to be used with unlabeled data PNU = *P*ositive *N*egative *U*nlabled
`creonsklearn.repeatedsampling.py` | Wraps classifiers to be used with massively unbalanced data using repeated oversampling
`creonsklearn.rfsubsample.py` | A modified Random Forest algorithm where every bootstrapped sample used adheres to a _target imbalance ratio_, uses oversampling

### Notebook File Descriptions
File | Description
--- | ---
`Creonmain Example.ipynb` | Example code on how to use creonmain.py
`ModelDeepDive with Model6.ipynb` | Example code on how to use ModelDeepDive with _Model 6_ from the memo
`LASSO - PN and PNU.ipynb` | Code to generate the LASSO baseline model using only PN (Positive and Negative) data and then a search for a model using PNU (PN + Unlabeled) data which is undersampled
`SVC - PN and PNU.ipynb` | Code to generate a Support Vector Classifier (SVC) using only PN data and then a search for an SVC using PNU data which is understampled


### Input data format

Note that the code will run as long as there is a `Gender` column in the data along with 

The input dataset used in the study had this format:

SAS Flag Name | Long Name | Description
--- | --- | ---
`MemberID`| ID of patient | anonymous patient
`age` | Age of patient | How old patient is
`Gender` | 
`epi_related_cond` | 
`epi_related_cond_subgrp` | 
`pert_flag` | 
`mmos` | 
`h_rank` | 
`elastase_flag` | 
`true_pos_flag` | 
`true_neg_flag` | 
`unlabel_flag` | 
`medical_claim_count` | 
`rx_claim_count` | 
`DIAG_FLAG1_Sum` | 
`DIAG_FLAG2_Sum` | 
`DIAG_FLAG3_Sum` | 
`DIAG_FLAG4_Sum` | 
`DIAG_FLAG5_Sum` | 
`DIAG_FLAG6_Sum` | 
`DIAG_FLAG7_Sum` | 
`DIAG_FLAG8_Sum` | 
`DIAG_FLAG9_Sum` | 
`DIAG_FLAG10_Sum` | 
`DIAG_FLAG11_Sum` | 
`DIAG_FLAG12_Sum` | 
`DIAG_FLAG15_Sum` | 
`DIAG_FLAG16_Sum` | 
`DIAG_FLAG17_Sum` | 
`DIAG_FLAG21_Sum` | 
`DIAG_FLAG22_Sum` | 
`DIAG_FLAG23_Sum` | 
`DIAG_FLAG24_Sum` | 
`DIAG_FLAG25_Sum` | 
`DIAG_FLAG26_Sum` | 
`DIAG_FLAG27_Sum` | 
`DIAG_FLAG28_Sum` | 
`DIAG_FLAG29_Sum` | 
`DIAG_FLAG30_Sum` | 
`DIAG_FLAG31_Sum` | 
`DIAG_FLAG32_Sum` | 
`DIAG_FLAG33_Sum` | 
`DIAG_FLAG34_Sum` | 
`DIAG_FLAG35_Sum` | 
`DIAG_FLAG36_Sum` | 
`DIAG_FLAG37_Sum` | 
`DIAG_FLAG38_Sum` | 
`DIAG_FLAG39_Sum` | 
`DIAG_FLAG40_Sum` | 
`DIAG_FLAG41_Sum` | 
`DIAG_FLAG42_Sum` | 
`DIAG_FLAG45_Sum` | 
`DIAG_FLAG46_Sum` | 
`DIAG_FLAG47_Sum` | 
`DIAG_FLAG48_Sum` | 
`DIAG_FLAG49_Sum` | 
`DIAG_FLAG50_Sum` | 
`DIAG_FLAG51_Sum` | 
`DIAG_FLAG52_Sum` | 
`DIAG_FLAG53_Sum` | 
`DIAG_FLAG54_Sum` | 
`DIAG_FLAG55_Sum` | 
`DIAG_FLAG60_Sum` | 
`DIAG_FLAG61_Sum` | 
`DIAG_FLAG62_Sum` | 
`DIAG_FLAG63_Sum` | 
`DIAG_FLAG66_Sum` | 
`DIAG_FLAG67_Sum` | 
`DIAG_FLAG68_Sum` | 
`DIAG_FLAG69_Sum` | 
`DIAG_FLAG70_Sum` | 
`DIAG_FLAG71_Sum` | 
`DIAG_FLAG72_Sum` | 
`DIAG_FLAG73_Sum` | 
`DIAG_FLAG74_Sum` | 
`DIAG_FLAG75_Sum` | 
`DIAG_FLAG76_Sum` | 
`DIAG_FLAG77_Sum` | 
`DIAG_FLAG78_Sum` | 
`DIAG_FLAG79_Sum` | 
`DIAG_FLAG80_Sum` | 
`DIAG_FLAG81_Sum` | 
`DIAG_FLAG82_Sum` | 
`DIAG_FLAG83_Sum` | 
`DIAG_FLAG84_Sum` | 
`DIAG_FLAG85_Sum` | 
`DIAG_FLAG86_Sum` | 
`DIAG_FLAG87_Sum` | 
`PROC_FLAG1_Sum` | 
`PROC_FLAG2_Sum` | 
`PROC_FLAG3_Sum` | 
`PROC_FLAG4_Sum` | 
`PROC_FLAG5_Sum` | 
`CPT_FLAG1_Sum` | 
`CPT_FLAG2_Sum` | 
`CPT_FLAG3_Sum` | 
`CPT_FLAG4_Sum` | 
`CPT_FLAG5_Sum` | 
`CPT_FLAG6_Sum` | 
`CPT_FLAG7_Sum` | 
`CPT_FLAG8_Sum` | 
`CPT_FLAG9_Sum` | 
`CPT_FLAG10_Sum` | 
`CPT_FLAG11_Sum` | 
`CPT_FLAG12_Sum` | 
`CPT_FLAG13_Sum` | 
`CPT_FLAG14_Sum` | 
`CPT_FLAG15_Sum` | 
`CPT_FLAG16_Sum` | 
`CPT_FLAG17_Sum` | 
`CPT_FLAG18_Sum` | 
`CPT_FLAG19_Sum` | 
`CPT_FLAG20_Sum` | 
`CPT_FLAG21_Sum` | 
`CPT_FLAG22_Sum` | 
`CPT_FLAG26_Sum` | 
`CPT_FLAG29_Sum` | 
`CPT_FLAG32_Sum` | 
`CPT_FLAG33_Sum` | 
`CPT_FLAG34_Sum` | 
`CPT_FLAG35_Sum` | 
`CPT_FLAG36_Sum` | 
`CPT_FLAG37_Sum` | 
`CPT_FLAG43_Sum` | 
`CPT_FLAG44_Sum` | 
`CPT_FLAG45_Sum` | 
`CPT_FLAG46_Sum` | 
`CPT_FLAG47_Sum` | 
`CPT_FLAG48_Sum` | 
`CPT_FLAG49_Sum` | 
`REVCODE_FLAG1_Sum` | 
`REVCODE_FLAG2_Sum` | 
`REVCODE_FLAG3_Sum` | 
`REVCODE_FLAG4_Sum` | 
`REVCODE_FLAG5_Sum` | 
`REVCODE_FLAG6_Sum` | 
`REVCODE_FLAG7_Sum` | 
`REVCODE_FLAG8_Sum` | 
`REVCODE_FLAG9_Sum` | 
`REVCODE_FLAG10_Sum` | 
`REVCODE_FLAG11_Sum` | 
`REVCODE_FLAG12_Sum` | 
`REVCODE_FLAG13_Sum` | 
`REVCODE_FLAG14_Sum` | 
`REVCODE_FLAG15_Sum` | 
`REVCODE_FLAG16_Sum` | 
`REVCODE_FLAG17_Sum` | 
`REVCODE_FLAG18_Sum` | 
`REVCODE_FLAG19_Sum` | 
`REVCODE_FLAG20_Sum` | 
`REVCODE_FLAG21_Sum` | 
`REVCODE_FLAG22_Sum` | 
`REVCODE_FLAG23_Sum` | 
`REVCODE_FLAG24_Sum` | 
`REVCODE_FLAG25_Sum` | 
`REVCODE_FLAG26_Sum` | 
`REVCODE_FLAG27_Sum` | 
`REVCODE_FLAG28_Sum` | 
`REVCODE_FLAG29_Sum` | 
`REVCODE_FLAG30_Sum` | 
`REVCODE_FLAG31_Sum` | 
`REVCODE_FLAG32_Sum` | 
`REVCODE_FLAG33_Sum` | 
`REVCODE_FLAG34_Sum` | 
`REVCODE_FLAG35_Sum` | 
`REVCODE_FLAG36_Sum` | 
`REVCODE_FLAG37_Sum` | 
`REVCODE_FLAG38_Sum` | 
`REVCODE_FLAG39_Sum` | 
`REVCODE_FLAG40_Sum` | 
`REVCODE_FLAG41_Sum` | 
`REVCODE_FLAG42_Sum` | 
`REVCODE_FLAG43_Sum` | 
`REVCODE_FLAG44_Sum` | 
`REVCODE_FLAG45_Sum` | 
`REVCODE_FLAG46_Sum` | 
`REVCODE_FLAG47_Sum` | 
`REVCODE_FLAG48_Sum` | 
`REVCODE_FLAG49_Sum` | 
`REVCODE_FLAG50_Sum` | 
`REVCODE_FLAG51_Sum` | 
`REVCODE_FLAG52_Sum` | 
`REVCODE_FLAG53_Sum` | 
`REVCODE_FLAG54_Sum` | 
`REVCODE_FLAG55_Sum` | 
`REVCODE_FLAG56_Sum` | 
`REVCODE_FLAG57_Sum` | 
`REVCODE_FLAG58_Sum` | 
`REVCODE_FLAG59_Sum` | 
`REVCODE_FLAG60_Sum` | 
`REVCODE_FLAG61_Sum` | 
`REVCODE_FLAG62_Sum` | 
`REVCODE_FLAG63_Sum` | 
`REVCODE_FLAG64_Sum` | 
`REVCODE_FLAG65_Sum` | 
`REVCODE_FLAG66_Sum` | 
`REVCODE_FLAG67_Sum` | 
`REVCODE_FLAG68_Sum` | 
`REVCODE_FLAG69_Sum` | 
`REVCODE_FLAG70_Sum` | 
`ndc_cat1_Sum` | 
`ndc_cat2_Sum` | 
`ndc_cat3_Sum` | 
`ndc_cat4_Sum` | 
`ndc_cat5_Sum` | 
`ndc_cat6_Sum` | 
`ndc_cat7_Sum` | 
`ndc_cat8_Sum` | 
`ndc_cat9_Sum` | 
`ndc_cat10_Sum` | 
`ndc_cat11_Sum` | 
`ndc_cat12_Sum` | 
`ndc_cat13_Sum` | 
`ndc_cat14_Sum` | 
`ndc_cat15_Sum` | 
`ndc_cat16_Sum` | 
`ndc_cat17_Sum` | 
`ndc_cat18_Sum` | 
`ndc_cat19_Sum` | 
`ndc_cat20_Sum` | 
`ndc_cat21_Sum` | 
`ndc_cat22_Sum` | 
`ndc_cat23_Sum` | 
`ndc_cat24_Sum` | 
`ndc_cat25_Sum` | 
`ndc_cat26_Sum` | 
`ndc_cat27_Sum` | 
`ndc_cat28_Sum` | 
`ndc_cat29_Sum` | 
`ndc_cat30_Sum` | 
`ndc_cat31_Sum` | 
`ndc_cat32_Sum` | 
`ndc_cat33_Sum` | 
`ndc_cat34_Sum` | 
`ndc_cat35_Sum` | 
`ndc_cat36_Sum` | 
`ndc_cat37_Sum` | 
`ndc_cat38_Sum` | 
`ndc_cat39_Sum` | 
`ndc_cat40_Sum` | 
`ndc_cat41_Sum` | 
`ndc_cat42_Sum` | 
`ndc_cat43_Sum` | 
`ndc_cat44_Sum` | 
`ndc_cat45_Sum` | 
`ndc_cat46_Sum` | 
`ndc_cat47_Sum` | 
`ndc_cat48_Sum` | 
`ndc_cat49_Sum` | 
`ndc_cat50_Sum` | 
`ndc_cat51_Sum` | 
`ndc_cat52_Sum` | 
`ndc_cat53_Sum` | 
`ndc_cat54_Sum` | 
`ndc_cat55_Sum` | 
`ndc_cat56_Sum` | 
`ndc_cat57_Sum` | 
`ndc_cat58_Sum` | 
`ndc_cat59_Sum` | 
`ndc_cat60_Sum` | 
`ndc_cat61_Sum` | 
`ndc_cat62_Sum` | 
`ndc_cat63_Sum` | 
`ndc_cat64_Sum` | 
`ndc_cat65_Sum` | 
`ndc_cat66_Sum` | 
`ndc_cat67_Sum` | 
`ndc_cat68_Sum` | 
`ndc_cat69_Sum` | 
`ndc_cat70_Sum` | 
`ndc_cat71_Sum` | 
`ndc_cat72_Sum` | 
`ndc_cat73_Sum` | 
`ndc_cat74_Sum` | 
`ndc_cat75_Sum` | 
`ndc_cat76_Sum` | 
`ndc_cat77_Sum` | 
`ndc_cat78_Sum` | 
`ndc_cat79_Sum` | 
`ndc_cat80_Sum` | 
`ndc_cat82_Sum` | 
`ndc_cat83_Sum` | 
`ndc_cat84_Sum` | 
`ndc_cat85_Sum` | 
`ndc_cat86_Sum` | 
`ndc_cat87_Sum` | 
`ndc_cat88_Sum` | 
`ndc_cat89_Sum` | 
`ndc_cat90_Sum` | 
`ndc_cat91_Sum` | 
`ndc_cat94_Sum` | 
`ndc_cat95_Sum` | 
`ndc_cat96_Sum` | 
`ndc_cat97_Sum` | 
`ndc_cat98_Sum` | 
`ndc_cat99_Sum` | 
`ndc_cat100_Sum` | 
`ndc_cat101_Sum` | 
`ndc_cat102_Sum` | 
`ndc_cat103_Sum` | 
`ndc_cat104_Sum` | 
`ndc_cat105_Sum` | 
`ndc_cat106_Sum` | 
`ndc_cat107_Sum` |





### Repo owners / admins

 **Jeff Gomberg**  
 jgomberg@aibraintree.com  
 **Monica Son**  
 Monica.Son@Milliman.com  
 **Motoharu Dei**  
 Motoharu.Dei@Milliman.com
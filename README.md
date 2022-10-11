# BIOMRKS_CARMGLY
Repo containing all published code for the training and use of RCNN networks to extract heart and lung detections and segmentaitons. Then the use bounding boxes and masks to caluclate cardiomegaly biomarkers (CTR and CPAR) from chest x-ray samples in the MIMIC-CXR-JPG database.

This repo is linked to 3 papers:
- main paper
- data paper (biomarkers)
- data paper (masks)

It is split into 2 different subfolders 
- Biomarkers Extraction 
  - this folder contains all necessary instruction and scripts to train RCNN models and extract biomarker values from all posterior-anterior chest x-ray images in the MIMIC-CXR-JGP database
- Cardiomegaly Classification
  - this folder contains all necessary scripts and instruction and scripts to train XGBoost models to classify cardiomegaly samples based on CTR and CPAR values, as well as other non-imaging data
  

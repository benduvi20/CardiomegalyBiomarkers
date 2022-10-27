# Biomarker Extraction 

-- CONTEXT: 

This folder contains code to train and use RCNN segmentation and detection models to extract cardiothoracic ratio (CTR) and cardiopulmonary area ratio (CPAR) of posterior-anterior chest X-ray samples in MIMIC-CXR-JPG. CTR and CPAR values can then be used to classify cardiomegaly samples using scripts found in the ./Cardiomegaly Classification/ folder of this repo. More information on biomarkers and methods can be found in papers listed in usage notes. 

Different ensemble models (detection prioritized, segmentation prioritized, best score ensemble, average ensemble) are available when retrieving CTR values. CPAR values can only be obtained via a segmentation only set of models.   

These scripts use 3 sources of medical imaging databases: MIMIC-CXR-JPG, Japanese society of radiological technology (JSRT), Montgomery Tuberculosis (MGMY). These images are paired with lung and heart segmentation masks. For MIMIC-CXR-JPG - 200 segmentation masks are available on PhysioNet in an associated data paper (LINK DATA PAPER); for JSRT - segmentations are included in database; for MGMY - lung segmentation are available in the Segmentation of Chest radiographs (SCR) database, heart segmentation are available with IMC segmentations. Links available below.

-- USAGE NOTES:

If using this code please cite
- Multimodal Cardiomegaly Classification with Image-Derived Digital Biomarkers (https://doi.org/10.1007/978-3-031-12053-4_2)
- LINK DATA PAPER (biomarkers)
- LINK DATA PAPER (masks)


Please follow required file downloads and assumed file structure shown in main repo README file

--- REQUIRED FILE DOWNLOADS ---  

- MIMIC-CXR-JPG files and image database --> https://physionet.org/content/mimic-cxr-jpg/2.0.0/
- MIMIC-CXR files (images not needed) --> https://physionet.org/content/mimic-cxr/2.0.0/
- MIMIC-CXR masks --> LINK DATA PAPER (masks)

- Montgomery County image database --> https://doi.org/10.3978%2Fj.issn.2223-4292.2014.11.20 (paper); https://www.kaggle.com/datasets/raddar/tuberculosis-chest-xrays-montgomery (download)
- Montgomery County associated lung mask database --> https://doi.org/10.3978%2Fj.issn.2223-4292.2014.11.20
- Montgomery County heart masks --> LINK DATA PAPER (masks)

- JSRT image database --> https://www.ajronline.org/doi/pdf/10.2214/ajr.174.1.1740071 (paper); http://db.jsrt.or.jp/eng.php (download)
- SCR mask database for JSRT database --> https://www.isi.uu.nl/Research/Databases/SCR/index.php 

- support python files (engine.py and utils.py) from pytorch for RCNN models --> https://github.com/pytorch/vision/tree/main/references/detection

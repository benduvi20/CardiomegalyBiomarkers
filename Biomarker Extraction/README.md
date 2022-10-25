# Biomarker Extraction 

CONTEXT: 
This folder contains code to train and use RCNN segmentation and detection models to extract cardiothoracic ratio (CTR) and cardiopulmonary area ratio (CPAR) of posterior-anterior chest X-ray samples in MIMIC-CXR-JPG. CTR and CPAR values can then be used to classify cardiomegaly samples using scripts found in the ./Cardiomegaly Classification/ folder of this repo. More information on biomarkers and methods can be found in papers listed in usage notes. 

Different ensemble models (detection prioritised, segmentation prioritised, best score ensemble, average ensemble) are avalible when retreiving CTR values. CPAR values can only be obtained via a segmentaiton only set of models.   

These scrpits use 3 sources of medical imaging databases: MIMIC-CXR-JPG, Japanese scociety of radiological technology (JSRT), Montgomery Tuberculosis (MGMY). These images are paired with lung and heart segmentaion masks. For MIMIC-CXR-JPG - 200 segmentation masks are avalible on PhysioNet in an assocaited datapaper (); for JSRT - segmentations are included in database; for MGMY - lung segmentation are avalible in the Segmentation of Chest radiographs (SCR) database, heart segmentation are avalible with IMC segmentations. Links avalible below.


AUTHOR: 
Benjamin Duvieusart (University of Oxford, Department of Engineering Science) with particular acknowledgments to Felix Krones (University of Oxford, Oxford Internet Institute) for help in writing code and Adam Mahdi (University of Oxford, Oxford Institute) and Bartłomiej Papież (University of Oxford, Big Data Insitutue) for thier supervision and guidance.


USAGE NOTES: 
If using this code please cite
- Multimodal Cardiomegaly Classification with Image-Derived Digital Biomarkers (https://doi.org/10.1007/978-3-031-12053-4_2)
- LINK DATA PAPER (biomarkers)
- LINK DATA PAPER (masks)


Please follow required file downloads and assumed file structure shown in main repo README file

--- REQUIRED FILE DOWNLOADS ---  
Before using this folder please download the following and place into the
assumed file structure described in main repo README file

- MIMIC-CXR-JPG files and image database --> https://physionet.org/content/mimic-cxr-jpg/2.0.0/
- MIMIC-CXR files (images not needed) --> https://physionet.org/content/mimic-cxr/2.0.0/
- MIMIC-CXR masks --> LINK DATA PAPER (masks)

- Montgomery County image database --> https://doi.org/10.3978%2Fj.issn.2223-4292.2014.11.20 (paper); https://www.kaggle.com/datasets/raddar/tuberculosis-chest-xrays-montgomery (download)
- Montgomery County associated lung mask database --> https://doi.org/10.3978%2Fj.issn.2223-4292.2014.11.20
- Montgomery County heart masks --> LINK DATA PAPER (masks)

- JSRT image database --> https://www.ajronline.org/doi/pdf/10.2214/ajr.174.1.1740071 (paper); http://db.jsrt.or.jp/eng.php (download)
- SCR mask databse for JSRT database --> https://www.isi.uu.nl/Research/Databases/SCR/index.php 

- support python files (engine.py and utils.py) from pytorch for RCNN models --> https://github.com/pytorch/vision/tree/main/references/detection 
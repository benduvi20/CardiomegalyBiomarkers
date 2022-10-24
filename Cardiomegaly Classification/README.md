#Cardiomegaly Classification

CONTEXT: 
This folder contains code to train and use RCNN segmentation and detection models to extract cardiothoracic ratio (CTR) and cardiopulmonary area ratio (CPAR) of posterior-anterior chest X-ray samples in MIMIC-CXR-JPG. CTR and CPAR values can then be used to classify cardiomegaly samples. 

Different ensemble models (detection prioritised, segmentation prioritised, best score ensemble, average ensemble) are avalible when retreiving CTR values. CPAR values can only be obtained via a segmentaiton only set of models.   

These scrpits use 3 sources of medical imaging databases: MIMIC-CXR-JPG, Japanese scociety of radiological technology (JSRT), Montgomery Tuberculosis (MGMY). These images are paired with lung and heart segmentaion masks. For MIMIC-CXR-JPG - 200 segmentation masks are avalible on PhysioNet in an assocaited datapaper (); for JSRT - segmentations are included in database; for MGMY - lung segmentation are avalible in teh Segmentation of Chest radiographs (SCR) database, heart segmentation are avalible with IMC segmentations. Links avalible below.


AUTHORS: 
This folder was authored by Benjamin Duvieusart (University of Oxford, Department of  Engineering Science) with with significant contributions from Felix Krones (University of Oxford, Oxford Internet Institute) and Declan Grant (Univeristy of Oxford, Department of Engineering Science). Particular acknowledgement is given to Adam Mahdi (University of Oxford, Oxford Institute), Bartłomiej Papież (University of Oxford, Big Data Insitutue) and Dr Guy Parsons (Intensive Care Registrar, Thames Valley Deanery, NIHR Academic Clinical Fellow at Oxford University) for thier supervision and guidance.


USAGE NOTES: 
If using this code please cite
- https://doi.org/10.1007/978-3-031-12053-4_2
- LINK DATA PAPER (biomarkers)

Please follow required file downloads and assumed file structure shown in main repo README file

--- REQUIRED FILE DOWNLOADS ---  
Before using this folder please download the following and place into the assumed file structure described below

- MIMIC-CXR-JPG files and image database --> https://physionet.org/content/mimic-cxr-jpg/2.0.0/
- MIMIC-CXR files (images not needed) --> https://physionet.org/content/mimic-cxr/2.0.0/
- MIMIC-IV files --> https://physionet.org/content/mimiciv/2.0/ 

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
  

--- ASSUMED FILE STRUCTURE ---
  
    ***folder containing code for biomarker extraction via detection and segmentaion models***
    ./Biomarker Extraction
        ***script to train models***
        /heart_detection.py 
        /heart_segmentation.py
        /lung_detection.py
        /lung_segmentation.py

        ***scripts to extract biomarkers (CTR and CPAR) from PA MIMIC-CXR-JPG images***
        /CPAR_full_MIMIC.py
        /CTR_full_MIMIC.py

        ***folders containing proposed training, testing, and validation sets (combination of MIMIC, JSRT, and MGMY images)***
        /heart_detection/
        /heart_segmentation/
        /lung_detection/
        /lung_segmentation/

        ***folder with python files containing engine.py (for train_one_epoch fucntion) and utils.py fucntion scripts from pytorch, plus empty __init__.py file***
        /support/

    
    ***folder containing code for calssification of cardiomegaly samples using biomarkers values and other methods***
    ./Cardiomegaly Calssification



    ***folder for MIMIC database containing files from MIMIC-CXR and MIMIC-CXR-JPG files, images, and masks***
    ./MIMIC 
        /mimic-cxr-2.0.0-metadata.csv.gz
        /cxr-record-list.csv.gz
        
        /files/
            ***x-ray images in folder structure imported from MIMIC-CXR-JPG database***
        /masks
            /heart/
            /lungs/


    ***folders for Montgomery database containing images and masks***
    ./MGMY
        /CXR_png/
            ***folder with images from Montgomery database***
        /masks
            /heart/
            /leftMask/
            /rightMask/
            

    ***folders for JSRT database containing images and masks***
    ./JSRT
        /images/
            ***folder with all images from JSRT database***
        /masks
            /heart/
            /left_lung/
            /right_lung/

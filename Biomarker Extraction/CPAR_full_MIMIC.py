### Before using this please read READ_ME file ###

import pandas as pd
from PIL import Image, ImageDraw
from skimage.filters import threshold_otsu
import torchvision.transforms as transforms

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import maskrcnn_resnet50_fpn

import torch
import numpy as np
import matplotlib.pyplot as plt\



## INITIATION

view = 'PA'            # relevant x-ray view
input_size = (256,256) # shape of image when input to model
score_threshold = 0.7  # minimum confidence score for detection/segmentation to be accepted
num_classes = 2        # number of classes in models (taget - heart or lung - and background)

# path for file containing DICOM image metadata (incl. x-ray view) -- from MIMIC-CXR-JPG
cxr_metadata_path = '../MIMIC/mimic-cxr-2.0.0-metadata.csv.gz'  

# path for file which links DICOM IDs to image filepaths in MIMIC -- from MIMIC-CXR
cxr_records_path = '../MIMIC/cxr-record-list.csv.gz' 

# path to MIMIC-CXR-JPG database
data_path = '../MIMIC/'  

# paths for model loading
heart_model_path = '../Biomarker Extraction/heart_segmentation/model.pt'
lung_model_path = '../Biomarker Extraction/lung_segmentation/model.pt'

# save_folder
save_path = '../Biomarker Extraction/save_folder/CPAR/'

# set device to GPU if avalible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# read chest x-ray metadata and select only samples in posterior-anterior (PA) view
cxr_metadata = pd.read_pickle(cxr_metadata_path)
cxr_metadata_PA = cxr_metadata.loc[cxr_metadata['ViewPosition'] == view]

# read chest x-ray records and extract filepaths for PA images
cxr_records = pd.read_csv(cxr_records_path) 
all_data = cxr_records.loc[cxr_records['dicom_id'].isin(cxr_metadata_PA['dicom_id'].tolist())] 
all_data.reset_index(drop=True, inplace=True) 
 


## FUNCTIONS

# function for data noramlisation for image preporcessing before input to model
def get_transforms():
    transform = [transforms.ToTensor(),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    return transforms.Compose(transform)


# function to load and initialise segmentation model 
def _get_segmentation_model(num_classes):

    model = maskrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    hidden_layer = 256
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    
    return model

# function to retreieve bounding box from a binary segmentation mask
def get_bounding_box(mask_np):

    # set intial boundbounding box inversed extremes of image
    bbox = [input_size[0], input_size[1], 0, 0] #[col min, row min, col max, row max]

    # transpose image and cycle through columns to find relevant extremes
    for i, col in enumerate(mask_np.transpose()):
        if any(pixel for pixel in col):
            if i < bbox[0]:
                bbox[0] = i
            elif i > bbox[2]:
                bbox[2] = i

    # cycle through rows to find relevant extremes
    for j, row in enumerate(mask_np):
        if any(pixel for pixel in row):
            if j < bbox[1]:
                bbox[1] = j
            elif j > bbox[3]:
                bbox[3] = j

    return bbox
    
# function to generate final segmentaiton prediction using segmentation models
def get_pred(img, model, score_threshold):
    model_out = model(img)
    
    # Get segmentation prediction
    if len(model_out[0]['masks']): 
      
        # retreive mask prediction, and transform to PIL image
        mask = model_out[0]['masks'][0]
        mask_img = np.array(transforms.functional.to_pil_image(mask))

        # generate binary mask
        threshold = threshold_otsu(mask_img)
        mask_binary = mask_img > threshold
        
        # retreive confidence score, bouding box from mask, and mask area
        score = model_out[0]['scores'][0].item()
        bbox = get_bounding_box(np.array(mask_binary))
        area = np.sum(mask_binary)
        
        if score > score_threshold:
            return score, bbox, area, mask_binary

    return False, False, False, False

# function to count number of CPAR ratios within a range
def count_CPAR(ratio_list, min, max):
    count = 0

    for i, ratio in enumerate(ratio_list):
        if min < ratio <= max:
            count += 1  

    return count



## GET MODELS

# retreive and load heart segmentation model into device
model_heart = _get_segmentation_model(num_classes)
model_heart.load_state_dict(torch.load(heart_model_path, map_location=device))
model_heart.to(device)
model_heart.eval()

# retreive and load lungs segmentation model into device
model_lung = _get_segmentation_model(num_classes)
model_lung.load_state_dict(torch.load(lung_model_path, map_location=device))
model_lung.to(device)
model_lung.eval()

print('models loaded')



## FIND CPARs

# retreive normalisation transforms
transform = get_transforms()

# initialise lists for CPAR values and image IDs
CPAR = []
img_id = []

# extract segmetentation and detection predictions, and calcualte CPAR values for each PA image in all_data  
for i, row in all_data.iterrows():

    # get image ID
    img_id.append(row['dicom_id'])
    
    # open image and convert to approprite path (replace endting from DICOM to jgp), noramlised image an generate ImageDraw version of image
    image = Image.open(data_path + row['path'][:-3] + 'jpg').resize(input_size).convert('RGB')
    img = [transform(image).to(device)]
   
    # extract model predictions for heart and lung segmentation
    heart_score, heart_bbox, heart_area, heart_mask_binary = get_pred(img, model_heart, score_threshold = score_threshold)
    lung_score, lung_bbox, lung_area, lung_mask_binary = get_pred(img, model_lung, score_threshold = score_threshold)

    # for every 500th image draw prediction heart and lung masks and save image
    if (i % 500 == 0):
        
        # if heart prediction is avalible, draw heart segmentation mask in partially opaque red
        if heart_bbox:
            filler = Image.new('RGBA', image.size, color=(255,63,63,40))
            blank = Image.new('RGBA', image.size, color=(0,0,0,0))
            mask_tinted = Image.composite(filler, blank, Image.fromarray(heart_mask_binary))
            
            image = Image.alpha_composite(image.convert('RGBA'), mask_tinted).convert('RGB')
    
        # if lungs prediction is avalible, draw lungs segmentation mask in partially opaque blue
        if lung_bbox:
            filler = Image.new('RGBA', image.size, color=(63,63,255,40))
            blank = Image.new('RGBA', image.size, color=(0,0,0,0))
            mask_tinted = Image.composite(filler, blank, Image.fromarray(lung_mask_binary))
            
            image = Image.alpha_composite(image.convert('RGBA'), mask_tinted).convert('RGB')
            
        image.save(f'{save_path}image_{i}.png')

    # calcualte CPAR value if both lung and heart detection is avalible
    if heart_bbox and lung_bbox: 
        CPAR_patient = abs(heart_area/lung_area)
        CPAR.append(CPAR_patient)

    # if only lungs detection is avalible: error code 2
    elif not heart_bbox and lung_bbox: 
        CPAR.append(2)

    # if only heart detection is avalible: error code 3
    elif heart_bbox and not lung_bbox: 
        CPAR.append(3)

    # if neither heart nor lungs detections are avalible: error code 4
    elif not heart_bbox and not lung_bbox: 
        CPAR.append(4)

# generate and save csv file with dicom IDs and CPAR values
CPAR_pd = pd.DataFrame([[patient_img_id, patient_CPAR] for patient_img_id, patient_CPAR in zip(img_id, CPAR)], columns = ['dicom_id', 'CPAR'])
CPAR_pd.to_csv(save_path + 'CPARs.csv')

print('CPARs collected!')



## PROCESS CPARs

# generate CPAR ratio brackets
ratio_brackets = [iter*5/100 for iter in range(int(100/5)+1)]

# initiliase lists for CPAR value counts per range of CPAR values, and for CPAR values 
CPAR_counts = []
CPAR_values = []

# initiate error code counts
CPAR_missing_heart = 0
CPAR_missing_lung = 0
CPAR_missing_both = 0

# complete counts of ratio in each range of CPAR values
for i, ratio in enumerate(ratio_brackets):
    if i != 0:
        count = count_CPAR(CPAR, ratio_brackets[i-1], ratio)
        CPAR_counts.append(count)

# complete count of error code and collection of CPAR values
for i, ratio in enumerate(CPAR):
    if ratio == 2:
        CPAR_missing_heart += 1
    
    elif ratio == 3:
        CPAR_missing_lung += 1
        
    elif ratio == 4:  
        CPAR_missing_both += 1
    
    else:
        CPAR_values.append(ratio)


            
## PLOTS

# frist plot: bar plot of CPAR value distribution
barWidth = 1
ticks = [r for r in range(len(ratio_brackets))]
bar_ticks = [tick + barWidth/2 for tick in ticks[:-1]]

plt.figure(dpi=500)
plt.bar(bar_ticks, CPAR_counts, width = barWidth, edgecolor='white')

plt.ylabel('No. of Patients')
plt.xlabel('Cardiothoracic ratio')
plt.xticks(ticks, 
            ratio_brackets, 
            rotation=45)

for idx, CPAR_count in enumerate(CPAR_counts):
    if CPAR_count != 0:
        plt.text(bar_ticks[idx], CPAR_count, str(CPAR_count), horizontalalignment = 'center', rotation = 80)

plt.legend()
plt.savefig(save_path + 'CPAR_bar_plot1.png', dpi = 800)


# second plot: bar plot of error code prevalence
plt.figure(dpi=500)
plt.bar([1],[CPAR_missing_heart], width = 0.75, edgecolor = 'white', label = 'heart')
plt.bar([2],[CPAR_missing_lung], width = 0.75, edgecolor = 'white', label = 'lung')
plt.bar([3],[CPAR_missing_both], width = 0.75, edgecolor = 'white', label = 'both')
plt.xlabel('Class')
plt.ylabel('No. of Patients')
plt.xticks([1, 2, 3], 
            ['Missing Heart', 'Missing Lung', 'Missing Both'])

plt.text(1, CPAR_missing_heart, str(CPAR_missing_heart), horizontalalignment = 'center')
plt.text(2, CPAR_missing_lung, str(CPAR_missing_lung), horizontalalignment = 'center')
plt.text(3, CPAR_missing_both, str(CPAR_missing_both), horizontalalignment = 'center')

plt.legend()
plt.savefig(save_path + 'CPAR_bar_plot2.png', dpi = 800)


# third plot: box plot of CPAR value distribution
fig = plt.figure(dpi=500)
ax = fig.add_subplot(111)
bp = plt.boxplot(CPAR_values)

ax.axhline(0.5, color='r')
plt.ylabel('CPAR')
plt.ylim(0,1)
plt.savefig(save_path + 'CPAR_box_plot.png', dpi = 800)

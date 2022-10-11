### Before using this please read READ_ME file ###

import pandas as pd
from PIL import Image, ImageDraw
from skimage.filters import threshold_otsu
import torchvision.transforms as transforms

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import maskrcnn_resnet50_fpn, fasterrcnn_resnet50_fpn

import torch
import numpy as np
import matplotlib.pyplot as plt\



## INITIATION

Label = 'Cardiomegaly' # relevant condition
view = 'PA'            # relevant view
input_size = (256,256) # shape of image when input to model
score_threshold = 0.7  # minimum confidence score for detection/segmentation to be accepted
num_classes = 2        # number of classes in models (taget - heart or lung - and background)

# model types avalible are : 'det', 'seg', 'avg', 'bst'
    # det: prioritises detection model
    # seg: prioritises segmentaiton model
    # avg: averages bounding box extrated from detection and segmentation networks
    # bst: uses prediction with highest confidence score 
lung_model_type = 'avg'
heart_model_type = 'seg'

# path for file containing DICOM image metadata (incl. x-ray view) -- from MIMIC-CXR-JPG
cxr_metadata_path = './MIMIC/mimic-cxr-2.0.0-metadata.csv.gz'  

# path for file which links DICOM IDs to image filepaths in MIMIC -- from MIMIC-CXR
cxr_records_path = './MIMIC/cxr-record-list.csv.gz' 

# path to MIMIC-CXR-JPG database
data_path = './MIMIC'                  

# paths for model loading
seg_heart_model_path = './heart_segmentation/model.pt'
seg_lung_model_path = './lung_segmentation/model.pt'
det_heart_model_path = './heart_detection/model.pt'
det_lung_model_path = './lung_detection/model.pt'

# save_folder
save_path = './save_folder/'

# set device to GPU if avalible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# read chest x-ray metadata and select only samples in posterior-anterior (PA) view
cxr_metadata = pd.read_pickle(cxr_metadata_path)
cxr_metadata_PA = cxr_metadata.loc[cxr_metadata['ViewPosition'] == 'PA']

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

# function to load and initialise detection model 
def _get_detection_model(num_classes):
    
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
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
    
# function to generate final bounding box prediction using ensemble models (segmentation based, detection based, best score, average)
def get_pred(img, seg_model, det_model, score_threshold, model_type):
    seg_pred, det_pred = False, False

    seg_model_out = seg_model(img)
    det_model_out = det_model(img)
    
    # Get segmentation prediction
    if len(seg_model_out[0]['masks']):
        
        seg_score = seg_model_out[0]['scores'][0].item()
        if seg_score >= score_threshold:
            seg_pred = True

            # retreive mask prediction, transform to PIL image, and generate binary mask
            mask = seg_model_out[0]['masks'][0]
            mask_img = np.array(transforms.functional.to_pil_image(mask))
            seg_threshold = threshold_otsu(mask_img)
            mask_binary = mask_img > seg_threshold
            
            # retreive boundingbox from mask
            seg_bbox = get_bounding_box(np.array(mask_binary))
      
    # Get detection prediction
    if len(det_model_out[0]['scores']):
        
        det_score = det_model_out[0]['scores'][0].item()
        if det_score >= score_threshold:
            det_pred = True
        
            det_bbox = det_model_out[0]['boxes'][0].tolist()

    # Average model: return point wise average of seg and det if possible
    if model_type == 'avg': 
        if seg_pred and det_pred: 
            combined_score = (det_score+seg_score)/2
            combined_bbox = [(coord_det+coord_seg)/2 for (coord_det, coord_seg) in zip(det_bbox, seg_bbox)]
            return combined_score, combined_bbox

        elif seg_pred:
            return seg_score, seg_bbox

        elif det_pred:
            return det_score, det_bbox
            
        else:
            return False, False

    # Best score model: return prediction (det or seg) with highest score if possible
    elif model_type == 'bst': 
        if seg_pred and det_pred: 
            best_score = max(seg_score, det_score)
            if best_score == seg_score:  
                return seg_score, seg_bbox
            else:
                return det_score, det_bbox

        elif seg_pred:
            return seg_score, seg_bbox

        elif det_pred:
            return det_score, det_bbox
            
        else:
            return False, False

    # Detection based model: return detection prediction if possible, else try segmentation
    elif model_type == 'det':
        if det_pred: 
            return det_score, det_bbox

        elif seg_pred:
            return seg_score, seg_bbox
        
        else:
            return False, False

    # Segmentation based model: return segmentation prediction if possible, else try detection
    elif model_type == 'seg': 
        if seg_pred:
            return seg_score, seg_bbox

        elif det_pred: 
            return det_score, det_bbox
        
        else:
            return False, False
    
    else: 
        print('invalid model type')
        return False, False

# function to count number of CTR ratios within a range
def count_CTR(ratio_list, min, max):
    count = 0

    for i, ratio in enumerate(ratio_list):
        if min < ratio <= max:
            count += 1  

    return count



## GET MODELS

# retreive and load heart segmentation model into device
seg_model_heart = _get_segmentation_model(num_classes)
seg_model_heart.load_state_dict(torch.load(seg_heart_model_path, map_location=device))
seg_model_heart.to(device)
seg_model_heart.eval()

# retreive and load lungs segmentation model into device
seg_model_lung = _get_segmentation_model(num_classes)
seg_model_lung.load_state_dict(torch.load(seg_lung_model_path, map_location=device))
seg_model_lung.to(device)
seg_model_lung.eval()

# retreive and load heart detection model into device
det_model_heart = _get_detection_model(num_classes)
det_model_heart.load_state_dict(torch.load(det_heart_model_path, map_location=device))
det_model_heart.to(device)
det_model_heart.eval()

# retreive and load lungs detection model into device
det_model_lung = _get_detection_model(num_classes)
det_model_lung.load_state_dict(torch.load(det_lung_model_path, map_location=device))
det_model_lung.to(device)
det_model_lung.eval()

print('models loaded')



## FIND CTRs

# retreive normalisation transforms
transform = get_transforms()

# initialise lists for CTR values and image IDs
CTR = []
img_id = []

# extract segmetentation and detection predictions, and calcualte CTR values for each PA image in all_data  
for i, row in all_data.iterrows():
  
    # get image ID
    img_id.append(row['dicom_id'])
    
    # open image and convert to approprite path (replace endting from DICOM to jgp), noramlised image an generate ImageDraw version of image
    image = Image.open(data_path + row['path'][:-3] + 'jpg').resize(input_size).convert('RGB')
    img = [transform(image).to(device)]
    draw = ImageDraw.Draw(image, mode='RGBA')

    # extract model predictions for heart and lung detection
    heart_score, heart_bbox = get_pred(img, seg_model_heart, det_model_heart, score_threshold, heart_model_type)
    lung_score, lung_bbox = get_pred(img, seg_model_lung, det_model_lung, score_threshold, lung_model_type)

    # for every 500th image draw prediction heart and lung bounding box and save image
    if (i % 500 == 0):

        # if heart prediction is avalible, draw heart bounding box in partially opaque red
        if heart_bbox:
            draw.rectangle(heart_bbox, outline='red')

        # if lungs prediction is avalible, draw lungs bounding box in partially opaque blue
        if lung_bbox:
            draw.rectangle(lung_bbox, outline='blue')
            
        image.save(f'{save_path}image_{i}.png')

    # calcualte CTR value if both lung and heart detection is avalible
    if heart_bbox and lung_bbox: 
        CTR_patient = abs((heart_bbox[2] - heart_bbox[0])/(lung_bbox[2] - lung_bbox[0]))
        CTR.append(CTR_patient)

    # if only lungs detection is avalible: error code 2
    elif not heart_bbox and lung_bbox: 
        CTR.append(2)
        
    # if only heart detection is avalible: error code 3
    elif heart_bbox and not lung_bbox: 
        CTR.append(3)
        
    # if neither heart nor lungs detections are avalible: error code 4
    elif not heart_bbox and not lung_bbox: 
        CTR.append(4)

# generate and save csv file with dicom IDs and CTR values
CTR_pd = pd.DataFrame([[patient_img_id, patient_CTR] for patient_img_id, patient_CTR in zip(img_id, CTR)], columns = ['dicom_id', 'CTR'])
CTR_pd.to_csv(save_path + 'CTRs.csv')

print('CTRs collected!')



## PROCESS CTRs

# generate CTR ratio brackets
ratio_brackets = [iter*5/100 for iter in range(int(100/5)+1)]

# initiliase lists for CTR value counts per range of CTR values, and for CTR values 
CTR_counts = []
CTR_values = []

# initiate error code counts
CTR_missing_heart = 0
CTR_missing_lung = 0
CTR_missing_both = 0

# complete counts of ratio in each range of CTR values
for i, ratio in enumerate(ratio_brackets):
    if i != 0:
        count = count_CTR(CTR, ratio_brackets[i-1], ratio)
        CTR_counts.append(count)

# complete count of error code and collection of CTR values
for i, ratio in enumerate(CTR):
    if ratio == 2:
        CTR_missing_heart += 1
    
    elif ratio == 3:
        CTR_missing_lung += 1
        
    elif ratio == 4:  
        CTR_missing_both += 1
    
    else:
        CTR_values.append(ratio)


            
## PLOTS

# frist plot: bar plot of CTR value distribution
barWidth = 1
ticks = [r for r in range(len(ratio_brackets))]
bar_ticks = [tick + barWidth/2 for tick in ticks[:-1]]

plt.figure(dpi=500)
plt.bar(bar_ticks, CTR_counts, width = barWidth, edgecolor='white')

plt.ylabel('No. of Patients')
plt.xlabel('Cardiothoracic ratio')
plt.xticks(ticks, 
            ratio_brackets, 
            rotation=45)

for idx, CTR_count in enumerate(CTR_counts):
    if CTR_count != 0:
        plt.text(bar_ticks[idx], CTR_count, str(CTR_count), horizontalalignment = 'center', rotation = 80)

plt.legend()
plt.savefig(save_path + 'CTR_bar_plot1.png', dpi = 800)


# second plot: bar plot of error code prevalence
plt.figure(dpi=500)
plt.bar([1],[CTR_missing_heart], width = 0.75, edgecolor = 'white', label = 'heart')
plt.bar([2],[CTR_missing_lung], width = 0.75, edgecolor = 'white', label = 'lung')
plt.bar([3],[CTR_missing_both], width = 0.75, edgecolor = 'white', label = 'both')
plt.xlabel('Class')
plt.ylabel('No. of Patients')
plt.xticks([1, 2, 3], 
            ['Missing Heart', 'Missing Lung', 'Missing Both'])

plt.text(1, CTR_missing_heart, str(CTR_missing_heart), horizontalalignment = 'center')
plt.text(2, CTR_missing_lung, str(CTR_missing_lung), horizontalalignment = 'center')
plt.text(3, CTR_missing_both, str(CTR_missing_both), horizontalalignment = 'center')

plt.legend()
plt.savefig(save_path + 'CTR_bar_plot2.png', dpi = 800)


# third plot: box plot of CTR value distribution
fig = plt.figure(dpi=500)
ax = fig.add_subplot(111)
bp = plt.boxplot(CTR_values)

ax.axhline(0.5, color='r')
plt.ylabel('CTR')
plt.ylim(0,1)
plt.savefig(save_path + 'CTR_box_plot.png', dpi = 800)
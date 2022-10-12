from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import maskrcnn_resnet50_fpn

from support.engine import train_one_epoch
import support.utils as utils 

import torchvision.transforms as transforms
from skimage.filters import threshold_otsu
from PIL import Image, ImageDraw

import pandas as pd
import numpy as np
import random
import torch
import math
import os

## PATHS
base_path = '../Biomarker Extraction/lung_segmentation/'
save_path = '../Biomarker Extraction/lung_segmentation/'

## VARIABLES
# data pre-processing
input_size = (256,256)
Hflip_prob = 0.4
Vflip_prob = 0.2
max_rot = 10

# model
num_classes = 2  # object (foreground); background
num_epochs = 300
batch_size = 32 

# optimizer variables
lr=0.001
weight_decay=0.0005

# lr_scheduler variables
factor = 0.5
patience = 15

# validation variable
iou_loss_threshold = 0.15 # maximum IOU loss for segmentation prediction to be considered a TP sampl

# load training, validation, and testing dataset
train = pd.read_pickle(base_path + 'train.pkl')
val = pd.read_pickle(base_path + 'val.pkl')
test = pd.read_pickle(base_path + 'test.pkl')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


### DATA NORMALISATION AND AUGMENTATION

# class for random vertical flips
class RandomVerticalFlip(torch.nn.Module):

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    # funtion to perform flips; returns flipped image and V flip flag
    def forward(self, img):        
        Vflip_flag = False
        if torch.rand(1) < self.p:
            Vflip_flag = True
            return transforms.functional.vflip(img), Vflip_flag
        return img, Vflip_flag

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


# class for random horizontal flips 
class RandomHorizontalFlip(torch.nn.Module):

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    # funtion to perform flips; returns flipped image and H flip flag
    def forward(self, img):
        Hflip_flag = False
        if torch.rand(1) < self.p:
            Hflip_flag = True
            return transforms.functional.hflip(img), Hflip_flag
        return img, Hflip_flag

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


# class which creates augmented training data with augmentation flags
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    # funtion to perform augmentation; returns augmented image and augmentation flag
    def __call__(self, img, datatype):
        for t in self.transforms:
            if datatype == 'train':
                
                if isinstance(t, RandomHorizontalFlip):
                    img, Hflip_flag = t(img)
                
                elif isinstance(t, RandomVerticalFlip):
                    img, Vflip_flag = t(img)
                
                elif isinstance(t, transforms.RandomRotation):
                    img = t(img)
                    angle = t.get_params(t.degrees)
                
                else:
                    img = t(img)
            else:
                img = t(img)
                [Hflip_flag, Vflip_flag, angle] = [False, False, 0]

        return img, [Hflip_flag, Vflip_flag, angle]

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
        
# function to retreive data augmentation and normalisation
def get_transforms(datatype):
    transform = [transforms.ToTensor(),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    if datatype == 'train':
        transform.append(RandomHorizontalFlip(Hflip_prob))
        transform.append(RandomVerticalFlip(Vflip_prob))
        transform.append(transforms.RandomRotation(max_rot))
    return Compose(transform)
    

# funtion to transform bounding boxes depending on augmentation flags
def transform_bbox(bbox, size, flags):
    
    [Hflip_flag, Vflip_flag, angle] = flags

    if Hflip_flag:
        bbox = [size[0] - bbox[2], 
                bbox[1], 
                size[0] - bbox[0], 
                bbox[3]]

    if Vflip_flag:
        bbox = [bbox[0], 
                size[1] - bbox[3],
                bbox[2], 
                size[1] - bbox[1]]

    if angle:
        angle_rad = torch.tensor(math.radians(angle))
        rotation_matrix = torch.tensor([[torch.cos(angle_rad), -torch.sin(angle_rad)],
                                        [torch.sin(angle_rad),  torch.cos(angle_rad)]])

        corners = torch.tensor([[bbox[0], bbox[1]], 
                                [bbox[0], bbox[3]], 
                                [bbox[2], bbox[1]], 
                                [bbox[2], bbox[3]]])

        centre = torch.tensor([[size[0]/2, size[1]/2], 
                               [size[0]/2, size[1]/2], 
                               [size[0]/2, size[1]/2], 
                               [size[0]/2, size[1]/2]])

        new_corners = torch.matmul((corners - centre), rotation_matrix) + centre
        
        bbox = [max(min([corner[0] for corner in new_corners.tolist()]), 1),
                max(min([corner[1] for corner in new_corners.tolist()]), 1),
                min(max([corner[0] for corner in new_corners.tolist()]), size[0]-1),
                min(max([corner[1] for corner in new_corners.tolist()]), size[1]-1)]
    
    return torch.tensor([bbox])
    
# funtion to transform segmentation masks depending on augmentation flags
def transform_masks(mask, flags):
    
    [Hflip_flag, Vflip_flag, angle] = flags

    if Hflip_flag:
        mask = transforms.functional.hflip(mask)
    if Vflip_flag:
        mask = transforms.functional.vflip(mask)
    if angle:
        mask = transforms.functional.rotate(mask, angle)

    return mask


# function to retreive bounding boxes from segmentation mask    
def get_bounding_box(mask_np):
    bbox = [input_size[0], input_size[1], 0, 0] #[col min, row min, col max, row max]

    for i, col in enumerate(mask_np.transpose()):
        if any(pixel for pixel in col):
            if i < bbox[0]:
                bbox[0] = i
            elif i > bbox[2]:
                bbox[2] = i

    for j, row in enumerate(mask_np):
        if any(pixel for pixel in row):
            if j < bbox[1]:
                bbox[1] = j
            elif j > bbox[3]:
                bbox[3] = j

    return bbox 
    


### BUILD DATASET

# class to build dataset to link images to target segmentation masks
class Dataset(object):
    def __init__(self, data, datatype):
        self.data = data
        self.datatype = datatype
        self.transform = get_transforms(datatype)

    def __getitem__(self, idx):
        img_path = self.data['image'][idx]
        img = Image.open(img_path).convert('RGB').resize(input_size)
        img_id = torch.tensor(idx)

        # combine masks and place into binary field
        mask_L_path = self.data['mask left'][idx]
        mask_R_path = self.data['mask right'][idx]
        mask_L = Image.open(mask_L_path).resize(input_size)
        mask_R = Image.open(mask_R_path).resize(input_size)
        mask_comb = np.array(mask_L)+np.array(mask_R)
        mask_binary = np.where(mask_comb != 0, 1, 0)
        masks = mask_binary == [[[1]]]
        masks = torch.as_tensor(masks, dtype=torch.uint8)        
        
        # only one box per image
        num_boxes = 1
        boxes = torch.as_tensor([self.data['bbox both'][idx]], dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # only one label, and assuming all instances are not crowd
        labels = torch.ones(num_boxes, dtype=torch.int64)
        iscrowd = torch.zeros(num_boxes, dtype=torch.int64)
        
        target = {}
        target['boxes'] = boxes
        target['masks'] = masks
        target['labels'] = labels
        target['image_id'] = img_id
        target['area'] = area
        target['iscrowd'] = iscrowd

        if self.transform is not None:
            img, flags = self.transform(img, self.datatype)

            if self.datatype == 'train': 
                target['masks'] = transform_masks(target['masks'], flags)
                for i, box in enumerate(target['boxes']):
                    target['boxes'][i] = transform_bbox(box.tolist(), input_size, flags)
        
        return img, target

    def __len__(self):
        return len(self.data)

# generate datasets for training, validation, and testing
dataset_train = Dataset(train, datatype = 'train')
dataset_val = Dataset(val, datatype = 'val')
dataset_test = Dataset(test, datatype = 'test')

# put datasets into dataloader for model training and testing
data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle= True, num_workers=2, collate_fn=utils.collate_fn)
data_loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle= True, num_workers=2, collate_fn=utils.collate_fn)
data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle= True, num_workers=2, collate_fn=utils.collate_fn)


# TRAIN MODEL
 
### DEFINE MODEL

# function to retreive model
def _get_segmentation_model(num_classes):

    model = maskrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    hidden_layer = 256
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    
    return model
    
# function to calcualte Intersection Over Union score for bounding boxes
def get_IOU(pred_bbox, target_bbox, input_size):
    
    target = Image.new('1', input_size, 0)
    draw = ImageDraw.Draw(target)
    draw.rectangle(target_bbox, fill=255)

    pred = Image.new('1', input_size, 0)
    draw = ImageDraw.Draw(pred)
    draw.rectangle(pred_bbox, fill=255)

    intersection = (np.array(target) & np.array(pred)).sum()
    union = (np.array(target) | np.array(pred)).sum()

    iou = (intersection) / (union)

    return iou

# function to compute Root Mean Squared loss, Smooth L1 loss, and Intersection over Union loss for bounding boxes
def compute_loss(pred_bbox, target_bbox, input_size):

    MSE_loss = torch.nn.MSELoss()
    SmoothL1_loss = torch.nn.SmoothL1Loss()

    RMSE_loss = torch.sqrt(MSE_loss(pred_bbox.float(), target_bbox.float()))
    L1_loss = SmoothL1_loss(pred_bbox.float(), target_bbox.float())
    
    IoU = get_IOU(pred_bbox.tolist(), target_bbox.tolist(), tuple(input_size))
    IoU_loss = 1 - IoU

    return RMSE_loss, L1_loss, IoU_loss
    
# function to compute DICE loss for segmentaiton masks
def get_DICE_loss(target, pred):

    intersection = (target & pred).sum()
    target_area = target.sum()
    pred_area = pred.sum()

    dice = (2*intersection) / (target_area + pred_area)
    dice_loss = 1-dice
    
    return dice_loss

# retreive model, send to relevant device, and define model optimiser and learning rate scheduler parameters
model = _get_segmentation_model(num_classes)
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, 
                             lr=lr,
                             weight_decay=weight_decay)

lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                          factor = factor,
                                                          patience = patience)
    
### TRAIN MODEL

# set as inital validation error value very high
val_error = 1e10

training_loss = []
validation_loss = []

for epoch in range(num_epochs):

  # train one epoch using standard imported 'train_one_epoch' function
  train_loss = train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=20)
  train_loss_epoch = train_loss.__getattr__('loss').avg
  
  model.eval()# put model in evaluation mode (this mode does not output any form of error!)

  epoch_val_results = []
  epoch_val_error = 0
  epoch_val_iou_loss = 0
  epoch_val_dice_loss = 0

  TP = 0 # True positive count
  FP = 0 # False positive count
  FN = 0 # False negative count
  TP_first_pred = 0 # TP for first predictions (i.e. preds with top score)
  FP_first_pred = 0 # FP for first predictions
  
  # run through image batches - one img per batch - in validation dataset and compute loss to save model with lowest loss
  with torch.no_grad():
    for batch_i, (batch_images, batch_targets) in enumerate(data_loader_val):
    
      image = list(img.to(device) for img in batch_images)
      target = [{k: v.to(device) for k, v in t.items()} for t in batch_targets]
      
      batch_pred = model(image)
      num_preds = len(batch_pred[0]['scores'])
      
      batch_error = 0
      batch_iou_loss = 0
      batch_dice_loss = 0
      
      # if no predictions made compute max loss
      if num_preds == 0:
        # generate empty masks to calculate DICE loss
        pred_mask = np.ma.make_mask(np.zeros(input_size))
        pred_bbox = torch.tensor([0, 0, 0, 0]).to(device)
        
        RMSE_loss, L1_loss, iou_loss = compute_loss(pred_bbox,
                                                    target[0]['boxes'][0], 
                                                    input_size)
        
        dice_loss = get_DICE_loss(pred_mask, np.ma.make_mask(target[0]['masks'][0].cpu()))
        
        FN += 1
        
        batch_error += (RMSE_loss + L1_loss).item()
        batch_iou_loss += iou_loss
        batch_dice_loss += dice_loss

        epoch_val_results.append([epoch,
                                  batch_targets[0]['image_id'].item(), 
                                  batch_targets[0]['boxes'][0].tolist(),
                                  [0, 0, 0, 0],
                                  0,
                                  batch_error])

      # for instances of predictions
      else:
        for i in range(num_preds):
            # extract binary masks from predictions to calculate DICE loss
            pred_mask = np.array(batch_pred[0]['masks'][i].cpu())
            threshold = threshold_otsu(pred_mask)
            pred_mask_binary = pred_mask > threshold
            pred_bbox = torch.tensor(get_bounding_box(np.squeeze(pred_mask_binary))).to(device)
            
            RMSE_loss, L1_loss, iou_loss = compute_loss(pred_bbox.to(device),
                                                        target[0]['boxes'][0], 
                                                        input_size)
            
            dice_loss = get_DICE_loss(pred_mask_binary, np.ma.make_mask(target[0]['masks'][0].cpu()))
            
            if iou_loss <= iou_loss_threshold: # if IoU loss is less than threshold then prediction is a true positive (TP)
                TP += 1
                if i == 0: # if only using prediction with highest score
                  TP_first_pred += 1
            else: # if IoU loss is more than threshold then prediction is a false positive (FP)
                FP += 1
                if i == 0: # if only using prediction with hgihest score
                  FP_first_pred += 1
                
            batch_error += (RMSE_loss + L1_loss).item()
            batch_iou_loss += iou_loss
            batch_dice_loss += dice_loss
            
        batch_error = batch_error/num_preds
        batch_iou_loss = batch_iou_loss/num_preds
        batch_dice_loss = batch_dice_loss/num_preds
        
        epoch_val_results.append([epoch,
                                  batch_targets[0]['image_id'].item(), 
                                  batch_targets[0]['boxes'][0].tolist(),
                                  batch_pred[0]['boxes'].tolist(), 
                                  batch_pred[0]['scores'].tolist(),
                                  batch_error])

      epoch_val_error += batch_error
      epoch_val_iou_loss += batch_iou_loss
      epoch_val_dice_loss += batch_dice_loss

  # average validation error per validaiton image, and find total accuracy
  epoch_val_error = epoch_val_error/len(data_loader_val)
  epoch_val_iou_loss = epoch_val_iou_loss/len(data_loader_val)
  epoch_val_dice_loss = epoch_val_dice_loss/len(data_loader_val)
  batch_val_acc = TP / (TP + FP + FN)
  batch_val_acc_first_pred = TP_first_pred / (TP_first_pred + FP_first_pred + FN)

  # step learning rate scheduler
  lr_scheduler.step(epoch_val_iou_loss)

  # record training and validation loss over epoch
  training_loss.append(train_loss_epoch)
  validation_loss.append(epoch_val_iou_loss)
  print(f'Epoch {epoch} : calculated validation loss {epoch_val_error:.3}, iou loss {epoch_val_iou_loss:.3}, and dice loss {epoch_val_dice_loss}')

  # if epoch validation error is lower than current best validaiton error, save new model
  if epoch_val_iou_loss < val_error:
    val_error = epoch_val_iou_loss
    val_results = epoch_val_results
    torch.save(model.state_dict(), save_path + 'model.pt')

    print(f'\t Better model saved at epoch {epoch}')

# record training and validation results and losses; send to pickle files
val_results_pd = pd.DataFrame(val_results, columns = ['epoch', 'image id', 'target box', 'pred box', 'pred score', 'pred error'])
val_results_pd.to_pickle(save_path + 'val_results.pkl')

train_val_loss = pd.DataFrame(columns = ['training loss', 'validation loss'])
train_val_loss['training loss'] = training_loss
train_val_loss['validation loss'] = validation_loss
train_val_loss.to_pickle(save_path + 'train_val_loss.pkl')

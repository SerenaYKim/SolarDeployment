# -*- coding: utf-8 -*-

# Solar Panel and Roof Detection Using Fast RCNN

## 00. SetUp
"""

!pip install pycocotools --quiet
!git clone https://github.com/pytorch/vision.git
!git checkout v0.3.0

!cp vision/references/detection/utils.py ./
!cp vision/references/detection/transforms.py ./
!cp vision/references/detection/coco_eval.py ./
!cp vision/references/detection/engine.py ./
!cp vision/references/detection/coco_utils.py ./

"""---

** YOU WILL NEED TO CHANGE ENGINE.PY BEFORE EXECUTING THE SECOND CELL **

        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size = 1, gamma = 1
        )

---
"""

import os
import random
import numpy as np
import pandas as pd
# for ignoring warnings
import warnings
warnings.filterwarnings('ignore')

# We will be reading images using OpenCV
import cv2

# xml library for parsing xml files
from xml.etree import ElementTree as et

# matplotlib for visualization
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# torchvision libraries
import torch
import torchvision
from torchvision import transforms as torchtrans
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN

# these are the helper libraries imported.
from engine import train_one_epoch, evaluate
import utils
import transforms as T

from google.colab import drive
drive.mount('/content/drive')

# If there's a GPU available...
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

!python -V
!nvcc --version
from psutil import virtual_memory
ram_gb = virtual_memory().total / 1e9
print('Your runtime has {:.1f} gigabytes of available RAM\n'.format(ram_gb))

if ram_gb < 20:
  print('Not using a high-RAM runtime')
else:
  print('You are using a high-RAM runtime!')

"""## 01. TRAINING AND EVALUATION

### 01(1) Dataset
"""

#files_dir = "/content/drive/MyDrive/solar_city_files/combined for solar training"
files_dir = "/YourDirectory/"

class SolarImagesDataset(torch.utils.data.Dataset):

    def __init__(self, files_dir, width, height, clas=None, flag = 1, transforms = None):
        self.transforms = transforms
        self.files_dir = files_dir
        self.height = height
        self.width = width
        self.flag = flag
        self.clas = clas

        # classes: 0 index is reserved for background
        self.classes = [_, self.clas]

    def __getitem__(self, idx):

        img_name = self.imgs[idx]
        image_path = os.path.join(self.files_dir, img_name)

        # reading the images and converting them to correct size and color
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img_res = cv2.resize(img_rgb, (self.width, self.height), cv2.INTER_AREA)
        # diving by 255
        img_res /= 255.0

        # annotation file
        annot_filename = img_name[:-4] + '.xml'
        annot_file_path = os.path.join(self.files_dir, annot_filename)

        try:
          boxes = []
          labels = []
          tree = et.parse(annot_file_path)
          root = tree.getroot()

          # cv2 image gives size as height x width
          wt = img.shape[1]
          ht = img.shape[0]

          # box coordinates for xml files are extracted and corrected for image size given
          for member in root.findall('object'):
            if member.find('name').text == self.clas:
              labels.append(self.classes.index(member.find('name').text))

              # bounding box
              xmin = int(member.find('bndbox').find('xmin').text)
              xmax = int(member.find('bndbox').find('xmax').text)

              ymin = int(member.find('bndbox').find('ymin').text)
              ymax = int(member.find('bndbox').find('ymax').text)

              xmin_corr = (xmin/wt)*self.width
              xmax_corr = (xmax/wt)*self.width
              ymin_corr = (ymin/ht)*self.height
              ymax_corr = (ymax/ht)*self.height

              boxes.append([xmin_corr, ymin_corr, xmax_corr, ymax_corr])

          # convert boxes into a torch.Tensor
          boxes = torch.as_tensor(boxes, dtype=torch.float32)

          # getting the areas of the boxes
          area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

          # suppose all instances are not crowd
          iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

          labels = torch.as_tensor(labels, dtype=torch.int64)

          target = {}
          target["boxes"] = boxes
          target["labels"] = labels
          target["area"] = area
          target["iscrowd"] = iscrowd
          # image_id
          image_id = torch.tensor([idx])
          target["image_id"] = image_id

          if self.transforms:
            img_res = self.transforms(img_res)
          #   sample = self.transforms(image = img_res,
          #                            bboxes = target['boxes'],
          #                            labels = labels)

          #   img_res = sample['image']
          #   target['boxes'] = torch.Tensor(sample['bboxes'])

          if self.flag == 1:
            return img_res, target, img_name
          else:
            return img_res, target

        except:
          if self.transforms:
            img_res = self.transforms(img_res)

          if self.flag == 1:
            return img_res, img_name
          else:
            return img_res

    def __len__(self):
        return len(self.imgs)

# check dataset
dataset_s = SolarImagesDataset(files_dir, 480, 480, 'solar', 0)
dataset_r = SolarImagesDataset(files_dir, 480, 480, 'roof', 0)
print('length of dataset = ', len(dataset_r), '\n')
print('length of dataset = ', len(dataset_s), '\n')

# getting the image and target for a test index.  Feel free to change the index.
img, target = dataset_s[50]
print(img.shape, '\n',target, '\n')
img, target = dataset_r[50]
print(img.shape, '\n',target)

def plot_img_bbox(img, target=None):
    # plot the image and bboxes
    # Bounding boxes are defined as follows: x-min y-min width height
    fig, a = plt.subplots(1,1)
    fig.set_size_inches(5,5)
    a.imshow(img)
    colors = ['r', 'y']
    try:
      for box, label in zip(target['boxes'], target['labels']):
          x, y, width, height  = box[0], box[1], box[2]-box[0], box[3]-box[1]
          rect = patches.Rectangle((x, y),
                                 width, height,
                                 linewidth = 2,
                                 edgecolor = colors[label-1],
                                 facecolor = 'none')

          # Draw the bounding box on top of the image
          a.add_patch(rect)
      plt.show()
    except:
      plt.show()


# plotting the image with bboxes. Feel free to change the index
img, target = dataset_r[100]
plot_img_bbox(img, target)
img, target = dataset_s[100]
plot_img_bbox(img, target)

def get_object_detection_model(num_classes):

    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # model = fasterrcnn_resnet152_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

dataset_r = SolarImagesDataset(files_dir, 480, 480, 'roof', 0, transforms=torchtrans.ToTensor())
dataset_s = SolarImagesDataset(files_dir, 480, 480, 'solar', 0, transforms=torchtrans.ToTensor())

# split the dataset in train and test set
torch.manual_seed(1)
indices = torch.randperm(len(dataset_r)).tolist()

# train test split
test_split = 0.2
tsize = int(len(dataset_r)*test_split)
dataset_train_r = torch.utils.data.Subset(dataset_r, indices[:-tsize])
dataset_test_r = torch.utils.data.Subset(dataset_r, indices[-tsize:])
dataset_train_s = torch.utils.data.Subset(dataset_s, indices[:-tsize])
dataset_test_s = torch.utils.data.Subset(dataset_s, indices[-tsize:])
# dataset_test = torch.utils.data.Subset(dataset_test, indices[-tsize:])

# define training and validation data loaders
data_loader_s = torch.utils.data.DataLoader(
    dataset_train_s, batch_size=8, shuffle=True, num_workers=4,
    collate_fn=utils.collate_fn)

data_loader_s_test = torch.utils.data.DataLoader(
    dataset_test_s, batch_size=8, shuffle=True, num_workers=4,
    collate_fn=utils.collate_fn)

data_loader_r = torch.utils.data.DataLoader(
    dataset_train_r, batch_size=8, shuffle=True, num_workers=4,
    collate_fn=utils.collate_fn)

data_loader_r_test = torch.utils.data.DataLoader(
    dataset_test_r, batch_size=8, shuffle=True, num_workers=4,
    collate_fn=utils.collate_fn)

"""### 01(2) Roof Model (model_r) Training & Evaluation"""

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_classes = 2
# get the model using our helper function
try:
  print("Loading pre-finetuned model...")
  model = get_object_detection_model(num_classes)
  model.load_state_dict(torch.load("/content/drive/MyDrive/solar_img_training/solar_fine-tuned.pt"))
except:
  print("No pre-finetuned model found... loading new pre-trained from torchvision...")
  model_r = get_object_detection_model(num_classes)

# move model to the right device
model_r.to(device)

# construct an optimizer
params = [p for p in model_r.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(params, lr=0.0002, weight_decay=0.001)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=1,
                                               gamma=0.7)

# training for 8 epochs
num_epochs = 4

for epoch in range(num_epochs):
    # training for one epoch
    train_one_epoch(model_r, optimizer, data_loader_r, device, epoch, print_freq=8)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model_r, data_loader_r_test, device=device)

"""### 01(3) Solar Model (model_s) Training & Evaluation"""

try:
  print(to_skip_try_clause)
  print("Loading pre-finetuned model...")
  model = get_object_detection_model(num_classes)
  model.load_state_dict(torch.load("/content/drive/MyDrive/solar_img_training/solar_fine-tuned.pt"))
except:
  print("No pre-finetuned model found... loading new pre-trained from torchvision...")
  model_s = get_object_detection_model(num_classes)

# move model to the right device
model_s.to(device)

# construct an optimizer
params = [p for p in model_s.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(params, lr=0.0002, weight_decay=0.000001)
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                               gamma=0.7)

# training for 8 epochs
num_epochs = 4

for epoch in range(num_epochs):
    # training for one epoch
    train_one_epoch(model_s, optimizer, data_loader_s, device, epoch, print_freq=8)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model_s, data_loader_s_test, device=device)

"""### 01(4) Save Both Solar and Roof Models and Apply NMS"""

torch.save(model_r.state_dict(), "/content/drive/MyDrive/solar_img_training/solar_fine-tuned_roof.pt")
torch.save(model_s.state_dict(), "/content/drive/MyDrive/solar_img_training/solar_fine-tuned_solar.pt")

def apply_nms(orig_prediction, iou_thresh=0.3):

    # torchvision returns the indices of the bboxes to keep
    keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)

    final_prediction = orig_prediction
    final_prediction['boxes'] = final_prediction['boxes'][keep]
    final_prediction['scores'] = final_prediction['scores'][keep]
    final_prediction['labels'] = final_prediction['labels'][keep]

    return final_prediction, final_prediction['boxes']

# function to convert a torchtensor back to PIL image
def torch_to_pil(img):
    return torchtrans.ToPILImage()(img).convert('RGB')

"""### 01(5) Check

#### Roof Check
"""

model_r.eval()
for img, target in dataset_test_r:
  with torch.no_grad():
      prediction = model_r([img.to(device)])[0]
  print('predicted #boxes: ', len(prediction['labels']))
  print('real #boxes: ', len(target['labels']))
  print('EXPECTED OUTPUT')
  plot_img_bbox(torch_to_pil(img), target)
  nms_prediction, nms_boxes = apply_nms(prediction, iou_thresh=0.2)
  print('NMS APPLIED MODEL OUTPUT')
  plot_img_bbox(torch_to_pil(img), nms_prediction)
  area_by_total_area = sum((nms_boxes[:, 3] - nms_boxes[:, 1]) * (nms_boxes[:, 2] - nms_boxes[:, 0]))/(480*480)
  try:
    print("Ratio of area covered by roof to the total image = ", (area_by_total_area).item())
  except:
    print("Ratio of area covered by roof to the total image = 0.000001")

"""#### Solar Check"""

model_s.eval()
for img, target in dataset_test_s:
  with torch.no_grad():
      prediction = model_s([img.to(device)])[0]
  print('predicted #boxes: ', len(prediction['labels']))
  print('real #boxes: ', len(target['labels']))
  print('EXPECTED OUTPUT')
  plot_img_bbox(torch_to_pil(img), target)
  nms_prediction, nms_boxes = apply_nms(prediction, iou_thresh=0.1)
  print('NMS APPLIED MODEL OUTPUT')
  plot_img_bbox(torch_to_pil(img), nms_prediction)
  area_solar_to_total_area = sum((nms_boxes[:, 3] - nms_boxes[:, 1]) * (nms_boxes[:, 2] - nms_boxes[:, 0]))/(480*480)
  try:
    print("Ratio of area covered by PV to the total image = ", area_solar_to_total_area.item())
  except:
    print("Ratio of area covered by PV to the total image = 0.000001")

"""### PREDICTION FOR BOTH AND ADDING TO CSV FILE

## 02. PREDICTION
"""

path = "/content/drive/MyDrive/Satellite_2K/satellite_2k_b18/"
model_s.eval()
model_r.eval()
for file in sorted(os.listdir(path)):
    if os.path.isdir(path+file):
        df = pd.DataFrame(columns=['img name'])
        print(path+file, len(os.listdir(path+file)))
        dataset = SolarImagesDataset(path+file, 480, 480)
        for img, img_n in dataset:

      # Solar
            img_tensor = torchtrans.ToTensor()(img)
            with torch.no_grad():
                prediction = model_s([img_tensor.to(device)])[0]
            nms_prediction, nms_boxes = apply_nms(prediction, iou_thresh=0.1)
            area_by_total_area = sum((nms_boxes[:, 3] - nms_boxes[:, 1]) * (nms_boxes[:, 2] - nms_boxes[:, 0]))/(480*480)
            try:
                final_area_solar = area_by_total_area.item()
            except:
                final_area_solar = 0.000000001

      # Roof
            with torch.no_grad():
                prediction = model_r([img_tensor.to(device)])[0]
            nms_prediction, nms_boxes = apply_nms(prediction, iou_thresh=0.3)
            area_by_total_area = sum((nms_boxes[:, 3] - nms_boxes[:, 1]) * (nms_boxes[:, 2] - nms_boxes[:, 0]))/(480*480)
            try:
                final_area_roof = area_by_total_area.item()
            except:
                final_area_roof = 0.000000001
            df = df.append({'img name': img_n, 'solar': final_area_solar, 'roof': final_area_roof}, ignore_index=True)

        df.to_csv("/content/drive/MyDrive/Satellite_2K/csv_v2/"+path[-17:-1]+'_'+file+'.csv')
    else:
      print("done")

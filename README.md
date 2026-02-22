Link to project: https://github.com/SEP-group-project/Main-repository
# Main-repository

# General Environment setup:
## Step 1: Create Conda Environment 
- conda create -n fer_cls python=3.10 -y
- conda activate fer_cls	
## Step 2: Install packages for model and data: 
- conda install pytorch torchvision torchaudio -c pytorch
- conda install numpy pillow
## Step 3 (extra): Install packages for confusion matrix and plots:
- conda install matplotlib scikit-learn	
## Step 4: Install additional packages for video demo and Grad-CAM
- This step requires user to pip install pytorch-grad-cam inside the conda environment since it isn’t consistently available on conda. 
- conda install opencv -c conda-forge 
- pip install pytorch-grad-cam



# Data Setup
## Using our dataset:
Download the data set from out GitHub by downloading the “data” folder: https://github.com/SEP-group-project/Main-repository .
The data has to be in the same folder as the other scripts.
  
Should the download over github not be an option, download the dataset here: https://www.kaggle.com/datasets/shuvoalok/raf-db-dataset. 
In this case you have to delete both folders 7 since we don't use neutral images and rename the 'test' and 'train' folders to 'test_images" and 'train_images'. In the end, the data has to match the subfolder structure below, including having 'data' as the folder name where 'test_images" and 'train_images' are saved.

## Using your own Dataset:
The data set has to follow the following folder structure. The folders have to be labeled as numbers according to the emotions:
1:”surprise”, 2:”fear”, 3:”disgust”, 4:”happiness”, 5:”sad-
ness”, 6:”anger”.

data/
├── test_images/
│   ├── 1/
│   ├── 2/
│   ├── 3/
│   ├── 4/
│   ├── 5/
│   └── 6/
└── train_images/
    ├── 1/
    ├── 2/
    ├── 3/
    ├── 4/
    ├── 5/
    └── 6/
    
# How to create a csv file using our model:

First make sure you have a “best_mode_cosine.pt” file in the folder you have saved “classifier_making_csv” in. This should be the case by default. Then run the script “classifier_making_csv”. Your terminal will ask for an input path to the folder you have all your images in. This folder must be a folder of images with no subfolders. Input the input path. If you accidentally put in a wrong path you will have to rerun the script. The csv file will be saved as “classification_results.csv” and override older versions.

## Environment Setup

Install the required Python packages:


import random
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
import os
import csv
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report,matthews_corrcoef
from torchvision import datasets, transforms

# Classification model
## Prerequirements

RAF-DB dataset installed and in the same folder 

## Environment setup using conda:

Step 1: Create Conda Environment 
- conda create -n fer_cls python=3.10 -y	

Step 2: Install packages for model itself: 
- conda activate fer_cls
- conda install pytorch torchvision torchaudio -c pytorch
- conda install numpy pillow

Step 3 (extra): Install packages for confusion matrix and plots:
- conda install matplotlib scikit-learn

## Output

The classification model outputs a trained model best_model_cosine.pt


# Video Demo: Facial Emotion Recognition with Grad-CAM

This program processes a video file to detect faces, predict emotions using our CNN model, and overlay **Grad-CAM heatmaps** along with predicted emotion labels on each detected face. The processed video is saved as output.

---

## Environment Setup

Install the required Python packages:


-conda create -n demo python=3.11

-conda activate demo

-conda install pytorch torchvision -c pytorch

-conda install opencv numpy -c conda-forge

-pip install pytorch-grad-cam (pytorch-grad-cam is not available on conda)


---

## Usage

1. open video_demo folder and place input video in the project directory

2. Run the Program:
python video_demo.py

3. Enter the input and output paths
4. The processed video with overlaid Grad-CAM and predicted emotion labels will be saved at the output path.

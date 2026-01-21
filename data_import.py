import numpy as np
from torchvision import datasets, transforms
import csv

transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
train_datapath = "data/train_images"
test_datapath = "data/test_images"


# Load Images
train_images = datasets.ImageFolder(root=train_datapath, transform=transform,)
test_images = datasets.ImageFolder(root=test_datapath, transform=transform)


# Load Labels
with open('data/test_labels.csv', mode='r', newline='') as file:
    reader = csv.DictReader(file) 
    test_labels = []
    for row in reader:
        filename = row['image']
        label = row['label'] 
        test_labels.append([label])

with open('data/train_labels.csv', mode='r', newline='') as file:
    reader = csv.DictReader(file) 
    train_labels = []
    for row in reader:
        filename = row['image']
        label = row['label'] 
        train_labels.append([label])







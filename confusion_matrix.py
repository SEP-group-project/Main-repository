import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
import random
from data_import import train_images, test_images
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

class RandomHistEqualize:
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, img):
        if random.random() < self.p:
            img = TF.equalize(img)
        return img

test_tfms = T.Compose([
    T.Resize((64, 64)),
    RandomHistEqualize(p=1.0),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

test_images.transform = test_tfms

class EmotionCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        def block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(0.15),
            )

        self.features = nn.Sequential(
            block(3, 32),
            block(32, 64),
            block(64, 128),
            block(128, 256),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


test_loader = DataLoader(
    test_images,
    batch_size=256,
    shuffle=False
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = len(train_images.classes)
model = EmotionCNN(num_classes).to(device)

model.load_state_dict(torch.load("best_model_cosine.pt", map_location=device))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)

        outputs = model(x)                  
        preds = torch.argmax(outputs, 1)    

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.numpy())

cm = confusion_matrix(all_labels, all_preds)
print(cm)
classes = train_images.classes


cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

plt.figure(figsize=(8, 8))
plt.imshow(cm_norm)
plt.colorbar()

plt.xticks(np.arange(len(classes)), classes, rotation=45, ha="right")
plt.yticks(np.arange(len(classes)), classes)

plt.xlabel("predicted")
plt.ylabel("true")
plt.title("normalized confusion matrix")

plt.tight_layout()

plt.savefig("confusion_matrix.png", dpi=300)
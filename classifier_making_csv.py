import random
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
import os
import csv

model_path = "best_model_cosine.pt"
output_csv = "classification_results.csv"
classes = ['surprise', 'fear', 'disgust', 'happiness', 'sadness', 'anger']

# CSV colum order as in the example in the slides
csv_order = ['happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']

class RandomGamma:
    def __init__(self, gamma_range=(0.7, 1.6), p=0.7):
        self.gamma_range = gamma_range
        self.p = p
    def __call__(self, img):
        if random.random() < self.p:
            img = TF.adjust_gamma(img, random.uniform(*self.gamma_range))
        return img


class RandomHistEqualize:
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, img):
        if random.random() < self.p:
            img = TF.equalize(img)
        return img


#EmotionCNN and test_tfms copied from classification model so it can run without having to run the whle training process due to the import.

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




test_tfms = T.Compose([
    T.Resize((64, 64)),
    RandomHistEqualize(p=1.0),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

def classify_image(model, image_path, device):
    img = Image.open(image_path).convert("RGB")
    x = test_tfms(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy().flatten()
    return probs


def classify_folder_images(folder_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Filepath read")

    model = EmotionCNN(num_classes=len(classes)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    image_files = [os.path.join(folder_path, f)
                   for f in os.listdir(folder_path)
                   if f.lower().endswith(exts)]

    results = []

    for img_path in image_files:
        probs = classify_image(model, img_path, device)
        if probs is not None:
            row = {"filepath": img_path}
            row.update({cls: round(float(p), 4)
                        for cls, p in zip(classes, probs)})
            results.append(row)

    # Write to CSV in order from the project requirements
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath"] + csv_order)
        writer.writeheader()
        for row in results:
            reordered = {"filepath": row["filepath"]}
            reordered.update({emo: row.get(emo, None) for emo in csv_order})
            writer.writerow(reordered)

    print(f"Saved classification results to: {output_csv}")


classify_folder_images(input("please add your input path: "))


import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import csv
from classification_model import EmotionCNN, test_tfms

model_path = "best_model_cosine.pt"
output_csv = "classification_results.csv"
classes = ['surprise', 'fear', 'disgust', 'happiness', 'sadness', 'anger']

# CSV colum order as in the example in the slides
csv_order = ['happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']


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

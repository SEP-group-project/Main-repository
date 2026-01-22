import csv
from PIL import Image
from torchvision import transforms, datasets
from pathlib import Path

original_images_path = "archive(3)/train_images"        # add original image folder path
resized_images_path  = "data_resized/train_images"  # add resized image folder path
input_csv     = "archive(3)/labels.csv" # add original CSV path
filtered_csv  = "formatted_output.csv"

label_map = {
    "surprise": 1,
    "fear": 2,
    "disgust": 3,
    "happiness": 4,
    "sadness": 5,
    "anger": 6
}
exclude_labels = {"neutral", "content"}



with open(input_csv, "r", newline="", encoding="utf-8") as infile, \
     open(filtered_csv, "w", newline="", encoding="utf-8") as outfile:

    reader = csv.DictReader(infile)
    writer = csv.writer(outfile)
    writer.writerow(["image", "label"])

    for row in reader:
        path = row["pth"]
        folder = path.split("/")[0].strip().lower() if "/" in path else ""

        if folder in exclude_labels:
            continue

        label_num = label_map.get(folder, "")
        if label_num == "":
            continue

        writer.writerow([path, label_num])



def resize_images(input_root, output_root, size=(64, 64)):
    resize = transforms.Resize(size)
    input_root, output_root = Path(input_root), Path(output_root)

    for folder in input_root.iterdir():
        if folder.is_dir():
            out_folder = output_root / folder.name
            out_folder.mkdir(parents=True, exist_ok=True)
            for idx, img_path in enumerate(folder.iterdir()):
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    img = Image.open(img_path)
                    img = resize(img)
                    img.save(out_folder / f"{idx}.jpg")


resize_images(original_images_path, resized_images_path)



transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
train_datapath = "archive(3)/train_images"
test_datapath  = "archive(3)/test_images"

train_images = datasets.ImageFolder(root=train_datapath, transform=transform)
test_images  = datasets.ImageFolder(root=test_datapath,  transform=transform)

print("finished data preparation")
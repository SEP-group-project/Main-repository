from PIL import Image
from torchvision import transforms, datasets
from pathlib import Path

original_images_path = "archive(3)/train_images"        # add original image folder path
resized_images_path  = "data_resized/train_images"  # add resized image folder path


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
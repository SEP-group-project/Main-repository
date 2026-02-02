from torchvision import datasets, transforms

transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
train_transform = transforms.Compose([ transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])

train_datapath = "data/train_images"
test_datapath = "data/test_images"


# Load Images
train_images = datasets.ImageFolder(root=train_datapath, transform=transform,)
test_images = datasets.ImageFolder(root=test_datapath, transform=transform)

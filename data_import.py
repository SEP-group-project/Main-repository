from torchvision import datasets, transforms
from torch.utils.data import DataLoader

traindata_path = r"C:\Users\finnr\OneDrive\Dokumente\SEP\datasets\DATASET\train"
testdata_path = r"C:\Users\finnr\OneDrive\Dokumente\SEP\datasets\DATASET\test"

transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

# Load dataset
train_dataset = datasets.ImageFolder(root=traindata_path, transform=transform)
test_dataset = datasets.ImageFolder(root=testdata_path, transform=transform)


# DataLoader for batching
trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=32, shuffle=False)



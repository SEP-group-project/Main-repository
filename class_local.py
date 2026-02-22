


"""



#####################

these 2 are one of our attempts to improve our model. we compute the mean and std of the training set and use them to normalize the data
it isnt part of our final submission, but it was a step towards improving our model and we wanted to keep it for reference

#######################




import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.optim as optim
from data_import import train_images, test_images
from torchvision import transforms





def compute_mean_std(loader, device=None):

    mean = torch.zeros(3)
    std = torch.zeros(3)
    n = 0

    for x, _ in loader:
        b, c, h, w = x.shape
        x = x.view(b, c, -1)  

        mean += x.mean(dim=2).sum(dim=0)
        std  += x.std(dim=2).sum(dim=0)
        n += b

    mean /= n
    std /= n
    return mean, std


stats_tfms = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(), 
])


train_images.transform = stats_tfms

stats_loader = DataLoader(train_images, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

mean, std = compute_mean_std(stats_loader)
print("computed train mean:", mean.tolist())
print("computed train std: ", std.tolist())

train_tfms = transforms.Compose([
    transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean.tolist(), std.tolist()),
])

test_tfms = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize(mean.tolist(), std.tolist()),
])

train_images.transform = train_tfms
test_images.transform  = test_tfms


num_classes = len(train_images.classes)
idx_to_emotion = {
    0: "surprise",    
    1: "fear",        
    2: "disgust",     
    3: "happiness",   
    4: "sadness",     
    5: "anger",       
} 

train_loader = DataLoader(train_images, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_images,  batch_size=64, shuffle=False)


class EmotionCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  

            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), 
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
    


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", device)
model = EmotionCNN(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


def train_epoch(model, loader):
    model.train()
    correct, total, loss_sum = 0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader):
    model.eval()
    correct, total, loss_sum = 0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)

        loss_sum += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum / total, correct / total

if device.type == "cuda":
    num_epochs = 100
else :
    num_epochs = 5

best_acc = 0.0
for epoch in range(num_epochs):

    train_loss, train_acc = train_epoch(model, train_loader)
    test_loss, test_acc   = eval_epoch(model, test_loader)

    print(f"epoch nr {epoch:02d} | "
          f"train acc {train_acc:.4f} | "
          f"test acc {test_acc:.4f}")

    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), "best_model.pt")


print("best test accuracy so far:", best_acc)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms


from data_import import train_images, test_images

from collections import Counter

def get_class_weights(dataset, num_classes, device):
    counts = torch.zeros(num_classes, dtype=torch.long)
    for i in range(len(dataset)):
        _, y = dataset[i]
        counts[y] += 1

    weights = 1.0 / counts.float().clamp(min=1)

    weights = weights / weights.sum() * num_classes
    return weights.to(device), counts


def compute_mean_std(loader):
    mean = torch.zeros(3)
    std = torch.zeros(3)
    n = 0

    for x, _ in loader:
        b, c, h, w = x.shape
        x = x.view(b, c, -1)  
        mean += x.mean(dim=2).sum(dim=0)
        std  += x.std(dim=2).sum(dim=0)
        n += b

    mean /= n
    std /= n
    return mean, std


class EmotionCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    correct, total, loss_sum = 0, 0, 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)

    return loss_sum / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)

        loss_sum += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)

    return loss_sum / total, correct / total


def main():
 
    num_workers = 0  
    pin_memory = False  

    stats_tfms = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
    ])
    train_images.transform = stats_tfms

    stats_loader = DataLoader(
        train_images, batch_size=64, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    mean, std = compute_mean_std(stats_loader)
    print("computed train mean:", mean.tolist())
    print("computed train std: ", std.tolist())

    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean.tolist(), std.tolist()),
    ])

    test_tfms = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(mean.tolist(), std.tolist()),
    ])

    train_images.transform = train_tfms
    test_images.transform = test_tfms

    train_loader = DataLoader(
        train_images, batch_size=64, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_images, batch_size=64, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    num_classes = len(train_images.classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device:", device)

    model = EmotionCNN(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 100 if device.type == "cuda" else 20

    best_acc = 0.0
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = eval_epoch(model, test_loader, criterion, device)

        print(f"epoch {epoch:02d} | train acc {train_acc:.4f} | test acc {test_acc:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "best_model1.pt")

    print("best test accuracy so far:", best_acc)


if __name__ == "__main__":
    main()


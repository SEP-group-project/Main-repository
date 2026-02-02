import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
import random
from data_import import train_images, test_images


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


train_tfms = T.Compose([
    T.Resize((64, 64)),
    RandomGamma((0.7, 1.6), p=0.7),
    RandomHistEqualize(p=0.5),
    T.RandomHorizontalFlip(),
    T.RandomAffine(10, translate=(0.06, 0.06), scale=(0.9, 1.1)),
    T.ColorJitter(0.2, 0.2, 0.1, 0.02),
    T.ToTensor(),
    T.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3)),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

test_tfms = T.Compose([
    T.Resize((64, 64)),
    RandomHistEqualize(p=1.0),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train_images.transform = train_tfms
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


def mixup(x, y, alpha=0.4):
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam

def mixup_loss(logits, y1, y2, lam, criterion):
    return lam * criterion(logits, y1) + (1 - lam) * criterion(logits, y2)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader = DataLoader(
    train_images,
    batch_size=128,
    shuffle=True,
    num_workers=4 if device.type == "cuda" else 0,
    pin_memory=(device.type == "cuda"),)

test_loader = DataLoader(
    test_images,
    batch_size=256,
    shuffle=False,
    num_workers=4 if device.type == "cuda" else 0,
    pin_memory=(device.type == "cuda"),)

num_classes = len(train_images.classes)
model = EmotionCNN(num_classes).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)

if device.type == "cuda":
    num_epochs = 60
else:
    num_epochs = 20

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)


def train_epoch():
    model.train()
    total, correct, loss_sum = 0, 0, 0.0

    for x, y in train_loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        x, y1, y2, lam = mixup(x, y, alpha=0.4)

        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = mixup_loss(out, y1, y2, lam, criterion)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        loss_sum += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)

    return loss_sum / total, correct / total


@torch.no_grad()
def eval_epoch():
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0

    for x, y in test_loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x)
        loss = criterion(out, y)

        loss_sum += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)

    return loss_sum / total, correct / total


best_acc = 0.0
for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch()
    test_loss, test_acc = eval_epoch()
    scheduler.step()

    print(f"epoch {epoch:03d} | train acc {train_acc:.4f} | test acc {test_acc:.4f} | train loss {train_loss:.4f} | test loss {test_loss:.4f}")

    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), "best_model_cosine.pt")

print("best test accuracy:", best_acc)

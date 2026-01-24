import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from data_import import train_images, test_images


num_classes = len(train_images.classes)
idx_to_emotion = {
    0: "surprise",    
    1: "fear",        
    2: "disgust",     
    3: "happiness",   
    4: "sadness",     
    5: "anger",       
} 

batch_size = 64
val_ratio = 0.1

num_total = len(train_images)
num_val   = int(val_ratio * num_total)
num_train = num_total - num_val

train_dataset, val_dataset = random_split(train_images, [num_train, num_val])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset,  batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_images,  batch_size=batch_size, shuffle=False)

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


best_val_acc = 0.0
for epoch in range(1, 20):

    train_loss, train_acc = train_epoch(model, train_loader)
    val_loss, val_acc     = eval_epoch(model, val_loader)
    test_loss, test_acc   = eval_epoch(model, test_loader)

    print(f"epoch nr {epoch:02d} | "
          f"train acc {train_acc:.4f} | "
          f"val acc {val_acc:.4f} | "
          f"test acc {test_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pt")


print("best validation accuracy so far:", best_val_acc)

model.load_state_dict(torch.load("best_model.pt", map_location=device))
test_loss, test_acc = eval_epoch(model, test_loader)
print(f"final test accuracy: {test_acc:.4f}")
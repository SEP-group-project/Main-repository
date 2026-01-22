import torch
import numpy as np
import cv2
import time
from torchvision import models
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.optim as optim
from data_import import train_images, test_images


#cnn model
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



#grancam
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])



def get_last_conv_layer(model):
    
    return model.features[6]


def gradcam(model, face_bgr, class_idx):
    
    model.eval()

    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_rgb = cv2.resize(face_rgb, (64, 64))
    x = preprocess(face_rgb).unsqueeze(0).to(device)
    
    x.requires_grad_(True)
    

    conv_layer = get_last_conv_layer(model)
    

    activations = None
    gradients = None
    
    def forward_hook(module, input, output): 
        nonlocal activations
        activations = output     #output: feature maps of the last conv-layer
    
    def backward_hook(module, grad_in, grad_out): 
        nonlocal gradients
        gradients = grad_out[0]  #gradient of feature maps

    
    hook_f = conv_layer.register_forward_hook(forward_hook)  
    hook_b = conv_layer.register_full_backward_hook(backward_hook)

    
    preds = model(x)
    score = preds[:, class_idx]  # score of the predicted class
    model.zero_grad(set_to_none=True)
    score.backward()

    hook_b.remove()
    hook_f.remove()

    gradients = gradients.detach().cpu().numpy()[0]
    activations = activations.detach().cpu().numpy()[0]

    weights = np.mean(gradients, axis=(1,2)) # importance of each feature map 
                                             # gradient.shape: (C, H, W)
                                             # weights stores one weight per feature map

    
    heatmap = np.zeros(activations.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        heatmap += w*activations[i]   # one weighted feature map
    
    heatmap = np.maximum(heatmap, 0) # apply ReLU
    heatmap /= heatmap.max()+ 1e-8   # normalize


    return heatmap

def overlay_heatmap(img, heatmap, alpha=0.4):
    
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(img, alpha, heatmap, 1-alpha, 0)
    return superimposed_img




#demo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionCNN(num_classes=6).to(device)
WEIGHTS_PATH = "best_model.pt"
state = torch.load(WEIGHTS_PATH, map_location=device)
model.load_state_dict(state)
model.eval()



preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])



#@torch.no_grad()
def predict_emotion(face_bgr):
    with torch.no_grad():
     face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
     face_rgb = cv2.resize(face_rgb, (64, 64), interpolation=cv2.INTER_AREA)

     x = preprocess(face_rgb).unsqueeze(0).to(device)  
     logits = model(x)
     probs = torch.softmax(logits, dim=1)
     conf, pred = torch.max(probs, dim=1)

     pred_idx = int(pred.item())
     conf = float(conf.item())
     emotion = idx_to_emotion.get(pred_idx, str(pred_idx))
     return emotion, conf, pred_idx


cap = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
if face_detector.empty():
    raise RuntimeError("couldn't load haarcascade_frontalface_default.xml ") #change:i put the face detector outside the loop cuz its more efficient this way
if not cap.isOpened():
    raise RuntimeError("couldn't open webcam.")


SHOW_GRADCAM = False
FROZEN_FRAME = None


while True:
    if not SHOW_GRADCAM:
      ret, frame = cap.read()
      if not ret:
        break
    
      frame = cv2.resize(frame, (640, 480))
    
    else:
       frame = FROZEN_FRAME.copy()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        pad = int(0.15 * w)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame.shape[1], x + w + pad)
        y2 = min(frame.shape[0], y + h + pad)

        face_roi = frame[y1:y2, x1:x2]
        emotion, conf, pred_idx = predict_emotion(face_roi)
        
        if SHOW_GRADCAM:
           heatmap = gradcam(model,face_roi,pred_idx)
           superimposed_img = overlay_heatmap(face_roi,heatmap)

           frame[y1:y2, x1:x2] = superimposed_img


        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{emotion}: {conf:.2f}"
        cv2.putText(frame, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    

   
        
    
    cv2.imshow('emotion detector', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('g'):
       SHOW_GRADCAM = not SHOW_GRADCAM
       if SHOW_GRADCAM:
          FROZEN_FRAME = frame.copy()
    
   

    
    
    if key == ord('q'):
        break  

cap.release()
cv2.destroyAllWindows()
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

idx_to_emotion = {
    0: "surprise",    
    1: "fear",        
    2: "disgust",    
    3: "happiness",   
    4: "sadness",     
    5: "anger",       
}


class EmotionCNN(nn.Module):
    def __init__(self, num_classes=6):
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
            nn.Dropout(0.0), 
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


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

@torch.no_grad()
def predict_emotion(face_bgr):
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
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        pad = int(0.15 * w)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame.shape[1], x + w + pad)
        y2 = min(frame.shape[0], y + h + pad)

        face_roi = frame[y1:y2, x1:x2]
        emotion, conf, _ = predict_emotion(face_roi)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{emotion}: {conf:.2f}"
        cv2.putText(frame, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('emotion detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  

cap.release()
cv2.destroyAllWindows()
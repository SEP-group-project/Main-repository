import torch
import numpy as np
import cv2
import time
from torchvision import models , transforms
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
import torch.optim as optim

from data_import import train_images, test_images
from xAI.occlusion import occlusion_saliency
from xAI.gradcam import gradcam, overlay_heatmap
from xAI.LayerActivation import get_conv_layer, get_layer_activation, layer_activation_heatmap_from_tensor
from xAI.smoothGrad import coumpute_smoothGrad
from classification_model import EmotionCNN

idx_to_emotion = {
    0: "surprise",    
    1: "fear",        
    2: "disgust",     
    3: "happiness",   
    4: "sadness",     
    5: "anger",       
} 





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def layer_activation_heatmap(model, face_bgr, which_layer="last"):
    model.eval()

    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_rgb = cv2.resize(face_rgb, (64,64), interpolation=cv2.INTER_AREA)
    x = preprocess(face_rgb).unsqueeze(0).to(device)

    layer = get_conv_layer(model, which_layer)
    activation = get_layer_activation(model, layer, x)

    heat = layer_activation_heatmap_from_tensor(activation).numpy().astype("float32")
    return heat


model = EmotionCNN(num_classes=6).to(device)
WEIGHTS_PATH = "best_model_cosine.pt"
state = torch.load(WEIGHTS_PATH, map_location=device)
model.load_state_dict(state)
model.eval()




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

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

if face_detector.empty():
    raise RuntimeError("couldn't load haarcascade_frontalface_default.xml ") #change:i put the face detector outside the loop cuz its more efficient this way
if not cap.isOpened():
    raise RuntimeError("couldn't open webcam.")


MODE = "none"  # "none", "gradcam", "vanilla", "occlusion", "smoothgrad", "activation"
FROZEN_FRAME = None


while True:
    if MODE == "none":
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

        if MODE == "gradcam":
           heatmap = gradcam(model,face_roi,pred_idx)
           superimposed_img = overlay_heatmap(face_roi,heatmap)

        #elif MODE == "vanilla":
           #heatmap = vanilla_grad_saliency(model, face_roi, pred_idx)
           #superimposed_img = overlay_heatmap(face_roi, heatmap)

        elif MODE == "occlusion":
           heatmap = occlusion_saliency(model, face_roi, pred_idx)
           superimposed_img = overlay_heatmap(face_roi, heatmap)
        
        elif MODE =="smoothgrad":
            heatmap = coumpute_smoothGrad(model, face_roi, pred_idx, 20)
            superimposed_img = overlay_heatmap(face_roi, heatmap)

        elif MODE == "activation":
            heatmap = layer_activation_heatmap(model, face_roi, which_layer="last")
            superimposed_img = overlay_heatmap(face_roi, heatmap)

        if MODE in ["gradcam", "vanilla", "occlusion", "smoothgrad", "activation"]:
            frame[y1:y2, x1:x2] = superimposed_img


        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{emotion}: {conf:.2f} | {MODE}" 
        cv2.putText(frame, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    

    cv2.imshow('emotion detector', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('g'):
       MODE = "none" if MODE == "gradcam" else "gradcam"
       FROZEN_FRAME = frame.copy() if MODE != "none" else None

    #if key == ord('v'):
       #MODE = "none" if MODE == "vanilla" else "vanilla"
       #FROZEN_FRAME = frame.copy() if MODE != "none" else None

    if key == ord('o'):
       MODE = "none" if MODE == "occlusion" else "occlusion"
       FROZEN_FRAME = frame.copy() if MODE != "none" else None

    if key == ord('s'):
        MODE = "none" if MODE == "smoothgrad" else "smoothgrad"
        FROZEN_FRAME = frame.copy() if MODE != "none" else None

    if key == ord('a'):
        MODE = "none" if MODE == "activation" else "activation"
        FROZEN_FRAME = frame.copy() if MODE != "none" else None

    if key == ord('n'):
       MODE = "none"
       FROZEN_FRAME = None
    
    if key == ord('q'):
        break  

cap.release()
cv2.destroyAllWindows()

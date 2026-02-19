import cv2
import torch
from torchvision import transforms
from model import EmotionCNN
from xAI.gradcam2 import gradcam_heatmap, overlay 

idx_to_emotion = {
    0: "surprise",    
    1: "fear",        
    2: "disgust",    
    3: "happiness",   
    4: "sadness",     
    5: "anger",       
}



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionCNN(num_classes=6).to(device)
WEIGHTS_PATH = "best_model_cosine.pt"
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
    return x,emotion, conf, pred_idx



def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {input_path}")
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    if face_detector.empty():
        raise RuntimeError("couldn't load haarcascade_frontalface_default.xml ")

    while True:
        ret,frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (width, height))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            pad = int(0.15 * w)
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(frame.shape[1], x + w + pad)
            y2 = min(frame.shape[0], y + h + pad)

            face_roi = frame[y1:y2, x1:x2]
            x, emotion, conf, pred_class = predict_emotion(face_roi)

            heatmap = gradcam_heatmap(model, x, pred_class)
           
           
            superimposed = overlay(face_roi, heatmap)
            superimposed_bgr = cv2.cvtColor(superimposed, cv2.COLOR_RGB2BGR)
            frame[y1:y2, x1:x2] = superimposed_bgr

            #frame[y1:y2, x1:x2] = superimposed


            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{emotion}: {conf:.2f}"
            cv2.putText(frame, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        out.write(frame)
    
    cap.release()
    out.release()
    print(f"Processed video saved to {output_path}")


if __name__ == "__main__":
    input_video = "input.mp4"
    output_video = "output_gradcam.avi"

    process_video(input_video, output_video)


        








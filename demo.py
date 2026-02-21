import torch
import numpy as np
import cv2
import time
from torchvision import models , transforms
import torch.nn as nn
from model import EmotionCNN
from xAI.gradcam import gradcam, overlay_heatmap
from xAI.smoothGrad import coumpute_smoothGrad



# -------------------------
# Hook class
# -------------------------
class LayerActivation:
    def __init__(self, layer: nn.Module):
        self.activation = None
        self.hook = layer.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.activation = output.detach().cpu()

    def remove(self):
        self.hook.remove()

# -------------------------
# Get layer activation
# -------------------------
def get_layer_activation(model, layer, image_tensor):
    model.eval()
    hook = LayerActivation(layer)

    with torch.no_grad():
        _ = model(image_tensor)

    activation = hook.activation
    hook.remove()

    if activation is None:
        raise RuntimeError(
            "Hook did not capture any activation. "
            "Check that the layer is used in forward()."
        )

    return activation

# -------------------------
# Conv layer helpers
# -------------------------
def list_conv_layers(model: nn.Module):
    return [
        (name, m)
        for name, m in model.named_modules()
        if isinstance(m, nn.Conv2d)
    ]

def get_conv_layer(model: nn.Module, which: str = "last") -> nn.Module:
    convs = list_conv_layers(model)
    if not convs:
        raise RuntimeError("No Conv2d layers found in model.")

    if which == "first":
        return convs[0][1]
    elif which == "middle":
        return convs[len(convs) // 2][1]
    elif which == "last":
        return convs[-1][1]
    else:
        for name, layer in convs:
            if name == which:
                return layer
        raise ValueError(
            f"Unknown layer '{which}'. "
            "Use 'first', 'middle', 'last' or layer name."
        )

# -------------------------
# Activation → Heatmap
# -------------------------
def layer_activation_heatmap_from_tensor(
    activation: torch.Tensor
) -> torch.Tensor:

    if activation.dim() != 4:
        raise ValueError("Activation must have shape [B, C, H, W]")

    heat = activation.mean(dim=1)[0]

    min_val = heat.min()
    max_val = heat.max()

    if (max_val - min_val) < 1e-8:
        return torch.zeros_like(heat)

    heat = (heat - min_val) / (max_val - min_val)
    return heat

# -------------------------
# Overlay heatmap
# -------------------------
def overlay_heatmap(img, heatmap, alpha=0.4):
    """
    img: BGR image (H, W, 3), uint8
    heatmap: (H, W) torch.Tensor or np.ndarray in [0,1]
    """

    # Torch → NumPy
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.detach().cpu().numpy()

    heatmap = heatmap.astype(np.float32)

    heatmap = cv2.resize(
        heatmap,
        (img.shape[1], img.shape[0])
    )

    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    return cv2.addWeighted(
        img, alpha,
        heatmap, 1 - alpha,
        0
    )










device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])

idx_to_emotion = {
    0: "surprise", 1: "fear", 2: "disgust",
    3: "happiness", 4: "sadness", 5: "anger",
}


model = EmotionCNN(num_classes=6).to(device)
state = torch.load("best_model_cosine.pt", map_location=device)
model.load_state_dict(state)
model.eval()


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
if face_detector.empty():
    raise RuntimeError("Could not load haarcascade_frontalface_default.xml")
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")


def predict_emotion(face_bgr):
    with torch.no_grad():
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        face_rgb = cv2.resize(face_rgb, (64,64))
        x = preprocess(face_rgb).unsqueeze(0).to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)
        pred_idx = int(pred.item())
        conf = float(conf.item())
        emotion = idx_to_emotion.get(pred_idx, str(pred_idx))
        return emotion, conf, pred_idx

def compute_xai_overlay(face_bgr, mode):
    """Compute heatmap overlay for any XAI mode"""
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_rgb = cv2.resize(face_rgb, (64,64))
    x = preprocess(face_rgb).unsqueeze(0).to(device)
    x.requires_grad_(True)

    # Predict
    emotion, conf, pred_idx = predict_emotion(face_bgr)

    # Compute heatmap
    try:
        if mode == "gradcam":
            heatmap = gradcam(model, face_bgr, pred_idx)
        elif mode == "smoothgrad":
            heatmap = coumpute_smoothGrad(model, face_bgr, pred_idx, samples=50)
        elif mode == "activation":
            layer = get_conv_layer(model, which="last")
            activation = get_layer_activation(model, layer, x)
            heatmap = layer_activation_heatmap_from_tensor(activation)
    except Exception as e:
        print("XAI computation error:", e)
        heatmap = np.zeros((64,64), dtype=np.float32)

    # Normalize heatmap properly
    heatmap = np.abs(heatmap)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    overlay_img = overlay_heatmap(face_bgr, heatmap)
    text = f"{emotion}: {conf:.2f}"
    cv2.putText(
        overlay_img,
        text,
        (5, 20),  # top-left corner
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255,255,255),  # white text
        2,
        cv2.LINE_AA
    )
    return overlay_img, emotion, conf

# Live demo loop 
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (640,480))
    display_frame = frame.copy()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    # Draw rectangles and labels for live frame
    for (x, y, w, h) in faces:
        pad = int(0.15*w)
        x1, y1 = max(0,x-pad), max(0,y-pad)
        x2, y2 = min(frame.shape[1], x+w+pad), min(frame.shape[0], y+h+pad)
        face_roi = frame[y1:y2, x1:x2]

        emotion, conf, pred_idx = predict_emotion(face_roi)
        cv2.rectangle(display_frame, (x1,y1), (x2,y2), (0,255,0),2)
        label = f"{emotion}: {conf:.2f}"
        cv2.putText(display_frame, label, (x1,max(20,y1-10)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2,cv2.LINE_AA)

    cv2.imshow("Live Emotion Detector", display_frame)

    key = cv2.waitKey(1) & 0xFF

    # --- XAI overlay keys ---
    if key in [ord('g'), ord('s'), ord('v'), ord('a')]:
        mode = {ord('g'):"gradcam", ord('s'):"smoothgrad", ord('a'):"activation"}[key]

        if len(faces) > 0:
            x, y, w, h = faces[0]  # first face
            pad = int(0.15*w)
            x1, y1 = max(0,x-pad), max(0,y-pad)
            x2, y2 = min(frame.shape[1], x+w+pad), min(frame.shape[0], y+h+pad)
            face_roi = frame[y1:y2, x1:x2]

            overlay_img, emotion, conf = compute_xai_overlay(face_roi, mode)
            window_name = f"{mode.upper()} Overlay"
            cv2.imshow(window_name, overlay_img)
        else:
            print("No face detected for XAI overlay.")

    # Close overlay windows ---
    if key == ord('n'):
        for win in ["GRADCAM Overlay","SMOOTHGRAD Overlay","ACTIVATION Overlay"]:
            cv2.destroyWindow(win)

    # Quit
    if key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

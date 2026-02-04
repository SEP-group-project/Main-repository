import torch
import numpy as np
import cv2
import time
from torchvision import models , transforms
import torch.nn as nn
from torch.optim import Adam
import torch.optim as optim
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader



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
    





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])





def get_last_conv_layer(model):
    
    return model.features[3][3]




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



from captum.attr import Saliency, NoiseTunnel
import cv2
import numpy
from torchvision import transforms
import torch
import numpy as np




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

def coumpute_smoothGrad(model, img, target_class, samples):

    
    face_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_rgb = cv2.resize(face_rgb, (64, 64))
    x = preprocess(face_rgb).unsqueeze(0).to(device)
    
    
    
    saliency = Saliency(model)
    nt = NoiseTunnel(saliency)
    attribution = nt.attribute(x, nt_type='smoothgrad',nt_samples=samples, target=target_class )

    attr_np = attribution.squeeze().cpu().detach().numpy()
    heatmap = np.sum(np.abs(attr_np), axis=0)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    return heatmap





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




def occlusion_saliency(
        model,
        img,
        target_class: int,
        patch_size: int = 8,
        stride: int = 4,
        baseline: float = 0.0,
        use_softmax = False,
):

    model.eval()

    if img.dim() != 4 or img.shape[0] != 1:
        raise ValueError("Image must have shape [1, C, H, W] ")

    device = img.device
    _, C, H, W = img.shape

    out = model(img)
    if use_softmax:
        out = torch.softmax(out, dim=1)
    base_score = out[0, target_class].item()

    sal_sum = torch.zeros((H,W), device=device)
    sal_cnt = torch.zeros((H,W), device=device)

    for y in range(0, H, stride):
        y1 = y
        y2 = min(y + patch_size, H)

        for x in range(0, W, stride):
            x1 = x
            x2 = min(x + patch_size, W)

            occ = img.clone()
            occ[:, :, y1:y2, x1:x2] = baseline

            out_occ = model(occ)
            if use_softmax:
                out_occ = torch.softmax(out_occ, dim=1)
            
            occ_score = out_occ[0, target_class].item()
            drop = base_score - occ_score
            
            sal_sum[y1:y2, x1:x2] += drop
            sal_cnt[y1:y2, x1:x2] += 1.0

    sal = sal_sum / torch.clamp(sal_cnt, min=1.0)
    sal = torch.clamp(sal, min=0.0)

    s_min, s_max = sal.min(), sal.max()
    if (s_max > s_min) > 1e-12:
        sal = (sal - s_min) / (s_max - s_min)
    else:
        sal = torch.zeros_like(sal)

    return sal.detach().cpu()






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
        elif mode == "occlusion":
            heatmap = occlusion_saliency(model, face_bgr, pred_idx)
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
    if key in [ord('g'), ord('s'), ord('v'), ord('o'), ord('a')]:
        mode = {ord('g'):"gradcam", ord('s'):"smoothgrad", ord('o'):"occlusion", ord('a'):"activation"}[key]

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
        for win in ["GRADCAM Overlay","SMOOTHGRAD Overlay","OCCLUSION Overlay","ACTIVATION Overlay"]:
            cv2.destroyWindow(win)

    # Quit
    if key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

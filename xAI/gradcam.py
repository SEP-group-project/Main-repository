import torch
import numpy as np
import cv2
from torchvision import models, transforms


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



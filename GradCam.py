import torch
import numpy as np
import cv2
from torchvision import models



def get_class_label(preds):
   return preds.argmax(dim=1).item()

def get_conv_layer(model, conv_layer_name):
    for name, layer in model.named_modules():
        if name == conv_layer_name:
            return layer
    raise ValueError(f"Layer '{conv_layer_name}' not found in the model.")




def gradcam(model, img_tensor, class_index, conv_layer_name=" "):
    conv_layer = get_conv_layer(model, conv_layer_name)
    model.eval()

    activations = None
    gradients = None
    
    def forward_hook(module, input, output): 
        nonlocal activations
        activations = output     #output: feature maps of the last conv-layer
    
    def backward_hook(module, grad_in, grad_out): 
        nonlocal gradients
        gradients = grad_out[0]  #gradient of feature maps

    
    hook_f = conv_layer.register_forward_hook(forward_hook)  
    hook_b = conv_layer.register_backward_hook(backward_hook)

    
    preds = model(img_tensor)
    score = preds[:, class_index]  # score of the predicted class
    model.zero_grad()
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

def overlay_heatmap(img_path, heatmap, alpha=0.4):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(img, alpha, heatmap, 1-alpha, 0)
    return superimposed_img



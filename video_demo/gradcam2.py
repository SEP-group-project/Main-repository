import cv2
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image




def gradcam_heatmap(model, x, pred_index):
    classes = [ClassifierOutputTarget(pred_index)]
    layers = [model.features[3][3]]

    cam = GradCAM(model=model, target_layers=layers)
    heatmap = cam (input_tensor=x, targets= classes)

    return heatmap

def overlay(frame_bgr, heatmap):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb = frame_rgb.astype(np.float32) / 255.0
    heatmap_resized = cv2.resize(heatmap[0], (frame_rgb.shape[1], frame_rgb.shape[0]))
    superimposed = show_cam_on_image(frame_rgb, heatmap_resized, use_rgb=True)

    return superimposed
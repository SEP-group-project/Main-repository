from captum.attr import Saliency, NoiseTunnel
import cv2
import numpy

def coumpute_smoothGrad(model, img, target_class):

    
    face_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_rgb = cv2.resize(face_rgb, (64, 64))
    x = preprocess(face_rgb).unsqueeze(0).to(device)
    
    
    
    saliency = Saliency(model)
    nt = NoiseTunnel(saliency)
    attribution = nt.attribution(img, nt_type='smoothgrad',nt_samples=10, target=target_class )

    attr_np = attribution.squeeze().cpu().detach().numpy()
    heatmap = np.sum(np.abs(attr_np), axis=0)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    return heatmap


import torch

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
   
    #for top in range(0, H, stride):
    #    bottom = min(top + patch_size, H)
    #    top = max(0, bottom - patch_size)

    #    for left in range(0, W, stride):
    #        right = min(left + patch_size, W)
    #        left = max(0, right - patch_size)
        
    #occ = img.clone()
    #occ[:, :, top:bottom, left:right] = baseline

    #out_occ = model(occ)
    #if use_softmax:
    #    out_occ = torch.softmax(out_occ, dim=1)
    #occ_score = out_occ[0, target_class].item()

    #drop = base_score - occ_score
    #sal_sum[top:bottom, left:right] += drop
    #sal_cnt[top:bottom, left:right] += 1.0

    sal = sal_sum / torch.clamp(sal_cnt, min=1.0)
    sal = torch.clamp(sal, min=0.0)

    s_min, s_max = sal.min(), sal.max()
    if (s_max > s_min) > 1e-12:
        sal = (sal - s_min) / (s_max - s_min)
    else:
        sal = torch.zeros_like(sal)

    return sal.detach().cpu()


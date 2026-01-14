import torch


def gradient_saliency(model, img, target_class):
    model.eval()
    img.requires_grad_()

    pred = model(img)
    score = pred[0, target_class]

    score.backward()

    saliency, _ = torch.max(img.grad.detach.abs(), dim=1)

    return saliency[0].cpu()


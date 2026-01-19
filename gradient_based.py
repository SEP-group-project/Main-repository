from captum.attr import Saliency, NoiseTunnel


def saliency_with_SmoothGrad(model, img, target_class):
    
    saliency = Saliency(model)
    nt = NoiseTunnel(saliency)
    attribution = nt.attribution(img, nt_typ='smoothgrad',nt_samples=10, target=target_class )

    return attribution


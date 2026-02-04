import torch
import torch.nn as nn

class LayerActivation:
    def __init__(self, layer: torch.nn.Module):
        self.activation = None
        self.hook = layer.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.activation = output.detach().cpu()

    def remove(self):
        self.hook.remove()

def get_layer_activation(model, layer, image_tensor):
    
    model.eval()

    hook = LayerActivation(layer)

    with torch.no_grad():
        _ = model(image_tensor)

    activation = hook.activation
    hook.remove()

    if activation is None:
        raise RuntimeError("Hook did not capture any activation, check that the layer is used in forward.")

    return activation

def list_conv_layers(model: nn.Module):
    convs = []
    for name, m in model.named.modules():
        is isinstance(m, nn.Conv2d):
        convs.append((name,m))
    return convs

def get_conv_layer(model: nn.Module, which: str = "last") -> nn.Module:
    convs = list_conv_layers(model)
    if not convs:
        raise RuntimeError("No conv2d layers found in this model.")

    if which == "first":
        return convs[0][1]
    elif which == "middle":
        return convs[len(convs)//2][1]
    elif which == "last":
        return convs[-1][1]
    else:
        for name, layer in convs:
            if name == which:
                return layer
        raise ValueError(f"Unknown layer selector '{which}'. Use 'first'/'middle'/'last' or a conv layer name.")

def layer_activation_heatmap_from_tensor(activation: torch.Tensor) -> torch.Tensor:
    if activation.dim() != 4:
        raise ValueError( Activation must have shape [B, C, H, W]")

    heat = activation.mean(dim=1)[0]
    heat = heat-heat.min
    heat = heat / (heat.max() + 1e-8)

    return heat

import torch

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

    return activation
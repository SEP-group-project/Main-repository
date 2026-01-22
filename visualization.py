import matplotlib.pyplot as plt
import math

def visualize_activation(activation, max_channel=64):

    assert activation.dim() == 4, "Activation tensor must be of shape [B, C, H, W]"

    num_channels = min(activation.shape[1], max_channel)
    grid_size = math.ceil(math.sqrt(num_channels))

    plt.figure(grid_size * 3, grid_size * 3)
    for i in range(num_channels):
        plt.subplot(grid_size, grid_size, i + 1)
        plt.imshow(activation[0, i].cpu().numpy(), cmap='viridis')
        plt.axis('off')
        plt.title(f'Channel {i+1}')
    
    plt.suptitle('Activation Maps', fontsize=16)
    plt.tight_layout()
    plt.show()

def visualize_mean_activation(activation):

    assert activation.dim() == 4, "Activation tensor must be of shape [B, C, H, W]"

    mean_activation = activation.mean(dim=1, keepdim=True)

    plt.figure(figsize=(6, 6))
    plt.imshow(mean_activation[0, 0].cpu().numpy(), cmap='viridis')
    plt.axis('off')
    plt.title('Mean Activation Map', fontsize=16)
    plt.show()
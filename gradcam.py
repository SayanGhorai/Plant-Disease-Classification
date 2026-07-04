import torch
import numpy as np
from PIL import Image


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, class_idx=None):
        # Forward pass
        output = self.model(input_tensor)

        # Predicted class
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Backward pass
        self.model.zero_grad()
        output[:, class_idx].backward()

        # Extract gradients and activations
        gradients = self.gradients[0]
        activations = self.activations[0]

        # Global average pooling
        weights = gradients.mean(dim=(1, 2))

        # Create CAM
        cam = torch.zeros(
            activations.shape[1:],
            dtype=torch.float32,
            device=activations.device
        )

        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU
        cam = torch.relu(cam)

        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        # Keep only strong activations (remove weak noise)
        cam[cam < 0.35] = 0

        return cam.detach().cpu().numpy()


def create_heatmap(cam, original_image):
    # Normalize
    cam = cam - np.min(cam)
    cam = cam / (np.max(cam) + 1e-8)

    # Keep stronger important regions only
    threshold = 0.65
    cam = np.where(cam >= threshold, cam, 0)

    # Convert
    cam = np.uint8(255 * cam)

    heatmap = Image.fromarray(cam)

    # Smooth but not blocky
    heatmap = heatmap.resize(
        original_image.size,
        Image.Resampling.BILINEAR
    )

    return heatmap
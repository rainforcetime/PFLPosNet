import time

import numpy as np
import torch
import torch.nn as nn
# import torch.nn as nn
from torchvision import models


class DepthDetector(nn.Module):
    def __init__(self, cfg):
        super(DepthDetector, self).__init__()
        self.seq_len = cfg.seq_len
        self.window_size = cfg.window_size
        self.hidden_dim = cfg.hidden_dim
        self._3dmm_dim = cfg._3dmm_dim
        self.depth_dim = cfg.depth_dim
        self.pretrained = cfg.pretrained

        # Load a pretrained ResNet model and modify it to suit our needs
        resnet = models.resnet18(pretrained=self.pretrained)

        # Remove the final fully connected layer from ResNet
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)

        # Define a new fully connected layer for depth estimation
        self.fc_depth = nn.Linear(resnet.fc.in_features, self.depth_dim)


    def forward(self, x, y):
        """
        :param x: video_clip: [batch_size, seq_len, 3, 224, 224]
        :param y: _3dmm_clip: [batch_size, seq_len, _3dmm_dim=58]
        :return:
        """
        batch_size, seq_len, channels, height, width = x.shape

        window_start = torch.randint(0, seq_len - self.window_size, (1,), device=x.device)
        window_end = window_start + self.window_size
        x = x[:, window_start:window_end]
        # Now selected_frames has shape [batch_size, window_size, 3, 224, 224]

        # Process selected frames with ResNet
        x = x.reshape(-1, channels, height, width)  # Flatten the batch and window dimensions
        x = self.resnet(x)  # Output shape: [batch_size * window_size, resnet.fc.in_features, 1, 1]
        x = x.reshape(batch_size * self.window_size, -1)  # Flatten the tensor

        # Pass through the fully connected layer for depth estimation
        x = self.fc_depth(x)

        # Reshape back to include window_size
        x = x.reshape(batch_size, self.window_size, -1).squeeze(-1) # [batch_size, window_size]
        y = y[:, window_start:window_end, -1]
        output = {
            "prediction": x,
            "target": y
        }

        return output

    def get_depth(self, x):
        """
        :param x: [batch_size, 3, 224, 224]
        :return:
        """
        batch_size, channels, height, width = x.shape

        x = self.resnet(x)  # Output shape: [batch_size * window_size, resnet.fc.in_features, 1, 1]
        x = x.reshape(batch_size, -1)  # Flatten the tensor

        # Pass through the fully connected layer for depth estimation
        x = self.fc_depth(x)

        # Reshape back to include window_size
        x = x.reshape(batch_size, -1)

        return x


if __name__ == "__main__":
    # Define a configuration object
    class Config:
        def __init__(self):
            self.seq_len = 600
            self.window_size = 50
            self.hidden_dim = 512
            self._3dmm_dim = 58
            self.depth_dim = 1
            self.pretrained = True

    cfg = Config()

    # Check if CUDA is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create an instance of the DepthDetector model and move it to the device
    model = DepthDetector(cfg).to(device)

    # Create some dummy input tensors and move them to the device
    video_clip = torch.randn(4, 600, 3, 224, 224, device=device)
    _3dmm_clip = torch.randn(4, 600, 58, device=device)

    # Set the model to evaluation mode
    model.eval()

    # Perform a forward pass
    with torch.no_grad():  # 禁用梯度计算以节省内存
        output = model(video_clip, _3dmm_clip)

    # Display the output
    for key, value in output.items():
        print(f"{key}: {value.shape}")
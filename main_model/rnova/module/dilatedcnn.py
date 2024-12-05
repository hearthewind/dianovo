import math

import torch.nn as nn
import torch.nn.functional as F

class DilatedConvolutionModule(nn.Module):
    def __init__(self, feature_dim, output_dim=256, max_time=5):
        super(DilatedConvolutionModule, self).__init__()

        hidden_size = output_dim // 8

        # Determine the necessary number of layers and dilation to reduce timesteps to 1
        self.conv_layers = []
        self.ln_layers = []
        current_timesteps = max_time
        layer_idx = 0
        while current_timesteps > 1:
            dilation = 2 ** layer_idx
            kernel_size = 3  # You can choose a different size if needed
            padding = (kernel_size - 1) * dilation // 2  # to maintain output length

            # Ensure the last convolutional layer reduces the timestep to 1
            final_layer = False
            if (current_timesteps - 1) // (dilation * (kernel_size - 1)) < 1:
                final_layer = True
                dilation = math.ceil((current_timesteps - 1) / (kernel_size - 1))
                padding = 0  # No padding for the last layer to reduce to single timestep

            conv = nn.Conv1d(feature_dim if layer_idx == 0 else hidden_size, hidden_size,
                             kernel_size=kernel_size, dilation=dilation, padding=padding)
            ln = nn.LayerNorm([hidden_size, max_time if not final_layer else 1])
            self.conv_layers.append(conv)
            self.ln_layers.append(ln)
            self.add_module('conv{}'.format(layer_idx), conv)
            self.add_module('ln{}'.format(layer_idx), ln)

            # Calculate the size of the output after this layer
            current_timesteps = math.floor((current_timesteps + 2 * padding - dilation * (kernel_size - 1) - 1) / 1) + 1
            layer_idx += 1

        self.output_layer = nn.Conv1d(hidden_size, output_dim, kernel_size=1)
        self.output_ln = nn.LayerNorm(output_dim)

    def forward(self, x):
        # x shape: (batch_size, num_points, timesteps, feature_dim)
        # We need to process over the timesteps, so we transpose the timesteps and feature_dim
        batch_size, num_points, timesteps, feature_dim = x.size()
        x = x.transpose(2, 3)  # New shape: (batch_size, num_points, feature_dim, timesteps)

        # Now, flatten the batch_size and num_points dimensions to treat each series of points as independent
        x = x.contiguous().view(batch_size * num_points, feature_dim, timesteps)

        # Pass through the dilated convolution layers
        for i, conv in enumerate(self.conv_layers):
            x = conv(x)
            x = self.ln_layers[i](x)
            x = F.relu(x)

        # Pass through the output layer to get the final feature dimension we want
        x = self.output_layer(x)

        # Now we need to reshape to get back to the (batch_size, num_points, output_dim)
        x = x.view(batch_size, num_points, -1)

        x = self.output_ln(x)

        return x
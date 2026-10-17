import torch.nn as nn
import torch

# From Momask [Guo et al. 2024]
class ResConv1DBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dilation=1, dropout=0.2):
        super(ResConv1DBlock, self).__init__()

        padding = dilation

        self.activation1 = nn.GELU()
        self.activation2 = nn.GELU()
        self.conv1 = nn.Conv1d(input_dim, output_dim, 3, 1, padding, dilation)
        self.conv2 = nn.Conv1d(output_dim, input_dim, 1, 1, 0, )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_orig = x
        x = self.activation1(x)

        x = self.conv1(x)
        x = self.activation2(x)

        x = self.conv2(x)
        x = self.dropout(x)
        x = x + x_orig # skip connection
        return x


class Resnet1D(nn.Module):
    def __init__(self, input_dim, num_blocks, dilation_growth_rate=1, reverse_dilation=True):
        super().__init__()

        blocks = [ResConv1DBlock(input_dim, input_dim, dilation=dilation_growth_rate ** depth)
                  for depth in range(num_blocks)]
        if reverse_dilation:
            blocks = blocks[::-1]

        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)
# will do the following:
# - contains PyTorch code for defining the ResNet CNN architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
import hyperion_nn.config as config

class ResidualBlock(nn.Module): # nn.Module is the parent class
    """A basic residual block with two convolutional layers."""

    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=config.ModelConfig.NUM_FILTERS, 
                               out_channels=config.ModelConfig.NUM_FILTERS,
                               kernel_size=3,
                               padding='same')
        self.bn1 = nn.BatchNorm2d(num_features=config.ModelConfig.NUM_FILTERS)
        self.conv2 = nn.Conv2d(in_channels=config.ModelConfig.NUM_FILTERS, 
                               out_channels=config.ModelConfig.NUM_FILTERS,
                               kernel_size=3,
                               padding='same')
        self.bn2 = nn.BatchNorm2d(num_features=config.ModelConfig.NUM_FILTERS)

    def forward(self, x):
        identity = x

        # first transformation/convolution layer
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        # second transformation/convolution layer
        out = self.conv2(out)
        out = self.bn2(out)

        # skip connection
        out += identity

        # final activation
        out = F.relu(out)
        return out


class HyperionNN(nn.Module):
    """The main ResNet CNN model for HyperionNN."""

    def __init__(self):
        super(HyperionNN, self).__init__()
        
        # first entry block/layer
        self.entry_block = nn.Sequential(
            nn.Conv2d(in_channels=config.ModelConfig.NUM_INPUT_PLANES,
                      out_channels=config.ModelConfig.NUM_FILTERS,
                      kernel_size=3,
                      padding='same'),
            nn.BatchNorm2d(num_features=config.ModelConfig.NUM_FILTERS),
            nn.ReLU()
        )

        # residual tower/blocks
        self.residual_ = nn.Sequential(
            *[ResidualBlock() for _ in range(config.ModelConfig.NUM_RESIDUAL_BLOCKS)]
        )

        # policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(in_channels=config.ModelConfig.NUM_FILTERS,
                      out_channels=2,  # 64 is a common choice for policy head
                      kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=2 * 8 * 8, out_features=config.ModelConfig.POLICY_HEAD_SIZE)  # assuming input size is 8x8
        )

        # value head
        self.value_head = nn.Sequential(
            nn.Conv2d(in_channels=config.ModelConfig.NUM_FILTERS,
                      out_channels=1,  # single output for value head
                      kernel_size=1),
            nn.BatchNorm2d(num_features=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=1 * 8 * 8, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=1)  # final output for value head
        )

    def forward(self, x):
        # entry block
        out = self.entry_block(x)

        # residual tower
        out = self.residual_(out)

        # policy head
        policy_logits = self.policy_head(out)

        # value head
        value_logits = self.value_head(out)

        return policy_logits, value_logits
        


        
        
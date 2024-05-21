# Some code from https://github.com/deepmind/deepmind-research/blob/master/adversarial_robustness/pytorch/model_zoo.py
import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from typing import Tuple, Union
from src.models.ResBlock import *
from src.models.WRN import _BlockGroup
WIDERESNET_WIDTH_WANG2023=10
WIDERESNET_WIDTH_MNIST=4
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2471, 0.2435, 0.2616)

class WRN2810VarHead(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        #The input to the head is the output of the body which is 64*width (where width is the width of the ResNet).
        self.fc1 = nn.Linear(64*WIDERESNET_WIDTH_WANG2023, latent_dim*3)
        self.fc2 = nn.Linear(latent_dim*3, latent_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.logsigmoid(self.fc2(x))
        return x

class WRN2810Head(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        #The input to the head is the output of the body which is 64*width (where width is the width of the ResNet).
        self.fc1 = nn.Linear(64*WIDERESNET_WIDTH_WANG2023, latent_dim*3)
        self.fc2 = nn.Linear(latent_dim*3, latent_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class WRN2810Body(nn.Module):
    """
    Adapted WideResNet model
    Arguments:
        num_classes (int): number of output classes.
        depth (int): number of layers.
        width (int): width factor.
        activation_fn (nn.Module): activation function.
        mean (tuple): mean of dataset.
        std (tuple): standard deviation of dataset.
        padding (int): padding.
        num_input_channels (int): number of channels in the input.
    """
    def __init__(self,
                 num_classes: int = 10,
                 depth: int = 28,
                 width: int = 10,
                 activation_fn: nn.Module = nn.ReLU,
                 mean: Union[Tuple[float, ...], float] = CIFAR10_MEAN,
                 std: Union[Tuple[float, ...], float] = CIFAR10_STD,
                 padding: int = 0,
                 num_input_channels: int = 3):
        super().__init__()
        self.padding = padding
        num_channels = [16, 16 * width, 32 * width, 64 * width]
        assert (depth - 4) % 6 == 0
        num_blocks = (depth - 4) // 6
        self.num_input_channels = num_input_channels
        self.init_conv = nn.Conv2d(num_input_channels, num_channels[0],
                                   kernel_size=3, stride=1, padding=1, bias=False)
        self.layer = nn.Sequential(
            _BlockGroup(num_blocks, num_channels[0], num_channels[1], 1,
                        activation_fn=activation_fn),
            _BlockGroup(num_blocks, num_channels[1], num_channels[2], 2,
                        activation_fn=activation_fn),
            _BlockGroup(num_blocks, num_channels[2], num_channels[3], 2,
                        activation_fn=activation_fn))
        self.batchnorm = nn.BatchNorm2d(num_channels[3], momentum=0.01)
        self.relu = activation_fn(inplace=True)
        #self.fc = nn.Linear(num_channels[3], latent_dim)
        self.num_channels = num_channels[3]
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    
    def forward(self, x):
        if self.padding > 0:
            x = F.pad(x, (self.padding,) * 4)
        out = self.init_conv(x)
        out = self.layer(out)
        out = self.relu(self.batchnorm(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.num_channels)
        return out
import torch
import torch.nn as nn



class Block(nn.Module):
  def __init__(self, in_channels, out_channels, device):
    super(Block, self).__init__()
    self.device = device
    self.intermediate_layers = {}
    self.out_channels = out_channels
    #Layer 1
    self.conv1_1 = nn.Conv2d(in_channels, out_channels, kernel_size = (3,3), padding=1)

    #Layer 2
    self.intermediate_layers[2] = self.create_subsequent_layers()
    #Layer 
    self.intermediate_layers[3] = self.create_subsequent_layers()
    #Layer 4
    self.intermediate_layers[4] = self.create_subsequent_layers()
    #Layer 5
    self.intermediate_layers[5] = self.create_subsequent_layers()
  
  def create_subsequent_layers(self):
    return nn.Sequential(nn.BatchNorm2d(self.out_channels), 
                         nn.ReLU(), 
                         nn.Conv2d(self.out_channels, self.out_channels, kernel_size = (3,3), padding=1)).to(self.device)
  
  def forward(self, x):
    #Input layer
    x_1 = self.conv1_1(x)
    x_2 = self.intermediate_layers[2](x_1)
    x_2a = x_1+x_2

    x_3 = self.intermediate_layers[3](x_2a)
    x_3a = x_1+x_3

    x_4 = self.intermediate_layers[4](x_3a)
    x_4a = x_1+x_4

    #Output layer
    x_5 = self.intermediate_layers[5](x_4a)
    x_5a = x_1+x_5
    return x_5a


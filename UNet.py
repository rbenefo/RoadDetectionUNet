import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm.notebook import tqdm
import os
import numpy as np
from torchvision import datasets, transforms
from skimage import io
import time
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from block import Block
class UNet(nn.Module):
  def __init__(self, device):
    super(UNet, self).__init__()
    ##Encoder
    self.device = device
    self.encoder_block1 = Block(3, 64, device)
    self.max_pool1 = nn.MaxPool2d(kernel_size =(2,2), stride = 2)
    self.encoder_block2 = Block(64, 128, device)
    self.max_pool2 = nn.MaxPool2d(kernel_size =(2,2), stride = 2)
    self.encoder_block3 = Block(128, 256, device)
    self.max_pool3 = nn.MaxPool2d(kernel_size =(2,2), stride = 2)


    #Bridge
    self.bridge = Block(256, 512, device)

    ##Decoder
    self.upsample_1 = nn.ConvTranspose2d(512, 256, kernel_size = (2,2), stride = 2)
    self.decoder_block1 = Block(512, 256, device)
    self.upsample_2 = nn.ConvTranspose2d(256, 128, kernel_size = (2,2), stride = 2)
    self.decoder_block2 = Block(256, 128, device)
    self.upsample_3 = nn.ConvTranspose2d(128, 64, kernel_size = (2,2), stride = 2)
    self.decoder_block3 = Block(128, 64, device)

    ##Output
    self.output_conv = nn.Conv2d(64, 1, kernel_size =(1,1), stride = 1)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    #B, N, N, C --> B, C, N, N
    # x = x.permute(0, 3, 1, 2)

    ##Encoder
    x_e1 = self.encoder_block1(x) #--> decoder layer 3
    x_e2 = self.max_pool1(x_e1) 

    x_e2 = self.encoder_block2(x_e2)  #--> decoder layer 2
    x_e3 = self.max_pool2(x_e2)

    x_e3 = self.encoder_block3(x_e3)#--> decoder layer 1
    x_e4 = self.max_pool3(x_e3)

    #Bridge
    x_b = self.bridge(x_e4)

    #Decoder
    x_d1 = self.upsample_1(x_b)
    x_d1 = torch.cat((x_e3, x_d1), dim = 1)
    x_d2 = self.decoder_block1(x_d1)

    x_d2 = self.upsample_2(x_d2)
    x_d2 = torch.cat((x_e2, x_d2), dim = 1)
    x_d3 = self.decoder_block2(x_d2)

    x_d3 = self.upsample_3(x_d3)
    x_d3 = torch.cat((x_e1, x_d3), dim = 1)
    x_d4 = self.decoder_block3(x_d3)

    x_out = self.output_conv(x_d4)
    x_out = self.sigmoid(x_out)
    return x_out



class UNet_Medium(nn.Module):
  def __init__(self, device):
    super(UNet_Medium, self).__init__()
    ##Encoder
    self.device = device
    self.encoder_block1 = Block(3, 64, device)
    self.max_pool1 = nn.MaxPool2d(kernel_size =(2,2), stride = 2)
    self.encoder_block2 = Block(64, 256, device)
    self.max_pool2 = nn.MaxPool2d(kernel_size =(2,2), stride = 2)
    # self.encoder_block3 = Block(128, 256, device)
    # self.max_pool3 = nn.MaxPool2d(kernel_size =(2,2), stride = 2)

    #Bridge
    self.bridge = Block(256, 512, device)

    ##Decoder
    # self.upsample_1 = nn.ConvTranspose2d(512, 256, kernel_size = (2,2), stride = 2)
    # self.decoder_block1 = Block(512, 256, device)
    self.upsample_2 = nn.ConvTranspose2d(512, 256, kernel_size = (2,2), stride = 2)
    self.decoder_block2 = Block(512, 256, device)
    self.upsample_3 = nn.ConvTranspose2d(256, 64, kernel_size = (2,2), stride = 2)
    self.decoder_block3 = Block(128, 64, device)

    ##Output
    self.output_conv = nn.Conv2d(64, 1, kernel_size =(1,1), stride = 1)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    #B, N, N, C --> B, C, N, N
    # x = x.permute(0, 3, 1, 2)

    ##Encoder
    x_e1 = self.encoder_block1(x) #--> decoder layer 3
    x_e2 = self.max_pool1(x_e1) 

    x_e2 = self.encoder_block2(x_e2)  #--> decoder layer 2; size: 256
    x_e3 = self.max_pool2(x_e2)

    # x_e3 = self.encoder_block3(x_e3)#--> decoder layer 1
    # x_e4 = self.max_pool3(x_e3)

    #Bridge
    x_b = self.bridge(x_e3)

    #Decoder
    # x_d1 = self.upsample_1(x_b)
    # x_d1 = torch.cat((x_e3, x_d1), dim = 1)
    # x_d2 = self.decoder_block1(x_d1)

    x_d2 = self.upsample_2(x_b)
    x_d2 = torch.cat((x_e2, x_d2), dim = 1)
    x_d3 = self.decoder_block2(x_d2)

    x_d3 = self.upsample_3(x_d3)
    x_d3 = torch.cat((x_e1, x_d3), dim = 1)
    x_d4 = self.decoder_block3(x_d3)

    x_out = self.output_conv(x_d4)
    x_out = self.sigmoid(x_out)
    return x_out


class UNetSmall(nn.Module):
  def __init__(self, device):
    super(UNetSmall, self).__init__()
    ##Encoder
    self.device = device
    self.encoder_block1 = Block(3, 32, device)
    self.max_pool1 = nn.MaxPool2d(kernel_size =(2,2), stride = 2)
    self.encoder_block2 = Block(32, 64, device)
    self.max_pool2 = nn.MaxPool2d(kernel_size =(2,2), stride = 2)


    #Bridge
    self.bridge = Block(64, 128, device)

    ##Decoder
    self.upsample_1 = nn.ConvTranspose2d(128, 64, kernel_size = (2,2), stride = 2)
    self.decoder_block1 = Block(128, 64, device)
    self.upsample_2 = nn.ConvTranspose2d(64, 32, kernel_size = (2,2), stride = 2)
    self.decoder_block2 = Block(64, 32, device)

    ##Output
    self.output_conv = nn.Conv2d(32, 1, kernel_size =(1,1), stride = 1)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    #B, N, N, C --> B, C, N, N
    # x = x.permute(0, 3, 1, 2)

    ##Encoder
    x_e1 = self.encoder_block1(x) #--> decoder layer 3
    x_e2 = self.max_pool1(x_e1) 
    
    x_e2 = self.encoder_block2(x_e2)  #--> decoder layer 2
    x_e3 = self.max_pool2(x_e2)


    #Bridge
    x_b = self.bridge(x_e3)

    #Decoder
    x_d1 = self.upsample_1(x_b)
    x_d1 = torch.cat((x_e2, x_d1), dim = 1)
    x_d2 = self.decoder_block1(x_d1)

    x_d2 = self.upsample_2(x_d2)
    x_d2 = torch.cat((x_e1, x_d2), dim = 1)
    x_d3 = self.decoder_block2(x_d2)

    x_out = self.output_conv(x_d3)
    x_out = self.sigmoid(x_out)
    return x_out



import torch.optim as optim
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import os
import numpy as np
import time
from UNet import *
from training_functions import *
from torchvision import datasets, transforms


def train(net, device):
    # net = UNet(device).to(device)
    criterion = nn.BCELoss(reduce=False)
    # criterion = dice_coeff
    optimizer = optim.Adam(net.parameters(), lr=0.00008)
    train_test(net, criterion, optimizer, train_dataloader, device, num_epochs=13)
    torch.save(net, "bestUNetMedium427-1.pt")

def vis_output(net, validation_dataloader):
    data = next(iter(validation_dataloader))
    sat, mask, sat_unnorm = data
    net.eval()
    pred = net(sat.to(device).float())
    pred_numpy = pred[0].cpu().detach().numpy()
    binarized_numpy = np.copy(pred_numpy)
    binarized_numpy[np.where(binarized_numpy >= 0.5)] = 1
    binarized_numpy[np.where(binarized_numpy < 0.5)] = 0
    plt.imshow(pred_numpy.squeeze())
    plt.title("Non Binarized Output")
    plt.show()
    
    plt.figure()
    plt.imshow(binarized_numpy.squeeze())
    plt.title("Binarized Output")
    plt.show()

    mask_show = mask[0]
    plt.figure()
    plt.title("Ground Truth Mask")
    plt.imshow(mask_show.squeeze())
    plt.show()

    plt.figure()
    plt.title("Input Satellite Image")
    plt.imshow(sat_unnorm[idx].squeeze())
    plt.show()



if __name__ == "__main__":
    device = "cuda"
    net = UNet_Medium(device).to(device)

    tforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainTest = TrainTest("train2_resized_np_sat/", "train2_resized_np_mask/")
    train_dataloader, validation_dataloader = trainTest.create_datasets(transforms = tforms)

    net = train(net, device)
    vis_output(net, validation_dataloader)
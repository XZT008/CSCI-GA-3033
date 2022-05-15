import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, transforms




def get_data_loader(opt, train=True, transform=None):
    mnistdata = datasets.FashionMNIST(opt.data_path,
                                      download=True,
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.2860,), (0.3530,))
                                      ])
                                      )

    data_loader = torch.utils.data.DataLoader(mnistdata,
                                               batch_size=opt.batch_size,
                                               shuffle=True,
                                               num_workers=opt.workers,
                                               pin_memory=True,
                                               )

    return data_loader

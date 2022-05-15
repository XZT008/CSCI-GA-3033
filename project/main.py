import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import argparse

import torchvision
from torchvision import datasets, transforms
import pandas as pd
from dataset import get_data_loader
import os
import time

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7 * 7 * 32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return F.log_softmax(out, dim=1)


def train(model, device, train_dataloader, optimizer, epoch):
    model.train()
    correct = 0.0
    total_loss = 0.0
    for i, (data, target) in enumerate(train_dataloader):
        data, target = data.to(device), target.to(device)
        pred = model.forward(data)
        loss = F.nll_loss(pred, target, reduction='mean')
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        pred = pred.argmax(dim=1)
        correct += pred.eq(target.view_as(pred)).sum().item()

    correct /= len(train_dataloader.dataset)
    total_loss /= len(train_dataloader.dataset)
    print(">> epoch {} train loss:{:.4f}, accuracy:{:.2f}%".format(epoch, total_loss, correct * 100))
    acc_l.append(correct * 100)
    loss_l.append(total_loss)


def test(model, device, test_dataloader):
    total_loss = 0.0
    correct = 0.0
    model.eval()
    with torch.no_grad():
        for i, (data, target) in enumerate(test_dataloader):
            data, target = data.to(device), target.to(device)
            pred = model.forward(data)
            loss = F.nll_loss(pred, target, reduction='sum')
            total_loss += loss.item()
            pred = pred.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()
    total_loss /= len(test_dataloader.dataset)
    correct /= len(test_dataloader.dataset)
    print(">> epoch {} test loss:{:.4f}, accuracy:{:.2f}%".format(epoch, total_loss, correct * 100))
    val_acc_l.append(correct * 100)
    val_loss_l.append(total_loss)
    model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./DataSet', help='data path')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch-size', type=int, default=64, help='total batch size for all GPUs')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--momentum', type=float, default=0.8, help='momentum for SGD')
    parser.add_argument('--workers', type=int, default=1, help='maximum number of dataloader workers')
    parser.add_argument('--optimizer', type=str, default='RMSprop', choices=['SGD', 'SGD_m', 'Adagrad', 'Adam', 'RMSprop'],
    help='use which optimizer')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    opt = parser.parse_args()


    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")


    train_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.2860,), (0.3530,))])
    train_dataloader = get_data_loader(opt, train=True, transform=train_transform)

    test_transform = transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.2860,), (0.3530,))])
    test_dataloader = get_data_loader(opt, train=False, transform=test_transform)
    model = CNN().to(device)
    criterion = torch.nn.CrossEntropyLoss()

    optim_dict = {'SGD_m': optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum),
                  'SGD': optim.SGD(model.parameters(), lr=opt.lr),
                  'Adagrad': optim.Adagrad(model.parameters(), lr=opt.lr),
                  'Adam': optim.Adam(model.parameters(), lr=opt.lr),
                  'RMSprop': optim.RMSprop(model.parameters(), lr=opt.lr)}



    optimizer = optim_dict[opt.optimizer]
    acc_l=[]
    val_acc_l=[]
    loss_l = []
    val_loss_l = []
    st = time.time()
    for epoch in range(opt.epochs):
        train(model, device, train_dataloader, optimizer, epoch)
        test(model, device, test_dataloader)
        if val_acc_l[-1] >= 98:
            break
    et = time.time()
    print(f'Time for training optim={opt.optimizer} is: {et-st}\n')
    df = pd.DataFrame({'loss': loss_l,
                           'val_loss': val_loss_l,
                           'acc': acc_l,
                           'val_acc': val_acc_l
                           })
    df.to_csv(os.path.join('/scratch/zx581/project/out', f'{opt.optimizer}.csv'))

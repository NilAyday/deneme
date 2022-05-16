import pickle
import copy
import random

import torch
import torch.nn as nn
import torch.nn.functional as func
import matplotlib.pyplot as plt
import numpy as np
from src import datasets, proj_utils, training
import os
from svd import get_Jacobian_svd

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# ResNet
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

num_epochs = 100
lr = 0.02
num_data = 2000
batch_size = 1

ds_train = datasets.load_CIFAR10(True)
ds_test = datasets.load_CIFAR10(False)

indices = torch.arange(num_data)
ds_train = torch.utils.data.Subset(ds_train, indices)
ds_test = torch.utils.data.Subset(ds_test, indices)

dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size)
dl_test = torch.utils.data.DataLoader(ds_test, batch_size=batch_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNet(ResidualBlock, [2, 2, 2]).to(device)

sv1=get_Jacobian_svd(model,dl_train)

optimizer = torch.optim.SGD(model.parameters(), lr = lr)
loss = torch.nn.CrossEntropyLoss()

s = training.train(model, optimizer, loss, dl_train, dl_test, num_epochs, device=device)
#print(s)

sv2=get_Jacobian_svd(model,dl_train)



'''
def plot_loghist(x, bins,label):
    hist, bins = np.histogram(x, bins=bins)
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    plt.hist(x, bins=logbins,alpha=0.5,label=label)
    plt.xscale('log')
#bins = np.linspace(0.01, 10, 100)
#plt.hist(sv,bins,range=[10^-2,10])
#plt.xscale('log')
plot_loghist(sv1,100,label="1")
plot_loghist(sv2,100,label="2")
plt.legend()
plt.show()
'''
sv=[sv1,sv2]

file= os.path.join(os.path.join(os.path.dirname(__file__)), 'figure4c_stats.pickle')
with open(file, 'wb') as handle:
    pickle.dump(sv, handle, protocol=pickle.HIGHEST_PROTOCOL)
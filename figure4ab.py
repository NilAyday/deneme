import pickle
import copy
import random

import torch
import torch.nn as nn
import torch.nn.functional as func
import matplotlib.pyplot as plt
import numpy as np
from src import datasets, training, perturbed_dataloader
import os
import torchvision


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
        
def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)

class SubLoader(torchvision.datasets.CIFAR10):
    def __init__(self, *args, exclude_list=[], **kwargs):
        super(SubLoader, self).__init__(*args, **kwargs)

        if exclude_list == []:
            return

        if self.train:
            labels = np.array(self.targets)
            exclude = np.array(exclude_list).reshape(1, -1)
            mask = ~(labels.reshape(-1, 1) == exclude).any(axis=1)

            self.data = self.data[mask]
            self.targets = labels[mask].tolist()
        else:
            labels = np.array(self.targets)
            exclude = np.array(exclude_list).reshape(1, -1)
            mask = ~(labels.reshape(-1, 1) == exclude).any(axis=1)

            self.data = self.data[mask]
            self.targets = labels[mask].tolist()

num_epochs = 200
lr = 0.005
num_data = 50000
batch_size = 128

#ds_train = datasets.load_CIFAR10(True)
#ds_test = datasets.load_CIFAR10(False)


ds_train=SubLoader(exclude_list=[2,3,4,5,6,7,8,9,10],root="./datasets", train=True, download=True, transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
ds_test=SubLoader(exclude_list=[2,3,4,5,6,7,8,9,10],root="./datasets", train=False, download=True, transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))

'''
idx = np.where(ds_train.targets==6)#| (ds_train.targets==2) 
print(idx)
ds_train.targets = ds_train.targets[idx]
ds_train.data = ds_train.data[idx]
'''

indices_test = [i for i in range(len(ds_test))]
random.shuffle(indices_test)

#indices = torch.arange(num_data)
#ds_train = torch.utils.data.Subset(ds_train, indices)
#ds_test = torch.utils.data.Subset(ds_test, indices)
num_data=len(ds_train.targets)
dataset_train_30 = perturbed_dataloader.PerturbedDataset(ds_train, 0.3, size = num_data,num_classes = 2,enforce_false = False)
dataloader_train_30 = torch.utils.data.DataLoader(dataset_train_30, batch_size=batch_size, shuffle=True)
dataset_train_50 = perturbed_dataloader.PerturbedDataset(ds_train, 0.5, size = num_data,num_classes = 2,enforce_false = False)
dataloader_train_50 = torch.utils.data.DataLoader(dataset_train_50, batch_size=batch_size, shuffle=True)
dataset_test = torch.utils.data.Subset(ds_test, indices_test[:10000])
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=300, shuffle=False)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNet(ResidualBlock, [2, 2, 2]).to(device)
model.apply(initialize_weights)


optimizer = torch.optim.SGD(model.parameters(), lr = lr)
loss = torch.nn.CrossEntropyLoss()

s_30 = training.train(model, optimizer, loss, dataloader_train_30, dataloader_test, num_epochs, device=device)

model = ResNet(ResidualBlock, [2, 2, 2]).to(device)
model.apply(initialize_weights)
optimizer = torch.optim.SGD(model.parameters(), lr = lr)
loss = torch.nn.CrossEntropyLoss()

s_50 = training.train(model, optimizer, loss, dataloader_train_50, dataloader_test, num_epochs, device=device)


history=[s_30,s_50]

#file= os.path.join(os.path.join(os.path.dirname(__file__)), 'figure4c_stats.pickle')
#with open(file, 'wb') as handle:
 #   pickle.dump(sv, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
file= os.path.join(os.path.join(os.path.dirname(__file__)), 'figure4ab_stats.pickle')
with open(file, 'wb') as handle:
    pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)

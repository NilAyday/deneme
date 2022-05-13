import pickle
import copy

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from src import datasets, perturbed_dataloader, training_fig3
import os
import numpy as np


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = 5, stride = 2, padding = 0)
        #torch.nn.init.uniform_(self.conv1.weight, a=0.0, b=1.0)
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size = 5, stride = 2, padding = 0)
        self.linear1 = nn.Linear(512, 256)
        self.linear2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()

    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)

        x = x.reshape(x.shape[0], -1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)


ds_train = datasets.load_MNIST(True)
ds_test = datasets.load_MNIST(False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 1
lr = 0.005
alphas = [0,0.1,0.5,1]
num_data = 3000
batch_size = 10

stats = {}
for alpha in alphas:
    print("alpha", alpha)
    ds_train = perturbed_dataloader.PerturbedDataset(ds_train, alpha, size = num_data)
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size)
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=batch_size)

    model = LeNet()
    #model.apply(initialize_weights)
    #optimizer = torch.optim.SGD(model.parameters(), lr = lr)
    optimizer = torch.optim.Adadelta(model.parameters(), lr = lr)
    
    loss = torch.nn.CrossEntropyLoss()

    
    s = training_fig3.train(model, optimizer, loss, dl_train, dl_test, num_epochs, device=device)
    stats[alpha] = s
    
    file= os.path.join(os.path.join(os.path.dirname(__file__),"data"), 'figure1_stats.pickle')
    with open(file, 'wb') as handle:
        pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
  



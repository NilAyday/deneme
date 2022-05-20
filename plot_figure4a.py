import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def plot_loghist(x, bins,label):
    hist, bins = np.histogram(x, bins=bins)
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    plt.hist(x, bins=logbins,alpha=0.5,label=label)
    plt.xscale('log')

file= os.path.join(os.path.join(os.path.dirname(__file__)), 'figure4a_stats.pickle')
with open(file, 'rb') as handle:
    b = pickle.load(handle)

[s,s_pertub_1,s_pertub_2]=b

#print(b['loss'])
#print(np.array(b['loss']))
#print(type(b['loss']))
#loss=b['loss'].detach().numpy()
#print(np.shape(b['loss']))



'''
train_acc = []
true_train_acc = []
val_acc = []
loss=[]

for x_ in x:
    early_stop_time = torch.argmax(torch.tensor(b[x_]['val_acc'])).item()
    #print(x_, early_stop_time, b[x_]['val_acc'][early_stop_time])

    train_acc.append(b[x_]['train_acc'][early_stop_time])
    true_train_acc.append(b[x_]['true_train_acc'][early_stop_time])
    val_acc.append(b[x_]['val_acc'][early_stop_time])
    loss.append(b[x_]['loss'][early_stop_time])
'''

#plot_loghist(loss,100,label="At initialization")
#plot_loghist(sv2,100,label="After training")
plt.hist(np.array(s['loss']),bins=100,alpha=0.5,range=(1.7,2.4),label="clean data")
plt.hist(np.array(s_pertub_1['loss']),bins=100,alpha=0.5,range=(1.7,2.4),label="corrupted data")
plt.legend()
#plt.show()
file= os.path.join(os.path.join(os.path.dirname(__file__)), 'fig4a')
plt.savefig(file+".png")
plt.clf()
plt.hist(np.array(s['loss']),bins=100,alpha=0.5,range=(1.7,2.4),label="clean data")
plt.hist(np.array(s_pertub_2['loss']),bins=100,alpha=0.5,range=(1.7,2.4),label="corrupted data")
plt.legend()
#plt.show()
file= os.path.join(os.path.join(os.path.dirname(__file__)), 'fig4b')
plt.savefig(file+".png")

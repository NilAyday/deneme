from cProfile import label
import pickle
import copy

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os 

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

file= os.path.join(os.path.join(os.path.dirname(__file__),"data"), 'figure1_stats.pickle')
with open(file, 'rb') as handle:
    b = pickle.load(handle)

for i in list(b.keys()):
    percentage=str(round(i*100)) + '%'
    plt.plot(b[i]["distance"], b[i]["true_train_acc"], label="{} Corruption".format(percentage))
plt.legend()
plt.xlabel("Distance from initialization")
plt.ylabel("Training accuracy")
file= os.path.join(os.path.join(os.path.dirname(__file__),"data"), 'fig3a')
plt.savefig(file+".png")
plt.clf()

for i in list(b.keys()):
    percentage=str(round(i*100)) + '%'
    plt.plot(b[i]["distance"], b[i]["loss"], label="{} Corruption".format(percentage))
plt.legend()
plt.xlabel("Distance from initialization")
plt.ylabel("Loss")
file= os.path.join(os.path.join(os.path.dirname(__file__),"data"), 'fig3b')
plt.savefig(file+".png")

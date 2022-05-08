

from cProfile import label
import pickle
import copy

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

with open('./data/figure1_stats.pickle', 'rb') as handle:
    b = pickle.load(handle)

x = list(b.keys())
'''
for epoch in [100, 499]:
    train_acc = [b[x_]['train_acc'][epoch] for x_ in x]
    true_train_acc = [b[x_]['true_train_acc'][epoch] for x_ in x]
    val_acc = [b[x_]['val_acc'][epoch] for x_ in x]

    plt.plot(x, train_acc, label="train_acc")
    plt.plot(x, true_train_acc, label="true_train_acc")
    plt.plot(x, val_acc, label="val_acc")
    plt.legend()
    plt.savefig("./data/fig1_epoch"+str(epoch)+".png")
'''
print(x)
print("----")
print(b[0])
plt.plot(x*100, [b[x_]['train_acc'] for x_ in x], marker='o', label="Train accuracy")
plt.plot(x*100, [b[x_]['val_acc'] for x_ in x], marker='s', label="Test accuracy")
plt.plot(x*100, [b[x_]['true_train_acc'] for x_ in x], color='red', marker='v', 
         label="Train accuracy w.r.t. true labels")
plt.xlabel("Fraction of labels corrupted (%)")
plt.ylabel("Accuracy (%)")
plt.legend()

plt.savefig("./data/fig1"+".png")

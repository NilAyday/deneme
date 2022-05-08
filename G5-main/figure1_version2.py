import pickle
import copy
import random

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from src import datasets, perturbed_dataloader, training
import numpy as np
import tensorflow as tf

import os
import numpy as np


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


'''
ds_train = datasets.load_MNIST(True)
#indices = torch.arange(10000)
#ds_train = torch.utils.data.Subset(ds_train, indices)
ds_test = datasets.load_MNIST(False)
X_train = ds_train.data.numpy()
y_train = ds_train.targets.numpy()

X_test = ds_test.data.numpy()
y_test = ds_test.targets.numpy()
'''
dataset = tf.keras.datasets.mnist
(X_train, y_train),(X_test, y_test) = dataset.load_data()
X_train = X_train[...,np.newaxis]
X_test = X_test[...,np.newaxis]
y_train = np.squeeze(y_train)
y_test = np.squeeze(y_test)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


num_epochs = 200
lr = 0.02
num_data = 50000
batch_size = 100

alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

LeNet = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(64, 3, input_shape=X_train[0].shape, activation='relu'),
            tf.keras.layers.Conv2D(128, 3, activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
    ])

history = {}
history['train_acc'] = []
history['true_train_acc'] = []
history['val_acc'] = []

stats = {}
for alpha in alphas:
    print("alpha", alpha)
    corruption_sz = int(np.floor(alpha*len(y_train)))
    corruption_indices = np.random.choice(len(y_train ), size=corruption_sz, replace=False)
    y_train_crpt = y_train.copy()
    y_train_crpt[corruption_indices] = np.random.choice(y_train_crpt.max()+1, size=corruption_sz)
   
    LeNet.compile(optimizer='adadelta',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1)
    LeNet.fit(X_train, y_train_crpt, validation_data=(X_test, y_test), epochs=num_epochs, callbacks=[es])

    history['train_acc'].append(LeNet.evaluate(X_train, y_train_crpt)[1])
    history['true_train_acc'].append(LeNet.evaluate(X_train, y_train)[1])
    history['val_acc'].append(LeNet.evaluate(X_test, y_test)[1])

    stats[alpha] = history

    file= os.path.join(os.path.join(os.path.dirname(__file__),"data"), 'figure1_stats.pickle')
    with open(file, 'wb') as handle:
        pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)


    

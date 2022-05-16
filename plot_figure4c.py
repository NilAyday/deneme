import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

def plot_loghist(x, bins,label):
    hist, bins = np.histogram(x, bins=bins)
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    plt.hist(x, bins=logbins,alpha=0.5,label=label)
    plt.xscale('log')

file= os.path.join(os.path.join(os.path.dirname(__file__)), 'figure4c_stats.pickle')
with open(file, 'rb') as handle:
    b = pickle.load(handle)

[sv1,sv2]=b

plot_loghist(sv1,100,label="At initialization")
plot_loghist(sv2,100,label="After training")
plt.legend()
file= os.path.join(os.path.join(os.path.dirname(__file__)), 'fig4c')
plt.savefig(file+".png")
#plt.show()


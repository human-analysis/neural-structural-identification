# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 21:57:24 2022

@author: lixuy
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as ticker
plt.style.reload_library()
plt.style.use('science')
plt.style.use(['science','nature'])

xl = [0.0,0.0235294,0.0470588,0.0705882,0.0941176,0.117647,0.141176,0.164706,0.188235,0.211765,0.235294,0.258824,0.282353,0.305882,0.329412,0.352941,0.376471,0.4]
xl = xl[1:-1]

L = 0.4
tmax = 0.045
Nt =  160
tl = np.linspace(0,tmax,Nt)
tl2 = np.linspace(0,2*tmax,2*Nt)


y = np.loadtxt("data/y.txt", dtype='float32', delimiter='\t') * 1e3
pred1 = np.loadtxt("data/NeuralSI-pred.txt", dtype='float32', delimiter='\t') * 1e3
#print(y.shape)


fig, ax = plt.subplots(1,3,figsize = (18,3.8))

#------------------heatmap-true
ax = plt.subplot(1, 3, 1)
im = plt.imshow(y.T, cmap='jet', aspect='auto',extent=[0,L,tmax,0])
ax.invert_yaxis()
cbar = plt.colorbar()
plt.title("Ground Truth", fontsize=22,y=1.04)
plt.xlabel('x', fontsize=22)
plt.ylabel('t', fontsize=22)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.015))
cbar.ax.tick_params(labelsize=16)


#------------------heatmap-pred
ax = plt.subplot(1, 3, 2)
im = plt.imshow(y.T, cmap='jet', aspect='auto',extent=[0,L,tmax,0])
ax.invert_yaxis()
cbar = plt.colorbar()
plt.title("Prediction", fontsize=22,y=1.04)
plt.xlabel('x', fontsize=22)
plt.ylabel('t', fontsize=22)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.015))
cbar.ax.tick_params(labelsize=16)


#------------------heatmap-error
ax = plt.subplot(1, 3, 3)
im = plt.imshow(abs(pred1-y).T, cmap='jet', aspect='auto',extent=[0,L,tmax,0])
ax.invert_yaxis()
cbar = plt.colorbar()
plt.title("Error", fontsize=22,y=1.04)
plt.xlabel('x', fontsize=22)
plt.ylabel('t', fontsize=22)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.015))
cbar.ax.tick_params(labelsize=16)

#fig.tight_layout()

plt.savefig('pdf/heatmap1.pdf')


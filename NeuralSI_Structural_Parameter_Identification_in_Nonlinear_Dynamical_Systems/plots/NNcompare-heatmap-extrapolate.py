# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 22:24:05 2022

@author: lixuy
"""



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as ticker
plt.style.reload_library()
plt.style.use('science')
plt.style.use(['science','nature'])

L = 0.4
tmax = 0.045
Nt =  160


y = np.loadtxt("data/y.txt", dtype='float32', delimiter='\t') * 1e3
y2 = np.loadtxt("data/y-extrapolate.txt", dtype='float32', delimiter='\t') * 1e3
pred1 = np.loadtxt("data/NeuralSI-pred.txt", dtype='float32', delimiter='\t') * 1e3
pred2 = np.loadtxt("data/NeuralSI-pred2.txt", dtype='float32', delimiter='\t') * 1e3
DNN1 = np.load("data/DNN-pred.npy")
DNN2 = np.load("data/DNN-pred2.npy")
PINN1 = np.load("data/PINN-pred.npy")
PINN2 = np.load("data/PINN-pred2.npy")





fig, ax = plt.subplots(2,3,figsize = (16,8))

#DNN

#------------------heatmap-pred
ax = plt.subplot(2, 3, 1)
im = plt.imshow(DNN2.T, cmap='jet', aspect='auto',extent=[0,L,tmax,0])
ax.invert_yaxis()
cbar = plt.colorbar()
plt.title("DNN Extrapolation", fontsize=22,y=1.04)
plt.xlabel('x', fontsize=22)
plt.ylabel('t', fontsize=22)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.01))
cbar.ax.tick_params(labelsize=16)


#PINN
#------------------heatmap-pred
ax = plt.subplot(2, 3, 2)
im = plt.imshow(PINN2.T, cmap='jet', aspect='auto',extent=[0,L,tmax,0])
ax.invert_yaxis()
cbar = plt.colorbar()
plt.title("PINN Extrapolation", fontsize=22,y=1.04)
plt.xlabel('x', fontsize=22)
plt.ylabel('t', fontsize=22)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.01))
cbar.ax.tick_params(labelsize=16)


#our
#------------------heatmap-pred
ax = plt.subplot(2, 3, 3)
im = plt.imshow(pred2.T, cmap='jet', aspect='auto',extent=[0,L,tmax,0])
ax.invert_yaxis()
cbar = plt.colorbar()
plt.title("NeuralSI Extrapolation", fontsize=22,y=1.04)
plt.xlabel('x', fontsize=22)
plt.ylabel('t', fontsize=22)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.01))
cbar.ax.tick_params(labelsize=16)




#------------------heatmap-error
ax = plt.subplot(2, 3, 4)
im = plt.imshow(abs(DNN2-y2).T, cmap='jet', aspect='auto',extent=[0,L,2*tmax,0])
ax.invert_yaxis()
cbar = plt.colorbar()
plt.title("DNN Error", fontsize=22,y=1.04)
plt.xlabel('x', fontsize=22)
plt.ylabel('t', fontsize=22)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.03))
cbar.ax.tick_params(labelsize=16)


#------------------heatmap-error
ax = plt.subplot(2, 3, 5)
im = plt.imshow(abs(PINN2-y2).T, cmap='jet', aspect='auto',extent=[0,L,2*tmax,0])
ax.invert_yaxis()
cbar = plt.colorbar()
#plt.title("Error extrapolation", fontsize=22,y=1.04)
plt.title("PINN Error", fontsize=22,y=1.04)
plt.xlabel('x', fontsize=22)
plt.ylabel('t', fontsize=22)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.03))
cbar.ax.tick_params(labelsize=16)


#------------------heatmap-error
ax = plt.subplot(2, 3, 6)
im = plt.imshow(abs(pred2-y2).T, cmap='jet', aspect='auto',extent=[0,L,2*tmax,0])
ax.invert_yaxis()
cbar = plt.colorbar()
#plt.title("Error extrapolation", fontsize=22,y=1.04)
plt.title("NeuralSI Error", fontsize=22,y=1.04)
plt.xlabel('x', fontsize=22)
plt.ylabel('t', fontsize=22)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.03))
cbar.ax.tick_params(labelsize=16)


fig.tight_layout(w_pad=-1)

#plt.subplots_adjust(left=0,
#                    bottom=0, 
#                    right=0.95, 
#                    top=1, 
#                    wspace=0.26, 
#                    hspace=0.0)

plt.savefig('pdf/heatmap-method-extrapolate.pdf')





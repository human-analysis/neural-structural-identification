# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 22:52:51 2022

@author: lixuy
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as ticker
plt.style.reload_library()
plt.style.use('science')
plt.style.use(['science','nature'])


fig, ax = plt.subplots(1, 3, figsize = (18,4.5))

ax = plt.subplot(1, 3, 1)
xlist = np.arange(3, 8, 1)
data1 = np.loadtxt("tasks-hyperparameters2\layers\e1.txt", dtype='float32', delimiter='\t')
data2 = np.loadtxt("tasks-hyperparameters2\layers\e2.txt", dtype='float32', delimiter='\t')
ax.plot(xlist, data1, '-h', markersize=14, linewidth=2, label='MAE')
ax.plot(xlist, data2, '--^', markersize=14, linewidth=2, label='MAE Extrapolation')
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.ylabel('Error', fontsize=24)
plt.xlabel('Number of layers', fontsize=24)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.xticks(fontsize=16)
ax.yaxis.get_offset_text().set_fontsize(16)
plt.yticks(fontsize=16)
plt.legend(fontsize=16,frameon=False)
plt.title('(i)',fontsize=24,y=1.04)
plt.grid('on')


ax = plt.subplot(1, 3, 2)
xlist = np.arange(0.1, 1.1, 0.1)
data1 = np.loadtxt("tasks-hyperparameters2\sample_ratio\e1.txt", dtype='float32', delimiter='\t')
data2 = np.loadtxt("tasks-hyperparameters2\sample_ratio\e2.txt", dtype='float32', delimiter='\t')
ax.plot(xlist, data1, '-h', markersize=14, linewidth=2, label='MAE')
ax.plot(xlist, data2, '--^', markersize=14, linewidth=2, label='MAE Extrapolation')
plt.ylabel('Error', fontsize=24)
plt.xlabel('Sample ratio', fontsize=24)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
plt.xticks(fontsize=16)
ax.yaxis.get_offset_text().set_fontsize(16)
plt.yticks(fontsize=16)
plt.legend(fontsize=16,frameon=False)
plt.title('(ii)',fontsize=22,y=1.04)
plt.grid('on')


ax = plt.subplot(1, 3, 3)
xlist = [8,16,32,64,128,256,512]
data1 = np.loadtxt("tasks-hyperparameters2\minibatch_size\e1.txt", dtype='float32', delimiter='\t')
data2 = np.loadtxt("tasks-hyperparameters2\minibatch_size\e2.txt", dtype='float32', delimiter='\t')
ax.set_yscale('log')
ax.set_xscale('log',base=2) 
ax.plot(xlist, data1, '-h', markersize=14, linewidth=2, label='MAE')
ax.plot(xlist, data2, '--^', markersize=14, linewidth=2, label='MAE Extrapolation')
plt.ylabel('Error', fontsize=24)
plt.xlabel('Minibatch size', fontsize=24)
ax.set_xticklabels(['','','8','16','32','64','128','256','512'],fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=16,frameon=False)
plt.title('(iii)',fontsize=24,y=1.04)
plt.grid('on')


plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.65, 
                    top=0.9, 
                    wspace=0.5, 
                    hspace=0.4)
fig.tight_layout()

plt.savefig('pdf/MAE-error-hyperparameter.pdf')



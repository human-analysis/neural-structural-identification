# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 21:16:46 2022

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
#fig, ax = plt.subplots(1, 3, figsize = (16,4.5))
#fig, ax = plt.subplots(1, 3, figsize = (15,4))
ax1 = plt.subplot(1, 3, 1)
xlist = np.arange(3, 8, 1)
dp = np.loadtxt("tasks-hyperparameters2\layers\A.txt", dtype='float32', delimiter='\t')
dc = np.loadtxt("tasks-hyperparameters2\layers\B.txt", dtype='float32', delimiter='\t')
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
ax2 = ax1.twinx()
plt.yticks(fontsize=16)
lns1 = ax1.plot(xlist, dp, 'b-h', markersize=10, linewidth=2, label='Modulus coefficient P')
lns2 = ax2.plot(xlist, dc, 'g-^', markersize=10, linewidth=2, label='Damping C')
ax1.plot(xlist, dp, 'b-h', markersize=14, linewidth=2, label='Modulus P')
ax2.plot(xlist, dc, 'g--^', markersize=14, linewidth=2, label='Damping C')
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
ax1.set_xlabel('Number of layers', fontsize=24)
leg = lns1 + lns2
labs = [l.get_label() for l in leg]
ax1.legend(leg, labs, loc=0, frameon=False, fontsize=16)
ax1.set_ylabel('Fréchet distance ', fontsize=24)
plt.yticks(fontsize=16)
plt.title('(i)',fontsize=24,y=1.04)
plt.grid('on')



ax1 = plt.subplot(1, 3, 2)
xlist = np.arange(0.1, 1.1, 0.1)
dp = np.loadtxt("tasks-hyperparameters2\sample_ratio\A.txt", dtype='float32', delimiter='\t')
dc = np.loadtxt("tasks-hyperparameters2\sample_ratio\B.txt", dtype='float32', delimiter='\t')
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
ax1.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax2 = ax1.twinx()
plt.yticks(fontsize=16)
ax1.plot(xlist, dp, 'b-h', markersize=14, linewidth=2, label='Modulus P')
ax2.plot(xlist, dc, 'g--^', markersize=14, linewidth=2, label='Damping C')
ax1.set_xlabel('Sample ratio', fontsize=24)
leg = lns1 + lns2
labs = [l.get_label() for l in leg]
ax1.legend(leg, labs, loc="upper right", frameon=False, fontsize=16)
ax1.set_ylabel('Fréchet distance ', fontsize=24)
plt.yticks(fontsize=16)
plt.title('(ii)',fontsize=24,y=1.04)
plt.grid('on')



ax1 = plt.subplot(1, 3, 3)
xlist = [8,16,32,64,128,256,512]
dp = np.loadtxt("tasks-hyperparameters2\minibatch_size\A.txt", dtype='float32', delimiter='\t')
dc = np.loadtxt("tasks-hyperparameters2\minibatch_size\B.txt", dtype='float32', delimiter='\t')
plt.yticks(fontsize=16)
ax2 = ax1.twinx()
ax1.set_xscale('log', base=2)
lns1 = ax1.plot(xlist, dp, 'b-h', markersize=14, linewidth=2, label='Modulus coefficient P')
lns2 = ax2.plot(xlist, dc, 'g--^', markersize=14, linewidth=2, label='Damping C')
ax1.set_xlabel('Minibatch size', fontsize=24)
ax1.set_ylabel('Fréchet distance ', fontsize=24)
ax1.set_xticklabels(['','','8','16','32','64','128','256','512'],fontsize=16)
leg = lns1 + lns2
labs = [l.get_label() for l in leg]
ax1.legend(leg, labs, loc="upper left", frameon=False, fontsize=16)
plt.yticks(fontsize=16)
plt.title('(iii)',fontsize=24,y=1.04)
plt.grid('on')


plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.65, 
                    top=0.9, 
                    wspace=0.5, 
                    hspace=0.4)
fig.tight_layout()

plt.savefig('pdf/distance-hyperparameter.pdf')



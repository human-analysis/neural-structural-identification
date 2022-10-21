# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 23:58:53 2022

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

y = np.loadtxt("data/y-high-resolution.txt", dtype='float32', delimiter='\t') * 1e3
#print(y.shape)

#------------------heatmap-true

fig,ax = plt.subplots(figsize = (7,5))
im = plt.imshow(y.T, cmap='jet', aspect='auto',extent=[0,L,tmax,0])
ax.invert_yaxis()
cbar = plt.colorbar()
cbar.remove() 
#plt.title("Ground Truth", fontsize=22,y=1.04)
#plt.xlabel('x', fontsize=22)
#plt.ylabel('t', fontsize=22)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

px = np.array([0,0.4,0.4,0]);
py = np.array([0,0,0.045,0.045]);
plt.fill(px,py,color='white', alpha=0.5)

ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([])


xl = np.linspace(0,L,60)
tl = np.linspace(0,tmax,500)
x = np.array([17,  45,  15,   3,  22,   6,  25,  34]) 
t = np.array([12, 29, 37, 31,  6, 44,  7, 41,])

plt.scatter(np.array(xl)[x], np.array(tl)[t*10], s=500, alpha=1, marker='*',
            label = 'Training Samples', color='k')

#plt.legend('Training Samples',fontsize=16,loc=)
legend = ax.legend(bbox_to_anchor=(0.04, 1.0),fontsize=24,frameon=True)
legend.get_frame().set_linewidth(1)
legend.get_frame().set_edgecolor("grey")

#plt.savefig('triaing-samples.png',dpi=2000)





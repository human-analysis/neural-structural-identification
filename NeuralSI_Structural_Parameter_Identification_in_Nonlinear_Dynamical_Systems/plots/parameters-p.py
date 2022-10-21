# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 21:42:26 2022

@author: lixuy
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as ticker
plt.style.reload_library()
plt.style.use('science')
plt.style.use(['science','nature'])



# the data was before the minor change of the element
v = np.loadtxt("data/p-nonlinear.txt", dtype='float32', delimiter='\t')
v0 = np.loadtxt("data/p0-nonlinear.txt", dtype='float32', delimiter='\t')


xl = [0.0,0.0235294,0.0470588,0.0705882,0.0941176,0.117647,0.141176,0.164706,0.188235,0.211765,0.235294,0.258824,0.282353,0.305882,0.329412,0.352941,0.376471,0.4]
xl = xl[1:-1]

vp0 = v0[0:16]
vc0 = v0[16:]

vp = v[:16]
vc = v[16:]


#
#
#fig, ax = plt.subplots(2,1,figsize = (7,5.5))
#fig.tight_layout(h_pad=8) 
#
#ax = plt.subplot(2, 1, 1)
#plt.plot(xl, vp0, '-h', markersize=14, linewidth=1.5, label='$P_{0}(x)$')
#plt.xlabel('x', fontsize=22)
#plt.xticks(fontsize=16)
#plt.yticks(fontsize=16)
#plt.title('Modulus coefficient P', fontsize=22,y=1.04)
#plt.legend(fontsize=16)
#ax.axes.yaxis.set_ticklabels([])
#
#ax = plt.subplot(2, 1, 2)
#plt.plot(xl, vc0, '-^', markersize=14, linewidth=1.5, label='$C_{0}(x)$')
#plt.xlabel('x', fontsize=22)
#plt.xticks(fontsize=16)
#plt.yticks(fontsize=16)
#plt.title('Damping C', fontsize=22,y=1.04)
#plt.legend(fontsize=16)
#plt.savefig('V0.pdf')
#ax.axes.yaxis.set_ticklabels([])
#


xl = [0.0,0.0235294,0.0470588,0.0705882,0.0941176,0.117647,0.141176,0.164706,0.188235,0.211765,0.235294,0.258824,0.282353,0.305882,0.329412,0.352941,0.376471,0.4]
xl = xl[1:-1]


fig, ax = plt.subplots(2,1,figsize = (9,6))
fig.tight_layout(h_pad=10) 

ax = plt.subplot(2, 1, 1)
plt.plot(xl, vp0, '-h', markersize=14, linewidth=1.5, label='Ground Truth')
plt.plot(xl, vp, '--h', markersize=14, linewidth=1.5, label='Prediction')
plt.xlabel('x', fontsize=22)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title('Modulus coefficient P', fontsize=22,y=1.04)
plt.legend(fontsize=16)
plt.grid('on')
ax.axes.yaxis.set_ticklabels([])

ax = plt.subplot(2, 1, 2)
plt.plot(xl, vc0, '-^', markersize=14, linewidth=1.5, label='Ground Truth')
plt.plot(xl, vc, '--^', markersize=14, linewidth=1.5, label='Prediction')
plt.xlabel('x', fontsize=22)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title('Damping C', fontsize=22,y=1.04)
plt.legend(fontsize=16)
plt.grid('on')
ax.axes.yaxis.set_ticklabels([])

plt.savefig('pdf/V.pdf')




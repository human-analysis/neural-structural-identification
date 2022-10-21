# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 23:40:35 2022

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
tl = np.linspace(0,tmax,Nt)
tl2 = np.linspace(0,2*tmax,2*Nt)


y = np.loadtxt("data/y.txt", dtype='float32', delimiter='\t') * 1e3
y2 = np.loadtxt("data/y-extrapolate.txt", dtype='float32', delimiter='\t') * 1e3
pred1 = np.loadtxt("data/NeuralSI-pred.txt", dtype='float32', delimiter='\t') * 1e3
pred2 = np.loadtxt("data/NeuralSI-pred2.txt", dtype='float32', delimiter='\t') * 1e3



fig, ax = plt.subplots(1,2,figsize = (16,4.5))
# single element 
element = 8
ax = plt.subplot(1, 2, 1)
plt.plot(tl2,y2[element,:], linewidth=3, label = 'Ground Truth')
plt.plot(tl2,pred2[element,:], '--',  linewidth=3, label = 'Prediction')
plt.xlim([0,0.094])
plt.axvspan(0.045, 0.093, facecolor='grey', alpha=0.4 ,label = 'Extrapolation')
plt.legend(fontsize=16,loc="upper right",frameon=False)
plt.xticks(fontsize=16)
plt.title("(a)", fontsize=22, y=1.04)
plt.yticks(fontsize=16)
plt.ylabel('u', fontsize=22)
plt.xlabel('t', fontsize=22)
plt.grid('on')

element = 4
ax = plt.subplot(1, 2, 2)
plt.plot(tl2,y2[element,:], linewidth=3, label = 'Ground Truth')
plt.plot(tl2,pred2[element,:], '--',  linewidth=3, label = 'Prediction')
plt.xlim([0,0.094])
plt.axvspan(0.045, 0.093, facecolor='grey', alpha=0.4 ,label = 'Extrapolation')
plt.legend(fontsize=16,loc="upper right",frameon=False)
plt.xticks(fontsize=16)
plt.title("(b)", fontsize=22, y=1.04)
plt.yticks(fontsize=16)
plt.ylabel('u', fontsize=22)
plt.xlabel('t', fontsize=22)
#fig.tight_layout()
plt.grid('on')
plt.savefig('pdf/single-element.pdf')




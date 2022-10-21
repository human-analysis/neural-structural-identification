# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 22:20:32 2022

@author: lixuy
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as ticker
plt.style.reload_library()
plt.style.use('science')
plt.style.use(['science','nature'])



fig, ax = plt.subplots(1,3,figsize = (19,6.2))


ax = plt.subplot(1, 3, 1)
#---------------DNN and PINN
#Neural
error1 = 8.0544713e-05
error2 = 1.0088530e-04

#DNN
e1 = 5.136466
e2 = 22.47875

#PINN
p1 = 0.34420902
p2 = 114.54138

Error1 = [e1, p1, error1]
Error2 = [e2, p2, error2]

barWidth = 0.21
width = 0.2

br1 = np.arange(len(Error1))
br2 = [x + barWidth for x in br1]


#fig,ax = plt.subplots(figsize =(10, 7))
plt.bar(br1, Error1, edgecolor ='grey', #color ='maroon',
        width = width, label = 'Interpolation')
plt.bar(br2, Error2, edgecolor ='grey', #color ='b',
        width = width, label = 'Extrapolation')
ax.set_yscale('log')
plt.ylabel('MAE', fontsize = 22)
plt.yticks(fontsize=16)
plt.xticks([r + barWidth/2 for r in range(len(Error1))],
        ['DNN', 'PINN', 'NeuralSI'], fontsize = 22)
plt.title('(a)',fontsize = 22, y=1.04)
plt.legend(fontsize = 16,loc="upper left")






ax = plt.subplot(1, 3, 2)


#DNN
t1 = 0.0009240007400512695
t12 =  0.0010320591926574708

#PINN
t2 = 0.0010540003776550292
t22 = 0.0010340795516967773

#our
t3 = 0.0156 #0.111 #0.135658
t32 = 0.018 #0.241 #0.316877


Error1 = [t1, t2, t3]
Error2 = [t12, t22, t32]

barWidth = 0.21
width = 0.2

br1 = np.arange(len(Error1))
br2 = [x + barWidth for x in br1]


plt.bar(br1,np.array(Error2),edgecolor ='grey', #color ='maroon',
        width = width, label = 'Interpolation')
ax.set_yscale('log')
#plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.ylabel('Time', fontsize = 22)
plt.yticks(fontsize=16)
plt.xticks([r for r in range(len(Error1))],
        ['DNN', 'PINN', 'NeuralSI'], fontsize = 22)
plt.title('(b)',fontsize = 22, y=1.04)
#plt.legend(fontsize = 16)






ax = plt.subplot(1, 3, 3)


samples = 16*160
samples2 = 16*160
labels = ['DNN', 'PINN', 'NeuralSI']

x = np.array([t1, t2, t3])/samples
y = np.array([e1, p1, error1])
plt.scatter(x, y, s=400, alpha=1,marker='*',
            label = 'Interpolation')



x2 = np.array([t12, t22, t32])/samples2
y2 = np.array([e2, p2, error2])
plt.scatter(x2, y2, s=400, alpha=1,marker='*',
            label = 'Extrapolation')

ax.set_yscale('log')



i = 0
ax.annotate(labels[i], (x[i]+2e-7, y[i]-1.3), fontsize=20)
ax.annotate(labels[i], (x2[i]+2e-7, y2[i]-6), fontsize=20)
i = 1
ax.annotate(labels[i], (x[i]+2e-7, y[i]-0.1), fontsize=20)
ax.annotate(labels[i], (x2[i]+2e-7, y2[i]-33), fontsize=20)
i = 2
ax.annotate(labels[i], (x[i]-1.8e-6, y[i]-2e-5), fontsize=20)
ax.annotate(labels[i], (x2[i]-1.3e-6, y2[i]+11e-5), fontsize=20)



#plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.yaxis.get_offset_text().set_fontsize(16)
ax.xaxis.get_offset_text().set_fontsize(16)
plt.xlabel('Time/samples', fontsize = 22)
plt.ylabel('MAE', fontsize = 22)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title('(c)',fontsize = 22, y=1.04)
plt.legend(fontsize = 16)






plt.tight_layout(w_pad=5)

plt.savefig('pdf/methods-compare.pdf')




